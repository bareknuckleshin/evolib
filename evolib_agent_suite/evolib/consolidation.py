from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evolib_agent_suite.evolib.prompts import LLM_MERGE_PROMPT, LLM_MERGE_SYSTEM_PROMPT
from evolib_agent_suite.utils import cosine, extract_json_block, hashed_embedding


@dataclass
class ConsolidationConfig:
    enabled: bool = True
    similarity_threshold: float = 0.88
    candidate_top_n: int = 1
    merge_strategy: str = "replace_if_longer"
    score_policy: str = "ema_score"
    allow_cross_type_merge: bool = False
    merge_history_limit: int = 20
    max_replace_content_chars: int = 1200
    ema_decay: float = 0.85


class LLMMerger:
    """Merge two library entries with an LLM and return cleaned fields."""

    def __init__(self, llm: Any, max_context_chars: int = 6000) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars

    def merge(self, existing: Any, candidate: Any, task_context: str = "", score: float = 0.0) -> Dict[str, Any]:
        prompt = LLM_MERGE_PROMPT.format(
            existing_type=getattr(existing, "type", "entry"),
            existing_title=getattr(existing, "title", ""),
            existing_content=self._trim(getattr(existing, "content", "")),
            existing_tags=", ".join(getattr(existing, "tags", []) or []),
            candidate_type=getattr(candidate, "type", "entry"),
            candidate_title=getattr(candidate, "title", ""),
            candidate_content=self._trim(getattr(candidate, "content", "")),
            candidate_tags=", ".join(getattr(candidate, "tags", []) or []),
            task_context=self._trim(task_context),
            score=f"{score:.3f}",
        )
        raw = self.llm.generate(LLM_MERGE_SYSTEM_PROMPT, prompt, max_tokens=1000)
        parsed = extract_json_block(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM merger did not return a JSON object")
        return {
            "title": str(parsed.get("title") or getattr(existing, "title", "")).strip()[:160],
            "content": str(parsed.get("content") or getattr(existing, "content", "")).strip(),
            "tags": self._clean_tags(parsed.get("tags")),
            "rationale": str(parsed.get("rationale", "")).strip(),
        }

    def _trim(self, text: str) -> str:
        text = text or ""
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + " ...[truncated]"

    def _clean_tags(self, tags: Any) -> List[str]:
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        if not isinstance(tags, list):
            return []
        return [str(t).strip() for t in tags if str(t).strip()][:12]


class ConsolidationPolicy:
    def __init__(self, config: Optional[ConsolidationConfig] = None, llm_merger: Optional[LLMMerger] = None) -> None:
        self.config = config or ConsolidationConfig()
        self.llm_merger = llm_merger

    def find_target(self, entries: Sequence[Any], candidate: Any) -> Optional[Tuple[Any, float]]:
        if not self.config.enabled:
            return None
        matches: List[Tuple[Any, float]] = []
        for entry in entries:
            if not self.config.allow_cross_type_merge and getattr(entry, "type", None) != getattr(candidate, "type", None):
                continue
            sim = cosine(getattr(candidate, "embedding", []), getattr(entry, "embedding", []))
            if sim >= self.config.similarity_threshold:
                matches.append((entry, sim))
        matches.sort(key=lambda item: item[1], reverse=True)
        top_n = max(1, int(self.config.candidate_top_n))
        return matches[:top_n][0] if matches else None

    def merge(self, entry: Any, candidate: Any, similarity: float, task_id: str, score: float, parents: Sequence[str], task_context: str = "") -> Any:
        old_score = float(getattr(entry, "score_ema", 0.0))
        strategy = self.config.merge_strategy
        rationale = ""

        entry.updated_at = time.time()
        entry.source_task_ids = list(dict.fromkeys(getattr(entry, "source_task_ids", []) + [task_id]))
        entry.parents = list(dict.fromkeys(getattr(entry, "parents", []) + list(parents)))
        entry.tags = list(dict.fromkeys((getattr(entry, "tags", []) or []) + (getattr(candidate, "tags", []) or [])))

        if strategy == "keep_existing":
            pass
        elif strategy == "replace_if_longer":
            if len(getattr(candidate, "content", "")) > len(getattr(entry, "content", "")) and len(getattr(candidate, "content", "")) < self.config.max_replace_content_chars:
                entry.content = candidate.content
                entry.title = candidate.title or entry.title
        elif strategy == "append_summary":
            addition = getattr(candidate, "content", "").strip()
            if addition and addition not in entry.content:
                entry.content = f"{entry.content.rstrip()}\n\nMerged note: {addition}"[: self.config.max_replace_content_chars]
        elif strategy == "llm_merge":
            if self.llm_merger is None:
                raise ValueError("merge_strategy='llm_merge' requires an LLMMerger")
            merged = self.llm_merger.merge(entry, candidate, task_context=task_context, score=score)
            entry.title = merged["title"] or entry.title
            entry.content = merged["content"] or entry.content
            entry.tags = list(dict.fromkeys(entry.tags + merged["tags"]))
            rationale = merged.get("rationale", "")
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")

        entry.embedding = hashed_embedding(entry.text)
        entry.score_ema = self._score(old_score, float(score), float(similarity), int(getattr(entry, "uses", 0)))
        entry.metadata["last_merge_similarity"] = similarity
        self._append_merge_history(entry, candidate, similarity, task_id, score, strategy, old_score, entry.score_ema, rationale)
        return entry

    def _score(self, old: float, new: float, similarity: float, uses: int) -> float:
        policy = self.config.score_policy
        if policy == "ema_score":
            return self.config.ema_decay * old + (1.0 - self.config.ema_decay) * new
        if policy == "max_score":
            return max(old, new)
        if policy == "weighted_by_similarity":
            w = max(0.0, min(1.0, similarity))
            return old * w + new * (1.0 - w)
        if policy == "weighted_by_usage":
            w = uses / (uses + 1.0) if uses > 0 else 0.0
            return old * w + new * (1.0 - w)
        raise ValueError(f"Unsupported score policy: {policy}")

    def _append_merge_history(self, entry: Any, candidate: Any, similarity: float, task_id: str, score: float, strategy: str, old_score: float, new_score: float, rationale: str) -> None:
        history = entry.metadata.setdefault("merge_history", [])
        history.append({
            "at": time.time(),
            "task_id": task_id,
            "candidate_id": getattr(candidate, "id", None),
            "candidate_title": getattr(candidate, "title", ""),
            "candidate_type": getattr(candidate, "type", ""),
            "similarity": similarity,
            "score": score,
            "score_before": old_score,
            "score_after": new_score,
            "strategy": strategy,
            "rationale": rationale,
        })
        limit = max(1, int(self.config.merge_history_limit))
        entry.metadata["merge_history"] = history[-limit:]
