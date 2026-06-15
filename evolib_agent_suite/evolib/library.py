from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, derive_seed
from evolib_agent_suite.utils import clamp, cosine, hashed_embedding


@dataclass
class LibraryEntry:
    id: str
    type: str  # "skill" or "insight"
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    embedding: List[float] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    uses: int = 0
    wins: int = 0
    ig_ema: float = 0.0
    future_ig_ema: float = 0.0
    score_ema: float = 0.0
    source_task_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.type}: {self.title}\n{self.content}\nTags: {', '.join(self.tags)}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LibraryEntry":
        return cls(**data)


class EvolvingLibrary:
    """Persistent EvoLib-style library.

    This implements the practical version used by the runner:
    - retrieve by embedding similarity, then weighted sampling;
    - add/merge new skills and insights;
    - update weights with immediate information gain and two-hop future gain.

    The exact estimator in the paper uses multiple conditional/baseline samples.
    For one-pass agentic evaluation, this implementation uses the previous running
    score as the baseline and propagates positive score deltas to retrieved parents.
    """

    def __init__(
        self,
        path: str | Path,
        similarity_merge_threshold: float = 0.88,
        retrieval_similarity_threshold: float = 0.05,
        seed: int = 0,
        sampling_config: Optional[SamplingConfig] = None,
        alpha_ig: float = 1.0,
        beta_future_ig: float = 0.7,
        ema_decay: float = 0.85,
    ) -> None:
        self.path = Path(path)
        self.similarity_merge_threshold = similarity_merge_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold
        self.alpha_ig = alpha_ig
        self.beta_future_ig = beta_future_ig
        self.ema_decay = ema_decay
        self.seed = int(seed)
        if sampling_config is None:
            sampling_config = SamplingConfig(seed=self.seed)
        elif sampling_config.seed == 0 and self.seed != 0:
            sampling_config.seed = self.seed
        self.sampling_policy = SamplingPolicy(sampling_config)
        self.entries: Dict[str, LibraryEntry] = {}
        self.stats: Dict[str, Any] = {"episodes": 0, "score_ema": 0.0, "score_sum": 0.0}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.stats = data.get("stats", self.stats)
        self.entries = {e["id"]: LibraryEntry.from_dict(e) for e in data.get("entries", [])}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stats": self.stats,
            "entries": [entry.to_dict() for entry in self.entries.values()],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.entries)

    def all_entries(self) -> List[LibraryEntry]:
        return list(self.entries.values())

    def get(self, entry_id: str) -> Optional[LibraryEntry]:
        return self.entries.get(entry_id)

    def _make_entry(
        self,
        item: Dict[str, Any],
        parents: Sequence[str],
        task_id: str,
        score: float,
    ) -> LibraryEntry:
        typ = item.get("type", "insight").strip().lower()
        if typ not in {"skill", "insight"}:
            typ = "insight"
        title = (item.get("title") or typ.title()).strip()[:160]
        content = (item.get("content") or item.get("description") or "").strip()
        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        text = f"{typ}: {title}\n{content}\n{', '.join(tags)}"
        return LibraryEntry(
            id=str(uuid.uuid4())[:12],
            type=typ,
            title=title,
            content=content,
            tags=list(tags),
            weight=1.0,
            embedding=hashed_embedding(text),
            parents=list(dict.fromkeys(parents)),
            score_ema=score,
            source_task_ids=[task_id],
        )

    def _find_merge_target(self, candidate: LibraryEntry) -> Optional[Tuple[LibraryEntry, float]]:
        best: Optional[Tuple[LibraryEntry, float]] = None
        for entry in self.entries.values():
            if entry.type != candidate.type:
                continue
            sim = cosine(candidate.embedding, entry.embedding)
            if sim >= self.similarity_merge_threshold and (best is None or sim > best[1]):
                best = (entry, sim)
        return best

    def add_or_merge_many(
        self,
        candidates: Sequence[Dict[str, Any]],
        parents: Sequence[str],
        task_id: str,
        score: float,
    ) -> List[str]:
        new_or_updated: List[str] = []
        for item in candidates:
            candidate = self._make_entry(item, parents=parents, task_id=task_id, score=score)
            if not candidate.content:
                continue
            target = self._find_merge_target(candidate)
            if target is not None:
                entry, sim = target
                entry.updated_at = time.time()
                entry.source_task_ids = list(dict.fromkeys(entry.source_task_ids + [task_id]))
                entry.parents = list(dict.fromkeys(entry.parents + list(parents)))
                entry.tags = list(dict.fromkeys(entry.tags + candidate.tags))
                # Keep the more general/longer wording, but avoid unbounded growth.
                if len(candidate.content) > len(entry.content) and len(candidate.content) < 1200:
                    entry.content = candidate.content
                    entry.title = candidate.title or entry.title
                    entry.embedding = hashed_embedding(entry.text)
                entry.score_ema = self._ema(entry.score_ema, score)
                entry.metadata["last_merge_similarity"] = sim
                new_or_updated.append(entry.id)
            else:
                self.entries[candidate.id] = candidate
                for parent_id in candidate.parents:
                    parent = self.entries.get(parent_id)
                    if parent and candidate.id not in parent.children:
                        parent.children.append(candidate.id)
                new_or_updated.append(candidate.id)
        return list(dict.fromkeys(new_or_updated))

    def retrieve(
        self,
        query: str,
        k_skills: int = 4,
        k_insights: int = 4,
        sample: bool = True,
        task_id: Optional[str] = None,
        episode_id: Optional[Any] = None,
    ) -> List[LibraryEntry]:
        if not self.entries:
            return []
        q_emb = hashed_embedding(query)
        scored: List[Tuple[LibraryEntry, float, float]] = []
        for entry in self.entries.values():
            sim = cosine(q_emb, entry.embedding)
            if sim >= self.retrieval_similarity_threshold:
                # Similarity gates relevance; weight controls exploit/explore.
                retrieval_weight = max(1e-6, (0.2 + sim) * max(entry.weight, 1e-6))
                scored.append((entry, sim, retrieval_weight))
        if not scored:
            return []
        selected: List[LibraryEntry] = []
        sample_meta: Dict[str, Any] = {}
        for typ, k in [("skill", k_skills), ("insight", k_insights)]:
            group = [(e, sim, w) for (e, sim, w) in scored if e.type == typ]
            group.sort(key=lambda x: (x[1] * x[2], x[1]), reverse=True)
            group = group[: max(k * 4, k)]
            context = f"retrieval:{task_id or 'unknown-task'}:{episode_id or 'unknown-episode'}:{typ}"
            derived_seed = derive_seed(self.sampling_policy.config.seed, context)
            if sample:
                chosen = self.sampling_policy.sample([g[0] for g in group], [g[2] for g in group], k, self.sampling_policy.rng_for(context))
            else:
                chosen = SamplingPolicy(SamplingConfig(strategy="topk", seed=derived_seed)).sample([g[0] for g in group], [g[2] for g in group], k)
            selected.extend(chosen)
            sample_meta[typ] = self.sampling_policy.metadata(derived_seed=derived_seed, context=context)
        for entry in selected:
            entry.uses += 1
        self.stats["last_sampling"] = {"retrieval": sample_meta}
        return selected

    def update_after_episode(
        self,
        retrieved_ids: Sequence[str],
        new_ids: Sequence[str],
        score: float,
        success: Optional[bool] = None,
        task_id: Optional[str] = None,
        episode_id: Optional[Any] = None,
    ) -> None:
        score = clamp(score)
        baseline_context = (
            f"baseline_bootstrap:{task_id or 'unknown-task'}:"
            f"{episode_id or 'unknown-episode'}:{','.join(retrieved_ids)}:{len(new_ids)}"
        )
        baseline_seed = derive_seed(self.sampling_policy.config.seed, baseline_context)
        baseline_entries = self.sampling_policy.sample(
            list(self.entries.values()),
            [entry.weight for entry in self.entries.values()],
            min(8, len(self.entries)),
            self.sampling_policy.rng_for(baseline_context),
        )
        self.stats["last_sampling"] = {
            **dict(self.stats.get("last_sampling", {})),
            "baseline_bootstrap": self.sampling_policy.metadata(derived_seed=baseline_seed, context=baseline_context),
            "baseline_entry_ids": [entry.id for entry in baseline_entries],
        }
        prev_baseline = float(self.stats.get("score_ema", 0.0))
        immediate_ig = score - prev_baseline
        positive_delta = max(0.0, immediate_ig)

        for entry_id in new_ids:
            entry = self.entries.get(entry_id)
            if not entry:
                continue
            entry.ig_ema = self._ema(entry.ig_ema, immediate_ig)
            entry.score_ema = self._ema(entry.score_ema, score)
            if success:
                entry.wins += 1
            self._recompute_weight(entry)

        # Future IG: if a retrieved abstraction helped produce useful new abstractions
        # or a better-than-baseline trajectory, credit the parent chain up to two hops.
        frontier = list(dict.fromkeys(retrieved_ids))
        visited = set()
        for depth in range(2):
            next_frontier: List[str] = []
            discount = 1.0 if depth == 0 else 0.5
            for entry_id in frontier:
                if entry_id in visited:
                    continue
                visited.add(entry_id)
                entry = self.entries.get(entry_id)
                if not entry:
                    continue
                entry.future_ig_ema = self._ema(entry.future_ig_ema, positive_delta * discount)
                entry.score_ema = self._ema(entry.score_ema, score)
                if success:
                    entry.wins += 1
                self._recompute_weight(entry)
                next_frontier.extend(entry.parents)
            frontier = next_frontier

        n = int(self.stats.get("episodes", 0)) + 1
        self.stats["episodes"] = n
        self.stats["score_sum"] = float(self.stats.get("score_sum", 0.0)) + score
        self.stats["score_ema"] = self._ema(prev_baseline, score)
        self.stats["score_mean"] = self.stats["score_sum"] / n

    def _ema(self, old: float, new: float) -> float:
        return self.ema_decay * float(old) + (1.0 - self.ema_decay) * float(new)

    def _recompute_weight(self, entry: LibraryEntry) -> None:
        usage_bonus = min(0.5, 0.03 * entry.uses)
        win_bonus = min(0.5, 0.05 * entry.wins)
        value = 1.0 + self.alpha_ig * entry.ig_ema + self.beta_future_ig * entry.future_ig_ema
        entry.weight = max(0.05, value + usage_bonus + win_bonus)
        entry.updated_at = time.time()

    def format_for_prompt(self, entries: Sequence[LibraryEntry], max_chars: int = 5000) -> str:
        if not entries:
            return "No prior skills or insights are available yet."
        chunks: List[str] = []
        used = 0
        for i, e in enumerate(entries, start=1):
            text = f"[{i}] id={e.id} type={e.type} weight={e.weight:.2f}\nTitle: {e.title}\nContent: {e.content}\n"
            if used + len(text) > max_chars:
                break
            chunks.append(text)
            used += len(text)
        return "\n".join(chunks)
