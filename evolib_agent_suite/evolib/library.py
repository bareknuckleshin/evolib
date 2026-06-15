from __future__ import annotations

import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from evolib_agent_suite.evolib.consolidation import ConsolidationConfig, ConsolidationPolicy, LLMMerger
from evolib_agent_suite.utils import clamp, cosine, hashed_embedding, weighted_sample_without_replacement
from evolib_agent_suite.evolib.ig import BaselineEstimator, IGConfig


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


@dataclass
class RetrievalConfig:
    k_skills: int = 4
    k_insights: int = 4
    similarity_threshold: float = 0.05
    candidate_pool_multiplier: int = 4
    sampling_strategy: str = "weighted"
    temperature: float = 1.0
    epsilon: float = 0.1
    weight_alpha: float = 1.0
    similarity_alpha: float = 1.0


@dataclass
class RetrievedEntry:
    entry: LibraryEntry
    similarity: float
    retrieval_weight: float
    rank: int
    selected_by: str


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
        alpha_ig: float = 1.0,
        beta_future_ig: float = 0.7,
        ema_decay: float = 0.85,
        consolidation_config: Optional[ConsolidationConfig] = None,
        llm: Any = None,
        ig_config: Optional[Union[IGConfig, Dict[str, Any]]] = None,
    ) -> None:
        self.path = Path(path)
        self.similarity_merge_threshold = similarity_merge_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold
        self.alpha_ig = alpha_ig
        self.beta_future_ig = beta_future_ig
        self.ema_decay = ema_decay
        if isinstance(ig_config, IGConfig):
            self.ig_config = ig_config
        else:
            ig_data = dict(ig_config or {})
            ig_data.setdefault("ema_decay", ema_decay)
            self.ig_config = IGConfig(**ig_data)
        self.rng = random.Random(seed)
        if consolidation_config is None:
            consolidation_config = ConsolidationConfig(
                similarity_threshold=similarity_merge_threshold,
                ema_decay=ema_decay,
            )
        self.consolidation_config = consolidation_config
        merger = LLMMerger(llm) if llm is not None and consolidation_config.merge_strategy == "llm_merge" else None
        self.consolidation_policy = ConsolidationPolicy(consolidation_config, llm_merger=merger)
        self.entries: Dict[str, LibraryEntry] = {}
        self.stats: Dict[str, Any] = {"episodes": 0, "score_ema": 0.0, "score_sum": 0.0}
        self.load()
        self.baseline_estimator = BaselineEstimator(self.stats, self.ig_config)

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
        return self.consolidation_policy.find_target(list(self.entries.values()), candidate)

    def add_or_merge_many(
        self,
        candidates: Sequence[Dict[str, Any]],
        parents: Sequence[str],
        task_id: str,
        score: float,
        task_context: str = "",
    ) -> List[str]:
        new_or_updated: List[str] = []
        for item in candidates:
            candidate = self._make_entry(item, parents=parents, task_id=task_id, score=score)
            if not candidate.content:
                continue
            target = self.consolidation_policy.find_target(list(self.entries.values()), candidate)
            if target is not None:
                entry, sim = target
                self.consolidation_policy.merge(
                    entry,
                    candidate,
                    similarity=sim,
                    task_id=task_id,
                    score=score,
                    parents=parents,
                    task_context=task_context,
                )
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
        sampling_strategy: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        candidate_pool_multiplier: int = 4,
        temperature: float = 1.0,
        epsilon: float = 0.1,
        weight_alpha: float = 1.0,
        similarity_alpha: float = 1.0,
    ) -> List[LibraryEntry]:
        config = RetrievalConfig(
            k_skills=k_skills,
            k_insights=k_insights,
            similarity_threshold=self.retrieval_similarity_threshold
            if similarity_threshold is None
            else similarity_threshold,
            candidate_pool_multiplier=candidate_pool_multiplier,
            sampling_strategy=sampling_strategy or ("weighted" if sample else "topk"),
            temperature=temperature,
            epsilon=epsilon,
            weight_alpha=weight_alpha,
            similarity_alpha=similarity_alpha,
        )
        return [item.entry for item in self.retrieve_with_metadata(query, config=config)]

    def retrieve_with_metadata(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievedEntry]:
        config = config or RetrievalConfig(similarity_threshold=self.retrieval_similarity_threshold)
        if not self.entries:
            return []
        q_emb = hashed_embedding(query)
        scored: List[Tuple[LibraryEntry, float, float, float]] = []
        for entry in self.entries.values():
            sim = cosine(q_emb, entry.embedding)
            if sim >= config.similarity_threshold:
                retrieval_weight = max(1e-6, (0.2 + sim) * max(entry.weight, 1e-6))
                composite = self._retrieval_composite_score(entry, sim, retrieval_weight, config)
                scored.append((entry, sim, retrieval_weight, composite))
        if not scored:
            return []

        selected: List[RetrievedEntry] = []
        for typ, k in [("skill", config.k_skills), ("insight", config.k_insights)]:
            if k <= 0:
                continue
            group = [(e, sim, w, score) for (e, sim, w, score) in scored if e.type == typ]
            group.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
            group = group[: max(k * max(1, config.candidate_pool_multiplier), k)]
            chosen = self._select_retrieval_group(group, k, config)
            selected.extend(
                RetrievedEntry(entry=e, similarity=sim, retrieval_weight=w, rank=rank, selected_by=selected_by)
                for rank, (e, sim, w, _score, selected_by) in enumerate(chosen, start=1)
            )
        for item in selected:
            item.entry.uses += 1
        return selected

    def _retrieval_composite_score(
        self, entry: LibraryEntry, similarity: float, retrieval_weight: float, config: RetrievalConfig
    ) -> float:
        return (max(similarity, 1e-6) ** config.similarity_alpha) * (max(entry.weight, 1e-6) ** config.weight_alpha)

    def _select_retrieval_group(
        self,
        group: Sequence[Tuple[LibraryEntry, float, float, float]],
        k: int,
        config: RetrievalConfig,
    ) -> List[Tuple[LibraryEntry, float, float, float, str]]:
        if not group:
            return []
        strategy = config.sampling_strategy.strip().lower()
        if strategy == "topk":
            return [(*item, "topk") for item in group[:k]]
        if strategy == "weighted":
            sampled = weighted_sample_without_replacement(group, [g[2] for g in group], k, self.rng)
            return [(*item, "weighted") for item in sampled]
        if strategy == "softmax":
            temperature = max(config.temperature, 1e-6)
            max_score = max(g[3] for g in group)
            weights = [math.exp((g[3] - max_score) / temperature) for g in group]
            sampled = weighted_sample_without_replacement(group, weights, k, self.rng)
            return [(*item, "softmax") for item in sampled]
        if strategy == "epsilon_greedy":
            out: List[Tuple[LibraryEntry, float, float, float, str]] = []
            remaining = list(group)
            while remaining and len(out) < k:
                if self.rng.random() < config.epsilon:
                    idx = self.rng.randrange(len(remaining))
                    item = remaining.pop(idx)
                    out.append((*item, "epsilon_explore"))
                else:
                    item = remaining.pop(0)
                    out.append((*item, "epsilon_topk"))
            return out
        raise ValueError(f"Unsupported sampling_strategy: {config.sampling_strategy}")

    def update_after_episode(
        self,
        retrieved_ids: Sequence[str],
        new_ids: Sequence[str],
        score: float,
        success: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        context.setdefault("retrieved_ids", list(retrieved_ids))
        context.setdefault("retrieved_count", len(retrieved_ids))
        ig_info = self.baseline_estimator.compute_immediate_ig(score, context)
        score = float(ig_info["score"])
        immediate_ig = float(ig_info["immediate_ig"])
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

        self.baseline_estimator.update(score, context)
        return ig_info

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
