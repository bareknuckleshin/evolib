from __future__ import annotations

import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from evolib_agent_suite.evolib.consolidation import ConsolidationConfig, ConsolidationPolicy, LLMMerger
from evolib_agent_suite.utils import clamp, cosine, hashed_embedding
from evolib_agent_suite.evolib.ig import BaselineEstimator, IGConfig
from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, SamplingTrace, rng_for_context
from evolib_agent_suite.evolib.storage import JsonLibraryStorage, LibraryStorage, SQLiteLibraryStorage


CURRENT_SCHEMA_VERSION = 2


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
class LineageEdge:
    parent_id: str
    child_id: str
    task_id: str
    created_at: float = field(default_factory=time.time)
    edge_type: str = "create"
    source_score: float = 0.0
    credit: float = 0.0
    depth: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageEdge":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})


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
    top_p: float = 0.9
    seed: int = 0
    without_replacement: bool = True
    context_id: str = ""


@dataclass
class RetrievedEntry:
    entry: LibraryEntry
    similarity: float
    retrieval_weight: float
    rank: int
    selected_by: str
    sampling_seed: Optional[int] = None
    sampling_base_seed: Optional[int] = None
    sampling_context_id: str = ""


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
        storage: Optional[LibraryStorage] = None,
        storage_backend: str = "json",
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.path = Path(path)
        self.embedding_fn = embedding_fn or hashed_embedding
        self.storage = storage or self._build_storage(self.path, storage_backend)
        self.schema_version = CURRENT_SCHEMA_VERSION
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
        self.seed = int(seed)
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
        self.lineage_edges: List[LineageEdge] = []
        self.fig_events: List[Dict[str, Any]] = []
        self.merge_events: List[Dict[str, Any]] = []
        self.retrieval_events: List[Dict[str, Any]] = []
        self.policy_snapshots: List[Dict[str, Any]] = []
        self.last_fig_credit_events: List[Dict[str, Any]] = []
        self.last_retrieval_event: Dict[str, Any] = {}
        self.last_consolidation_decisions: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {"episodes": 0, "score_ema": 0.0, "score_sum": 0.0}
        self.load()
        self.baseline_estimator = BaselineEstimator(self.stats, self.ig_config)

    def _build_storage(self, path: Path, backend: str) -> LibraryStorage:
        backend = (backend or "json").strip().lower()
        if backend == "json":
            return JsonLibraryStorage(path)
        if backend == "sqlite":
            return SQLiteLibraryStorage(path)
        raise ValueError(f"Unsupported library storage backend: {backend}")

    def load(self) -> None:
        data = self.storage.load()
        if not data:
            return
        loaded_schema_version = int(data.get("schema_version", 1))
        self.stats = dict(data.get("stats", self.stats))
        self.schema_version = max(CURRENT_SCHEMA_VERSION, loaded_schema_version)
        self.entries = {e["id"]: LibraryEntry.from_dict(e) for e in data.get("entries", [])}
        self.lineage_edges = [LineageEdge.from_dict(e) for e in data.get("lineage_edges", [])]
        self.fig_events = list(data.get("ig_events", data.get("fig_events", [])))
        self.merge_events = list(data.get("merge_events", []))
        self.retrieval_events = list(data.get("retrieval_events", []))
        self.policy_snapshots = list(data.get("policy_snapshots", []))
        self.last_fig_credit_events = []
        self.last_retrieval_event = {}
        self.last_consolidation_decisions = []

    def save(self) -> None:
        payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "stats": self.stats,
            "entries": [entry.to_dict() for entry in self.entries.values()],
            "lineage_edges": [edge.to_dict() for edge in self.lineage_edges],
            "merge_events": self.merge_events,
            "ig_events": self.fig_events,
            "fig_events": self.fig_events,
            "retrieval_events": self.retrieval_events,
            "policy_snapshots": self.policy_snapshots,
        }
        self.storage.save(payload)

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
            embedding=self._embed(text),
            parents=list(dict.fromkeys(parents)),
            score_ema=score,
            source_task_ids=[task_id],
        )

    def _find_merge_target(self, candidate: LibraryEntry) -> Optional[Tuple[LibraryEntry, float]]:
        return self.consolidation_policy.find_target(list(self.entries.values()), candidate)

    def _embed(self, text: str) -> List[float]:
        return list(self.embedding_fn(text))

    def _record_lineage_edges(
        self,
        parent_ids: Sequence[str],
        child_id: str,
        task_id: str,
        score: float,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        for parent_id in dict.fromkeys(parent_ids):
            if not parent_id or parent_id == child_id:
                continue
            self.lineage_edges.append(
                LineageEdge(
                    parent_id=parent_id,
                    child_id=child_id,
                    task_id=task_id,
                    edge_type=edge_type,
                    source_score=score,
                    depth=1,
                    metadata=dict(metadata or {}),
                )
            )

    def add_or_merge_many(
        self,
        candidates: Sequence[Dict[str, Any]],
        parents: Sequence[str],
        task_id: str,
        score: float,
        task_context: str = "",
    ) -> List[str]:
        new_or_updated: List[str] = []
        self.last_consolidation_decisions = []
        for item in candidates:
            candidate = self._make_entry(item, parents=parents, task_id=task_id, score=score)
            if not candidate.content:
                continue
            target = self.consolidation_policy.find_target(list(self.entries.values()), candidate)
            if target is not None:
                entry, sim = target
                entry.updated_at = time.time()
                entry.source_task_ids = list(dict.fromkeys(entry.source_task_ids + [task_id]))
                entry.parents = list(dict.fromkeys(entry.parents + list(parents)))
                for parent_id in parents:
                    parent = self.entries.get(parent_id)
                    if parent and entry.id not in parent.children:
                        parent.children.append(entry.id)
                entry.tags = list(dict.fromkeys(entry.tags + candidate.tags))
                # Keep the more general/longer wording, but avoid unbounded growth.
                if len(candidate.content) > len(entry.content) and len(candidate.content) < 1200:
                    entry.content = candidate.content
                    entry.title = candidate.title or entry.title
                    entry.embedding = self._embed(entry.text)
                entry.score_ema = self._ema(entry.score_ema, score)
                entry.metadata["last_merge_similarity"] = sim
                self._record_lineage_edges(
                    parents,
                    entry.id,
                    task_id,
                    score,
                    edge_type="merge",
                    metadata={"merge_similarity": sim, "candidate_id": candidate.id},
                )
                self.consolidation_policy.merge(
                    entry,
                    candidate,
                    similarity=sim,
                    task_id=task_id,
                    score=score,
                    parents=parents,
                    task_context=task_context,
                )
                entry.embedding = self._embed(entry.text)
                decision = {
                    "action": "merge",
                    "task_id": task_id,
                    "candidate_id": candidate.id,
                    "target_id": entry.id,
                    "similarity": sim,
                    "score": score,
                    "strategy": self.consolidation_config.merge_strategy,
                    "created_at": time.time(),
                }
                self.last_consolidation_decisions.append(decision)
                self.merge_events.append(decision)
                new_or_updated.append(entry.id)
            else:
                self.entries[candidate.id] = candidate
                for parent_id in candidate.parents:
                    parent = self.entries.get(parent_id)
                    if parent and candidate.id not in parent.children:
                        parent.children.append(candidate.id)
                self._record_lineage_edges(
                    candidate.parents,
                    candidate.id,
                    task_id,
                    score,
                    edge_type="create",
                )
                self.last_consolidation_decisions.append({
                    "action": "create",
                    "task_id": task_id,
                    "candidate_id": candidate.id,
                    "target_id": candidate.id,
                    "score": score,
                    "created_at": time.time(),
                })
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
            seed=self.seed,
        )
        return [item.entry for item in self.retrieve_with_metadata(query, config=config)]

    def retrieve_with_metadata(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievedEntry]:
        config = config or RetrievalConfig(similarity_threshold=self.retrieval_similarity_threshold)
        if not self.entries:
            self.last_retrieval_event = {"candidate_count": 0, "selected_entry_ids": [], "config": self._config_dict(config)}
            return []
        q_emb = self._embed(query)
        scored: List[Tuple[LibraryEntry, float, float, float]] = []
        for entry in self.entries.values():
            sim = cosine(q_emb, entry.embedding)
            if sim >= config.similarity_threshold:
                retrieval_weight = max(1e-6, (0.2 + sim) * max(entry.weight, 1e-6))
                composite = self._retrieval_composite_score(entry, sim, retrieval_weight, config)
                scored.append((entry, sim, retrieval_weight, composite))
        if not scored:
            self.last_retrieval_event = {"candidate_count": 0, "selected_entry_ids": [], "config": self._config_dict(config)}
            return []

        candidate_count = len(scored)
        selected: List[RetrievedEntry] = []
        for typ, k in [("skill", config.k_skills), ("insight", config.k_insights)]:
            if k <= 0:
                continue
            group = [(e, sim, w, score) for (e, sim, w, score) in scored if e.type == typ]
            group.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
            group = group[: max(k * max(1, config.candidate_pool_multiplier), k)]
            chosen = self._select_retrieval_group(group, k, config)
            selected.extend(
                RetrievedEntry(
                    entry=e,
                    similarity=sim,
                    retrieval_weight=w,
                    rank=rank,
                    selected_by=selected_by,
                    sampling_seed=trace.derived_seed,
                    sampling_base_seed=trace.base_seed,
                    sampling_context_id=trace.context_id,
                )
                for rank, (e, sim, w, _score, selected_by, trace) in enumerate(chosen, start=1)
            )
        for item in selected:
            item.entry.uses += 1
        self.last_retrieval_event = {
            "candidate_count": candidate_count,
            "selected_entry_ids": [item.entry.id for item in selected],
            "config": self._config_dict(config),
            "created_at": time.time(),
        }
        self.retrieval_events.append(self.last_retrieval_event)
        return selected

    def _config_dict(self, config: Any) -> Dict[str, Any]:
        return asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config or {})

    def _retrieval_composite_score(
        self, entry: LibraryEntry, similarity: float, retrieval_weight: float, config: RetrievalConfig
    ) -> float:
        return (max(similarity, 1e-6) ** config.similarity_alpha) * (max(entry.weight, 1e-6) ** config.weight_alpha)

    def _select_retrieval_group(
        self,
        group: Sequence[Tuple[LibraryEntry, float, float, float]],
        k: int,
        config: RetrievalConfig,
    ) -> List[Tuple[LibraryEntry, float, float, float, str, SamplingTrace]]:
        if not group:
            return []
        sampling_config = SamplingConfig(
            strategy=config.sampling_strategy,
            temperature=config.temperature,
            top_p=config.top_p,
            epsilon=config.epsilon,
            seed=config.seed or self.seed,
            without_replacement=config.without_replacement,
        )
        group_type = group[0][0].type if group else "entry"
        rng, trace = rng_for_context(sampling_config, config.context_id, group_type, k)
        scores = [g[3] for g in group]
        if sampling_config.strategy == "weighted":
            scores = [g[2] for g in group]
        sampled = SamplingPolicy(sampling_config).sample(group, scores, k, rng)
        return [(*item, sampling_config.strategy, trace) for item in sampled]

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
        prev_baseline = float(ig_info["baseline"])
        score = float(ig_info["score"])
        immediate_ig = float(ig_info["immediate_ig"])
        positive_delta = max(0.0, immediate_ig)
        episode_fig_events: List[Dict[str, Any]] = []

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
                credit = positive_delta * discount
                entry.future_ig_ema = self._ema(entry.future_ig_ema, credit)
                entry.score_ema = self._ema(entry.score_ema, score)
                if success:
                    entry.wins += 1
                self._recompute_weight(entry)
                event = {
                    "entry_id": entry_id,
                    "source_entry_ids": list(dict.fromkeys(new_ids)),
                    "score": score,
                    "baseline": prev_baseline,
                    "immediate_ig": immediate_ig,
                    "credit": credit,
                    "depth": depth + 1,
                    "success": success,
                    "created_at": time.time(),
                }
                episode_fig_events.append(event)
                next_frontier.extend(entry.parents)
            frontier = next_frontier

        self.last_fig_credit_events = episode_fig_events
        self.fig_events.extend(episode_fig_events)

        n = int(self.stats.get("episodes", 0)) + 1
        self.stats["episodes"] = n
        self.stats["score_sum"] = float(self.stats.get("score_sum", 0.0)) + score
        self.stats["score_ema"] = self._ema(prev_baseline, score)
        self.stats["score_mean"] = self.stats["score_sum"] / n
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

    def _walk_lineage(self, entry_id: str, max_depth: int, reverse: bool = False) -> List[Dict[str, Any]]:
        if max_depth < 1:
            return []
        results: List[Dict[str, Any]] = []
        frontier: List[Tuple[str, int]] = [(entry_id, 0)]
        visited = {entry_id}
        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for edge in self.lineage_edges:
                source_id = edge.child_id if reverse else edge.parent_id
                target_id = edge.parent_id if reverse else edge.child_id
                if source_id != current_id or target_id in visited:
                    continue
                visited.add(target_id)
                next_depth = depth + 1
                results.append(
                    {
                        "entry_id": target_id,
                        "depth": next_depth,
                        "edge": edge.to_dict(),
                    }
                )
                frontier.append((target_id, next_depth))
        return results

    def get_ancestors(self, entry_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        return self._walk_lineage(entry_id, max_depth=max_depth, reverse=True)

    def get_descendants(self, entry_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        return self._walk_lineage(entry_id, max_depth=max_depth, reverse=False)

    def summarize_lineage(self, entry_id: str) -> Dict[str, Any]:
        related_edges = [
            edge for edge in self.lineage_edges if edge.parent_id == entry_id or edge.child_id == entry_id
        ]
        credit_events = [event for event in self.fig_events if event.get("entry_id") == entry_id]
        by_type: Dict[str, int] = {}
        for edge in related_edges:
            by_type[edge.edge_type] = by_type.get(edge.edge_type, 0) + 1
        return {
            "entry_id": entry_id,
            "exists": entry_id in self.entries,
            "ancestor_count": len(self.get_ancestors(entry_id, max_depth=10)),
            "descendant_count": len(self.get_descendants(entry_id, max_depth=10)),
            "direct_parent_count": sum(1 for edge in self.lineage_edges if edge.child_id == entry_id),
            "direct_child_count": sum(1 for edge in self.lineage_edges if edge.parent_id == entry_id),
            "edge_count": len(related_edges),
            "edge_types": by_type,
            "fig_credit_event_count": len(credit_events),
            "fig_credit_total": sum(float(event.get("credit", 0.0)) for event in credit_events),
        }

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
