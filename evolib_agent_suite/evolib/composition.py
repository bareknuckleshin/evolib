from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

from evolib_agent_suite.evolib.library import LibraryEntry
from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, rng_for_context


@dataclass
class CompositionConfig:
    """Configuration for composing retrieved EvoLib entries into prompt candidates."""

    strategy: str = "all_context"
    max_candidates: int = 8
    max_skills_per_candidate: int = 4
    max_insights_per_candidate: int = 4
    include_singletons: bool = True
    include_mixed: bool = True
    score_policy: str = "sum_weight"
    sampling_strategy: str = "weighted"
    temperature: float = 1.0
    top_p: float = 0.9
    epsilon: float = 0.1
    seed: int = 0
    without_replacement: bool = True


@dataclass
class CandidateSolution:
    """A composed set of EvoLib entries that can be supplied to the agent prompt."""

    id: str
    entries: List[LibraryEntry]
    composition_type: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def entry_ids(self) -> List[str]:
        return [entry.id for entry in self.entries]


def compose_candidates(
    entries: Sequence[LibraryEntry],
    config: Optional[CompositionConfig] = None,
    rng: Optional[random.Random] = None,
) -> List[CandidateSolution]:
    """Compose retrieved entries into candidate solutions.

    The first returned candidate is the selected/default candidate used by the
    current agent. Additional candidates are logged for experimentation.
    """

    cfg = config or CompositionConfig()
    rng = rng or random.Random(0)
    entries = list(entries)
    skills = [entry for entry in entries if entry.type == "skill"]
    insights = [entry for entry in entries if entry.type == "insight"]
    strategy = (cfg.strategy or "all_context").strip().lower()

    if strategy == "all_context":
        return [_candidate("all_context", entries, cfg, index=0, metadata={"strategy": strategy})]
    if strategy == "singletons":
        candidates = _singletons(entries, cfg)
    elif strategy == "pairwise":
        candidates = _pairwise(skills, insights, cfg)
    elif strategy == "mixed_bundle":
        candidates = [_mixed_bundle(skills, insights, cfg)]
    elif strategy == "weighted_sampled_bundle":
        candidates = [_weighted_sampled_bundle(skills, insights, cfg, rng)]
    else:
        raise ValueError(f"Unsupported composition strategy: {cfg.strategy}")

    candidates = [candidate for candidate in candidates if candidate.entries]
    candidates.sort(key=lambda c: (c.score, len(c.entries)), reverse=True)
    return candidates[: max(1, cfg.max_candidates)] or [_candidate(strategy, entries, cfg, index=0)]


def select_candidate(
    entries: Sequence[LibraryEntry],
    config: Optional[CompositionConfig] = None,
    rng: Optional[random.Random] = None,
) -> CandidateSolution:
    return compose_candidates(entries, config=config, rng=rng)[0]


def _singletons(entries: Sequence[LibraryEntry], cfg: CompositionConfig) -> List[CandidateSolution]:
    if not cfg.include_singletons:
        return []
    return [_candidate("singleton", [entry], cfg, index=i) for i, entry in enumerate(entries)]


def _pairwise(skills: Sequence[LibraryEntry], insights: Sequence[LibraryEntry], cfg: CompositionConfig) -> List[CandidateSolution]:
    candidates: List[CandidateSolution] = []
    for i, pair in enumerate(combinations(skills, 2)):
        candidates.append(_candidate("skill_skill_pair", list(pair), cfg, index=i))
    offset = len(candidates)
    if cfg.include_mixed:
        for i, skill in enumerate(skills):
            for insight in insights:
                candidates.append(_candidate("skill_insight_pair", [skill, insight], cfg, index=offset + i))
    return candidates


def _mixed_bundle(skills: Sequence[LibraryEntry], insights: Sequence[LibraryEntry], cfg: CompositionConfig) -> CandidateSolution:
    bundle = list(skills[: max(0, cfg.max_skills_per_candidate)])
    if cfg.include_mixed:
        bundle.extend(insights[: max(0, cfg.max_insights_per_candidate)])
    return _candidate("mixed_bundle", bundle, cfg, index=0)


def _weighted_sampled_bundle(
    skills: Sequence[LibraryEntry],
    insights: Sequence[LibraryEntry],
    cfg: CompositionConfig,
    rng: random.Random,
) -> CandidateSolution:
    sampling_config = SamplingConfig(
        strategy=cfg.sampling_strategy,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        epsilon=cfg.epsilon,
        seed=cfg.seed,
        without_replacement=cfg.without_replacement,
    )
    policy = SamplingPolicy(sampling_config)
    skill_rng, skill_trace = rng_for_context(sampling_config, "composition", "skills")
    sampled_skills = policy.sample(
        list(skills), [max(entry.weight, 1e-6) for entry in skills], cfg.max_skills_per_candidate, skill_rng or rng
    )
    sampled_insights: List[LibraryEntry] = []
    insight_trace = None
    if cfg.include_mixed:
        insight_rng, insight_trace = rng_for_context(sampling_config, "composition", "insights")
        sampled_insights = policy.sample(
            list(insights), [max(entry.weight, 1e-6) for entry in insights], cfg.max_insights_per_candidate, insight_rng or rng
        )
    metadata = {"sampling": {"skills": skill_trace.to_dict(), "insights": insight_trace.to_dict() if insight_trace else None}}
    return _candidate("weighted_sampled_bundle", sampled_skills + sampled_insights, cfg, index=0, metadata=metadata)


def _candidate(
    composition_type: str,
    entries: Sequence[LibraryEntry],
    cfg: CompositionConfig,
    index: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> CandidateSolution:
    unique_entries = list({entry.id: entry for entry in entries}.values())
    score = _score(unique_entries, cfg.score_policy)
    entry_sig = "-".join(entry.id for entry in unique_entries) or "empty"
    return CandidateSolution(
        id=f"{composition_type}:{index}:{entry_sig}",
        entries=unique_entries,
        composition_type=composition_type,
        score=score,
        metadata=metadata or {},
    )


def _score(entries: Sequence[LibraryEntry], policy: str) -> float:
    if not entries:
        return 0.0
    weights = [float(entry.weight) for entry in entries]
    policy = (policy or "sum_weight").lower()
    if policy == "mean_weight":
        return sum(weights) / len(weights)
    if policy == "max_weight":
        return max(weights)
    return sum(weights)
