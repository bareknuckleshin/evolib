from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from evolib_agent_suite.utils import weighted_sample_without_replacement


SUPPORTED_SAMPLING_STRATEGIES = {"topk", "uniform", "weighted", "softmax", "top_p", "epsilon_greedy"}


@dataclass
class SamplingConfig:
    """Shared sampling configuration for retrieval, composition, and baselines."""

    strategy: str = "weighted"
    temperature: float = 1.0
    top_p: float = 0.9
    epsilon: float = 0.1
    seed: int = 0
    without_replacement: bool = True


@dataclass
class SamplingTrace:
    """Reproducibility metadata for a sampling decision."""

    strategy: str
    base_seed: int
    derived_seed: int
    context_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 32-bit seed from a config seed and episode/task identifiers."""

    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts if part is not None)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def rng_for_context(config: SamplingConfig, *parts: Any) -> tuple[random.Random, SamplingTrace]:
    context_id = ":".join(str(part) for part in parts if part is not None)
    seed = derive_seed(config.seed, context_id)
    return random.Random(seed), SamplingTrace(
        strategy=(config.strategy or "weighted").strip().lower(),
        base_seed=int(config.seed),
        derived_seed=seed,
        context_id=context_id,
    )


class SamplingPolicy:
    """Shared item sampler used across EvoLib selection surfaces."""

    def __init__(self, config: Optional[SamplingConfig] = None) -> None:
        self.config = config or SamplingConfig()
        strategy = (self.config.strategy or "weighted").strip().lower()
        if strategy not in SUPPORTED_SAMPLING_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_SAMPLING_STRATEGIES))
            raise ValueError(f"Unsupported sampling strategy: {self.config.strategy!r}. Supported: {supported}")

    def sample(
        self,
        items: Sequence[Any],
        scores: Optional[Sequence[float]],
        k: int,
        rng: Optional[random.Random] = None,
    ) -> List[Any]:
        pool = list(items)
        if not pool or k <= 0:
            return []
        k = min(int(k), len(pool)) if self.config.without_replacement else int(k)
        rng = rng or random.Random(self.config.seed)
        strategy = (self.config.strategy or "weighted").strip().lower()
        weights = self._weights(pool, scores)

        if strategy == "topk":
            return pool[:k]
        if strategy == "uniform":
            return self._uniform(pool, k, rng)
        if strategy == "weighted":
            return self._weighted(pool, weights, k, rng)
        if strategy == "softmax":
            return self._weighted(pool, self._softmax(weights), k, rng)
        if strategy == "top_p":
            ranked = sorted(zip(pool, weights), key=lambda x: x[1], reverse=True)
            total = sum(max(w, 0.0) for _, w in ranked) or float(len(ranked))
            cumulative = 0.0
            nucleus: List[tuple[Any, float]] = []
            threshold = min(max(float(self.config.top_p), 0.0), 1.0)
            for item, weight in ranked:
                nucleus.append((item, weight))
                cumulative += max(weight, 0.0) / total
                if cumulative >= threshold:
                    break
            return self._weighted([i for i, _ in nucleus], [w for _, w in nucleus], k, rng)
        if strategy == "epsilon_greedy":
            return self._epsilon_greedy(pool, weights, k, rng)
        raise ValueError(f"Unsupported sampling strategy: {self.config.strategy}")

    def _weights(self, items: Sequence[Any], scores: Optional[Sequence[float]]) -> List[float]:
        if scores is None:
            return [1.0] * len(items)
        out = [float(score) for score in scores]
        if len(out) != len(items):
            raise ValueError("items and scores must have the same length")
        return out

    def _uniform(self, items: Sequence[Any], k: int, rng: random.Random) -> List[Any]:
        if self.config.without_replacement:
            pool = list(items)
            rng.shuffle(pool)
            return pool[: min(k, len(pool))]
        return [items[rng.randrange(len(items))] for _ in range(k)]

    def _weighted(self, items: Sequence[Any], weights: Sequence[float], k: int, rng: random.Random) -> List[Any]:
        safe_weights = [max(float(weight), 1e-8) for weight in weights]
        if self.config.without_replacement:
            return weighted_sample_without_replacement(items, safe_weights, k, rng)
        return rng.choices(list(items), weights=safe_weights, k=k)

    def _softmax(self, scores: Sequence[float]) -> List[float]:
        temperature = max(float(self.config.temperature), 1e-6)
        max_score = max(scores) if scores else 0.0
        return [math.exp((float(score) - max_score) / temperature) for score in scores]

    def _epsilon_greedy(self, items: Sequence[Any], scores: Sequence[float], k: int, rng: random.Random) -> List[Any]:
        ranked = [item for item, _score in sorted(zip(items, scores), key=lambda x: x[1], reverse=True)]
        out: List[Any] = []
        while ranked and len(out) < k:
            if rng.random() < max(0.0, min(1.0, float(self.config.epsilon))):
                idx = rng.randrange(len(ranked))
            else:
                idx = 0
            out.append(ranked.pop(idx) if self.config.without_replacement else ranked[idx])
        return out
