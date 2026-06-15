from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from evolib_agent_suite.utils import weighted_sample_without_replacement


SUPPORTED_STRATEGIES = {"topk", "uniform", "weighted", "softmax", "top_p", "epsilon_greedy"}


@dataclass
class SamplingConfig:
    strategy: str = "weighted"
    temperature: float = 1.0
    top_p: float = 0.9
    epsilon: float = 0.1
    seed: int = 0
    without_replacement: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], *, default_strategy: str = "weighted", default_seed: int = 0) -> "SamplingConfig":
        data = dict(data or {})
        data.setdefault("strategy", default_strategy)
        data.setdefault("seed", default_seed)
        return cls(
            strategy=str(data.get("strategy", default_strategy)),
            temperature=float(data.get("temperature", 1.0)),
            top_p=float(data.get("top_p", 0.9)),
            epsilon=float(data.get("epsilon", 0.1)),
            seed=int(data.get("seed", default_seed)),
            without_replacement=bool(data.get("without_replacement", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join([str(base_seed), *[str(p) for p in parts if p is not None]])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


class SamplingPolicy:
    def __init__(self, config: Optional[SamplingConfig] = None) -> None:
        self.config = config or SamplingConfig()
        if self.config.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported sampling strategy: {self.config.strategy}")

    def rng_for(self, *parts: Any) -> random.Random:
        return random.Random(derive_seed(self.config.seed, *parts))

    def metadata(self, *, derived_seed: Optional[int] = None, context: Optional[str] = None) -> Dict[str, Any]:
        data = self.config.to_dict()
        if derived_seed is not None:
            data["derived_seed"] = derived_seed
        if context is not None:
            data["context"] = context
        return data

    def sample(
        self,
        items: Sequence[Any],
        scores: Optional[Sequence[float]],
        k: int,
        rng: Optional[random.Random] = None,
    ) -> List[Any]:
        if k <= 0 or not items:
            return []
        rng = rng or random.Random(self.config.seed)
        paired = list(zip(items, self._scores(items, scores)))
        k = min(k, len(paired)) if self.config.without_replacement else k
        strategy = self.config.strategy
        if strategy == "topk":
            return [item for item, _ in sorted(paired, key=lambda x: x[1], reverse=True)[:k]]
        if strategy == "uniform":
            return self._uniform([item for item, _ in paired], k, rng)
        if strategy == "epsilon_greedy" and rng.random() < self._clamp(self.config.epsilon):
            return self._uniform([item for item, _ in paired], k, rng)
        if strategy == "top_p":
            paired = self._top_p_paired(paired)
            k = min(k, len(paired)) if self.config.without_replacement else k
        weights = self._weights_for_strategy(paired, strategy)
        if self.config.without_replacement:
            return weighted_sample_without_replacement([item for item, _ in paired], weights, k, rng)
        return [self._weighted_choice([item for item, _ in paired], weights, rng) for _ in range(k)]

    def _weights_for_strategy(self, paired: Sequence[tuple[Any, float]], strategy: str) -> List[float]:
        scores = [score for _, score in paired]
        if strategy == "softmax":
            return self._softmax(scores)
        # weighted, epsilon_greedy exploitation, and top_p's filtered subset use
        # non-negative scores directly.
        return [max(score, 1e-8) for score in scores]

    def _top_p_paired(self, paired: Sequence[tuple[Any, float]]) -> List[tuple[Any, float]]:
        ranked = sorted(paired, key=lambda x: x[1], reverse=True)
        probs = self._softmax([score for _, score in ranked])
        selected: List[tuple[Any, float]] = []
        total = 0.0
        for pair, prob in zip(ranked, probs):
            selected.append(pair)
            total += prob
            if total >= self._clamp(self.config.top_p):
                break
        return selected

    def _softmax(self, scores: Sequence[float]) -> List[float]:
        temp = max(float(self.config.temperature), 1e-8)
        if not scores:
            return []
        scaled = [float(s) / temp for s in scores]
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def _uniform(self, items: Sequence[Any], k: int, rng: random.Random) -> List[Any]:
        pool = list(items)
        if self.config.without_replacement:
            rng.shuffle(pool)
            return pool[: min(k, len(pool))]
        return [pool[rng.randrange(len(pool))] for _ in range(k)] if pool else []

    @staticmethod
    def _scores(items: Sequence[Any], scores: Optional[Sequence[float]]) -> List[float]:
        if scores is None:
            return [1.0 for _ in items]
        out = [float(s) for s in scores]
        if len(out) != len(items):
            raise ValueError("items and scores must have the same length")
        return out

    @staticmethod
    def _weighted_choice(items: Sequence[Any], weights: Sequence[float], rng: random.Random) -> Any:
        total = sum(max(float(w), 0.0) for w in weights)
        if total <= 0:
            return items[rng.randrange(len(items))]
        r = rng.random() * total
        upto = 0.0
        for item, weight in zip(items, weights):
            upto += max(float(weight), 0.0)
            if upto >= r:
                return item
        return items[-1]

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))
