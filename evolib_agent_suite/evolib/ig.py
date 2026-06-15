from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from evolib_agent_suite.utils import clamp


@dataclass
class IGConfig:
    baseline_strategy: str = "global_ema"
    ema_decay: float = 0.85
    window_size: int = 50
    min_samples: int = 1
    bootstrap_samples: int = 0
    per_domain_baseline: bool = False


class BaselineEstimator:
    """Computes immediate information gain baselines and persists state in library stats."""

    SUPPORTED_STRATEGIES = {
        "global_ema",
        "global_mean",
        "rolling_window",
        "domain_ema",
        "retrieval_ablation_proxy",
    }

    def __init__(self, stats: Dict[str, Any], config: Optional[IGConfig] = None) -> None:
        self.stats = stats
        self.config = config or IGConfig()
        if self.config.baseline_strategy not in self.SUPPORTED_STRATEGIES:
            supported = ", ".join(sorted(self.SUPPORTED_STRATEGIES))
            raise ValueError(f"Unsupported baseline_strategy={self.config.baseline_strategy!r}. Supported: {supported}")
        self._ensure_state()

    def compute_baseline(self, context: Optional[Dict[str, Any]] = None) -> float:
        context = context or {}
        strategy = self.config.baseline_strategy
        if strategy == "global_ema":
            return float(self.stats.get("score_ema", 0.0))
        if strategy == "global_mean":
            return float(self.stats.get("score_mean", 0.0))
        if strategy == "rolling_window":
            history = self._score_history()
            if not history:
                return float(self.stats.get("score_ema", 0.0))
            window = history[-max(1, int(self.config.window_size)) :]
            return sum(window) / len(window)
        if strategy == "domain_ema":
            domain = str(context.get("domain") or "generic")
            domain_stat = self.stats.get("domain_stats", {}).get(domain)
            if domain_stat and int(domain_stat.get("count", 0)) >= max(1, int(self.config.min_samples)):
                return float(domain_stat.get("score_ema", self.stats.get("score_ema", 0.0)))
            return float(self.stats.get("score_ema", 0.0))
        if strategy == "retrieval_ablation_proxy":
            proxy = self.stats.get("retrieval_ablation_proxy", {})
            if int(proxy.get("count", 0)) >= max(1, int(self.config.min_samples)):
                return float(proxy.get("score_mean", self.stats.get("score_ema", 0.0)))
            return float(self.stats.get("score_ema", 0.0))
        return float(self.stats.get("score_ema", 0.0))

    def compute_immediate_ig(self, score: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Union[float, str]]:
        score = clamp(score)
        baseline = self.compute_baseline(context)
        return {
            "baseline": baseline,
            "score": score,
            "immediate_ig": score - baseline,
            "baseline_strategy": self.config.baseline_strategy,
        }

    def update(self, score: float, context: Optional[Dict[str, Any]] = None) -> None:
        context = context or {}
        score = clamp(score)
        prev_ema = float(self.stats.get("score_ema", 0.0))
        n = int(self.stats.get("episodes", 0)) + 1
        score_sum = float(self.stats.get("score_sum", 0.0)) + score

        self.stats["episodes"] = n
        self.stats["score_sum"] = score_sum
        self.stats["score_ema"] = self._ema(prev_ema, score)
        self.stats["score_mean"] = score_sum / n
        self.stats["baseline_strategy"] = self.config.baseline_strategy

        history = self._score_history()
        history.append(score)
        max_history = max(1, int(self.config.window_size), int(self.config.bootstrap_samples))
        # Keep enough data for rolling windows without unbounded growth.
        self.stats["score_history"] = history[-max(max_history, 1000) :]

        domain = str(context.get("domain") or "generic")
        domain_stats = self.stats.setdefault("domain_stats", {})
        domain_stat = domain_stats.setdefault(domain, {"count": 0, "score_ema": 0.0, "score_sum": 0.0, "score_mean": 0.0})
        domain_count = int(domain_stat.get("count", 0)) + 1
        domain_sum = float(domain_stat.get("score_sum", 0.0)) + score
        domain_stat["count"] = domain_count
        domain_stat["score_sum"] = domain_sum
        domain_stat["score_ema"] = self._ema(float(domain_stat.get("score_ema", 0.0)), score)
        domain_stat["score_mean"] = domain_sum / domain_count

        retrieved_count = int(context.get("retrieved_count", len(context.get("retrieved_ids", []) or [])))
        if retrieved_count <= max(0, int(self.config.min_samples)):
            proxy = self.stats.setdefault("retrieval_ablation_proxy", {"count": 0, "score_sum": 0.0, "score_mean": 0.0})
            proxy_count = int(proxy.get("count", 0)) + 1
            proxy_sum = float(proxy.get("score_sum", 0.0)) + score
            proxy["count"] = proxy_count
            proxy["score_sum"] = proxy_sum
            proxy["score_mean"] = proxy_sum / proxy_count

    def _ensure_state(self) -> None:
        self.stats.setdefault("episodes", 0)
        self.stats.setdefault("score_ema", 0.0)
        self.stats.setdefault("score_sum", 0.0)
        self.stats.setdefault("score_mean", 0.0)
        self.stats.setdefault("score_history", [])
        self.stats.setdefault("domain_stats", {})
        self.stats["baseline_strategy"] = self.config.baseline_strategy
        self.stats["ig_config"] = self.config.__dict__.copy()

    def _score_history(self) -> List[float]:
        history = self.stats.setdefault("score_history", [])
        return [float(x) for x in history]

    def _ema(self, old: float, new: float) -> float:
        return self.config.ema_decay * float(old) + (1.0 - self.config.ema_decay) * float(new)
