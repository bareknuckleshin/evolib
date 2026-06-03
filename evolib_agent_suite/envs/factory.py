from __future__ import annotations

from typing import Any, Dict

from evolib_agent_suite.envs.base import EnvironmentAdapter
from evolib_agent_suite.envs.mock import MockHouseholdAdapter


def build_env(config: Dict[str, Any]) -> EnvironmentAdapter:
    backend = (config or {}).get("backend", "mock").lower()
    kwargs = dict(config.get("kwargs", {}))
    if backend == "mock":
        return MockHouseholdAdapter(**kwargs)
    if backend in {"webshop", "original_webshop", "webshop_original"}:
        from evolib_agent_suite.envs.original_webshop import OriginalWebShopAdapter

        return OriginalWebShopAdapter(**kwargs)
    if backend in {"alfworld", "original_alfworld", "alfworld_original"}:
        from evolib_agent_suite.envs.original_alfworld import OriginalALFWorldAdapter

        return OriginalALFWorldAdapter(**kwargs)
    raise ValueError(f"Unknown environment backend: {backend}")
