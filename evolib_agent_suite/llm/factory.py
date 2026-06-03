from __future__ import annotations

from typing import Any, Dict

from evolib_agent_suite.llm.base import BaseLLM
from evolib_agent_suite.llm.providers import HeuristicLLM, LiteLLMLLM, OpenAICompatibleLLM


def build_llm(config: Dict[str, Any]) -> BaseLLM:
    provider = (config or {}).get("provider", "heuristic").lower()
    if provider == "heuristic":
        return HeuristicLLM()
    if provider in {"openai", "openai_compatible", "http"}:
        return OpenAICompatibleLLM(
            model=config.get("model"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 512)),
            timeout=int(config.get("timeout", 60)),
            retries=int(config.get("retries", 3)),
        )
    if provider == "litellm":
        return LiteLLMLLM(
            model=config["model"],
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 512)),
        )
    raise ValueError(f"Unknown llm provider: {provider}")
