from __future__ import annotations

from typing import Any, Dict

from evolib_agent_suite.llm.base import BaseLLM
from evolib_agent_suite.llm.providers import AzureOpenAILLM, HeuristicLLM, LiteLLMLLM, OpenAICompatibleLLM


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_llm(config: Dict[str, Any]) -> BaseLLM:
    provider = (config or {}).get("provider", "heuristic").lower()
    if provider == "heuristic":
        return HeuristicLLM()
    if provider in {"openai", "openai_compatible", "http"}:
        return OpenAICompatibleLLM(
            model=config.get("model"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            endpoint_url=config.get("endpoint_url") or config.get("url"),
            api_key_header=str(config.get("api_key_header", "authorization")),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            max_tokens=int(config.get("max_tokens", 512)),
            timeout=int(config.get("timeout", config.get("timeout_seconds", 60))),
            retries=int(config.get("retries", 3)),
            use_proxy=_as_bool(config.get("use_proxy", False)),
            proxy_url=config.get("proxy_url"),
            verify_ssl=_as_bool(config.get("verify_ssl", True)),
        )
    if provider in {"azure", "azure_openai"}:
        return AzureOpenAILLM(
            model=config.get("model"),
            api_key=config.get("api_key"),
            endpoint_url=config.get("endpoint_url") or config.get("url"),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            max_tokens=int(config.get("max_tokens", 512)),
            timeout=int(config.get("timeout", config.get("timeout_seconds", 120))),
            retries=int(config.get("retries", 3)),
            use_proxy=_as_bool(config.get("use_proxy", False)),
            proxy_url=config.get("proxy_url"),
            verify_ssl=_as_bool(config.get("verify_ssl", True)),
        )
    if provider == "litellm":
        return LiteLLMLLM(
            model=config["model"],
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 512)),
        )
    raise ValueError(f"Unknown llm provider: {provider}")
