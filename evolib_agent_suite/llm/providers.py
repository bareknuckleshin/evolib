from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from evolib_agent_suite.llm.base import BaseLLM


class HeuristicLLM(BaseLLM):
    """Tiny offline model for smoke tests.

    It is not meant to solve WebShop/ALFWorld. It only lets you verify that the
    EvoLib loop, library persistence, and CLI are wired correctly.
    """

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        lower = user_prompt.lower()
        if "return json" in lower and "score" in lower:
            success_like = any(x in lower for x in ["success: true", "reward: 1", "done: true"])
            return json.dumps({"score": 1.0 if success_like else 0.35, "progress": 1.0 if success_like else 0.35, "notes": "heuristic score"})
        if "extract" in lower and "abstractions" in lower:
            return json.dumps(
                [
                    {
                        "type": "skill",
                        "title": "Inspect available actions before acting",
                        "content": "At each step, read the latest observation and available actions, then choose the action that directly advances the goal.",
                        "tags": ["generic", "planning"],
                    },
                    {
                        "type": "insight",
                        "title": "Avoid unsupported commands",
                        "content": "Prefer exact commands shown by the environment instead of inventing new action strings.",
                        "tags": ["grounding"],
                    },
                ]
            )
        # Mock env policy.
        action_lines = re.findall(r"-\s*([^\n]+)", user_prompt)
        if "put apple in basket" in lower:
            if "picked up apple" in lower and "put apple in basket" in lower:
                return "Thought: Finish the task.\nAction: put apple in basket"
            if "apple" in lower and "not holding" in lower:
                return "Thought: The apple is visible; pick it up.\nAction: pick apple"
            return "Thought: Look for the apple.\nAction: look"
        if action_lines:
            return f"Thought: Choose a valid listed action.\nAction: {action_lines[0].strip()}"
        return "Thought: I need to inspect the environment.\nAction: look"


class OpenAICompatibleLLM(BaseLLM):
    """Minimal Chat-Completions-compatible HTTP client.

    Configure with env vars or constructor arguments:
    - api_key: defaults to OPENAI_API_KEY
    - base_url: defaults to https://api.openai.com/v1
    - model: defaults to LLM_MODEL or gpt-4o-mini
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        if not self.api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass api_key.")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last_error}")


class LiteLLMLLM(BaseLLM):
    """Optional provider using litellm if it is installed."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512, **kwargs: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("Install litellm to use provider=litellm") from exc
        resp = litellm.completion(
            model=kwargs.get("model", self.model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **self.kwargs,
        )
        return resp.choices[0].message.content
