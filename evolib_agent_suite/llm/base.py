from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from evolib_agent_suite.utils import extract_json_block


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> Any:
        text = self.generate(system_prompt, user_prompt, **kwargs)
        parsed = extract_json_block(text)
        if parsed is None:
            raise ValueError(f"Model did not return valid JSON. Text begins: {text[:400]!r}")
        return parsed
