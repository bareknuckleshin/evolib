from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

from evolib_agent_suite.schema import ResetResult, StepOutput, TaskSpec


class EnvironmentAdapter(ABC):
    domain: str = "generic"

    @abstractmethod
    def iter_tasks(self, limit: Optional[int] = None, split: str = "test") -> Iterable[TaskSpec]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task: TaskSpec) -> ResetResult:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> StepOutput:
        raise NotImplementedError

    def available_actions(self):
        return None

    def close(self) -> None:
        return None
