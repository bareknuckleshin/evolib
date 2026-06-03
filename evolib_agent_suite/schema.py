from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    task_id: str
    goal: str
    split: str = "test"
    domain: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_hint: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)


@dataclass
class ResetResult:
    observation: str
    goal: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutput:
    observation: str
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    success: Optional[bool] = None
    progress: Optional[float] = None


@dataclass
class StepRecord:
    t: int
    observation: str
    action: str
    thought: str = ""
    next_observation: str = ""
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    action: str
    thought: str = ""
    raw_response: str = ""
    used_entry_ids: List[str] = field(default_factory=list)


@dataclass
class Trajectory:
    task: TaskSpec
    initial_observation: str
    steps: List[StepRecord] = field(default_factory=list)
    used_entry_ids: List[str] = field(default_factory=list)
    final_reward: Optional[float] = None
    success: Optional[bool] = None
    progress: Optional[float] = None
    score_estimate: Optional[float] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)
        if step.reward is not None:
            self.final_reward = step.reward
        if step.done:
            self.metadata["terminated_at"] = step.t

    def transcript(self, max_chars_per_obs: int = 1600) -> str:
        def trim(text: str) -> str:
            text = text or ""
            if len(text) <= max_chars_per_obs:
                return text
            return text[:max_chars_per_obs] + " ...[truncated]"

        chunks = [
            f"Task id: {self.task.task_id}",
            f"Domain: {self.task.domain}",
            f"Goal: {self.task.goal}",
            f"Initial observation: {trim(self.initial_observation)}",
        ]
        for s in self.steps:
            chunks.append(
                "\n".join(
                    [
                        f"Step {s.t}",
                        f"Observation: {trim(s.observation)}",
                        f"Thought: {s.thought}",
                        f"Action: {s.action}",
                        f"Reward: {s.reward}",
                        f"Done: {s.done}",
                        f"Next observation: {trim(s.next_observation)}",
                    ]
                )
            )
        chunks.append(f"Final reward: {self.final_reward}")
        chunks.append(f"Success: {self.success}")
        chunks.append(f"Progress: {self.progress}")
        return "\n\n".join(chunks)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
