from __future__ import annotations

from typing import Iterable, Optional

from evolib_agent_suite.envs.base import EnvironmentAdapter
from evolib_agent_suite.schema import ResetResult, StepOutput, TaskSpec


class MockHouseholdAdapter(EnvironmentAdapter):
    domain = "mock"

    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps
        self.t = 0
        self.holding = False
        self.done = False

    def iter_tasks(self, limit: Optional[int] = None, split: str = "test") -> Iterable[TaskSpec]:
        n = limit or 3
        for i in range(n):
            yield TaskSpec(
                task_id=f"mock-{i}",
                goal="put apple in basket",
                split=split,
                domain=self.domain,
                action_hint="Use one of: look, pick apple, put apple in basket.",
                subgoals=["see apple", "pick apple", "put apple in basket"],
            )

    def reset(self, task: TaskSpec) -> ResetResult:
        self.t = 0
        self.holding = False
        self.done = False
        return ResetResult(observation="Room. You see an apple and a basket. You are not holding anything.", goal=task.goal)

    def available_actions(self):
        if self.done:
            return []
        if not self.holding:
            return ["look", "pick apple"]
        return ["look", "put apple in basket"]

    def step(self, action: str) -> StepOutput:
        self.t += 1
        action = (action or "").strip().lower()
        reward = 0.0
        if action == "look":
            obs = "You see an apple and a basket." + (" You are holding the apple." if self.holding else " You are not holding anything.")
        elif action == "pick apple" and not self.holding:
            self.holding = True
            obs = "Picked up apple. You are holding the apple. The basket is here."
            reward = 0.3
        elif action == "put apple in basket" and self.holding:
            self.done = True
            obs = "The apple is now in the basket. Goal complete."
            reward = 1.0
        else:
            obs = f"Nothing happens for unsupported action: {action}"
        if self.t >= self.max_steps and not self.done:
            self.done = True
        return StepOutput(observation=obs, reward=reward, done=self.done, success=reward >= 1.0, progress=reward)
