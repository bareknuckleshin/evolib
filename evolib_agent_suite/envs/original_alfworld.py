from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from evolib_agent_suite.envs.base import EnvironmentAdapter
from evolib_agent_suite.schema import ResetResult, StepOutput, TaskSpec


class OriginalALFWorldAdapter(EnvironmentAdapter):
    """Adapter for alfworld/alfworld text environment.

    Expected external setup:
        pip install alfworld[full]
        alfworld-download
    """

    domain = "alfworld_original"

    def __init__(
        self,
        config_path: Optional[str] = None,
        train_eval: str = "eval_out_of_distribution",
        batch_size: int = 1,
        max_steps: int = 50,
        **kwargs: Any,
    ) -> None:
        try:
            from alfworld.agents.environment import get_environment
            import alfworld.agents.modules.generic as generic
        except Exception as exc:
            raise RuntimeError("Could not import ALFWorld. Install alfworld[full] and run alfworld-download.") from exc

        self.generic = generic
        self.get_environment = get_environment
        self.config_path = config_path
        self.train_eval = train_eval
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.t = 0
        self.current_info: dict = {}
        self.current_goal = ""

        # generic.load_config reads sys.argv in many ALFWorld versions.
        old_argv = sys.argv[:]
        try:
            if config_path:
                sys.argv = [sys.argv[0], config_path]
            config = generic.load_config()
        finally:
            sys.argv = old_argv
        for k, v in kwargs.items():
            config[k] = v
        env_type = config["env"]["type"]
        self.env = get_environment(env_type)(config, train_eval=train_eval).init_env(batch_size=batch_size)

    def iter_tasks(self, limit: Optional[int] = None, split: str = "test") -> Iterable[TaskSpec]:
        n = limit or 134
        for i in range(n):
            yield TaskSpec(
                task_id=str(i),
                goal="ALFWorld goal is included in the initial observation.",
                split=split,
                domain=self.domain,
                action_hint="Use one exact admissible command from the ALFWorld TextWorld environment.",
            )

    def reset(self, task: TaskSpec) -> ResetResult:
        self.t = 0
        obs, info = self.env.reset()
        self.current_info = info or {}
        observation = str(obs[0] if isinstance(obs, (list, tuple)) else obs)
        goal = self._infer_goal(observation, self.current_info) or task.goal
        self.current_goal = goal
        return ResetResult(observation=observation, goal=goal, info=self.current_info)

    def available_actions(self):
        cmds = self.current_info.get("admissible_commands")
        if isinstance(cmds, (list, tuple)) and cmds:
            first = cmds[0]
            if isinstance(first, (list, tuple)):
                return list(first)
            return list(cmds)
        return None

    def step(self, action: str) -> StepOutput:
        self.t += 1
        obs, scores, dones, infos = self.env.step([action])
        self.current_info = infos or {}
        observation = str(obs[0] if isinstance(obs, (list, tuple)) else obs)
        score = float(scores[0] if isinstance(scores, (list, tuple)) else scores)
        done = bool(dones[0] if isinstance(dones, (list, tuple)) else dones)
        if self.t >= self.max_steps:
            done = True
        return StepOutput(
            observation=observation,
            reward=score,
            done=done,
            info=self.current_info,
            success=bool(done and score > 0),
            progress=score,
        )

    def _infer_goal(self, observation: str, info: dict) -> str:
        for key in ["goal", "task", "extra.gamefile"]:
            val = info.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # ALFWorld observations usually begin with task text followed by room description.
        for line in observation.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("you are in"):
                return line[:500]
        return ""

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
