from __future__ import annotations

from typing import Any, Iterable, List, Optional

from evolib_agent_suite.envs.base import EnvironmentAdapter
from evolib_agent_suite.schema import ResetResult, StepOutput, TaskSpec


class OriginalWebShopAdapter(EnvironmentAdapter):
    """Adapter for princeton-nlp/WebShop text environment.

    Expected external setup:
        git clone https://github.com/princeton-nlp/WebShop webshop
        cd webshop && ./setup.sh -d small
        export PYTHONPATH=/path/to/webshop:$PYTHONPATH
    """

    domain = "webshop_original"

    def __init__(
        self,
        num_products: int = 1000,
        observation_mode: str = "text",
        human_goals: int = 0,
        show_attrs: bool = False,
        max_steps: int = 15,
        sessions: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401 - registers env
        except Exception as exc:
            raise RuntimeError(
                "Could not import WebShop. Install/setup princeton-nlp/WebShop and put it on PYTHONPATH."
            ) from exc
        self.gym = gym
        self.max_steps = max_steps
        self.sessions = sessions
        self.env = gym.make(
            "WebAgentTextEnv-v0",
            observation_mode=observation_mode,
            num_products=num_products,
            human_goals=human_goals,
            show_attrs=show_attrs,
            **kwargs,
        )
        self.current_task: Optional[TaskSpec] = None
        self.t = 0

    @property
    def raw_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def iter_tasks(self, limit: Optional[int] = None, split: str = "test") -> Iterable[TaskSpec]:
        raw = self.raw_env
        goals = getattr(getattr(raw, "server", None), "goals", []) or []
        if self.sessions is not None:
            indices = self.sessions
        else:
            n = limit or min(len(goals), 500)
            indices = list(range(n))
        for idx in indices[: limit or len(indices)]:
            goal_text = ""
            metadata = {"session_int": idx}
            if isinstance(idx, int) and idx < len(goals):
                goal = goals[idx]
                goal_text = goal.get("instruction_text", "")
                metadata["goal"] = goal
            yield TaskSpec(
                task_id=str(idx),
                goal=goal_text or f"WebShop session {idx}",
                split=split,
                domain=self.domain,
                metadata=metadata,
                action_hint=(
                    "Use WebShop text actions: search[query] from the search page; "
                    "click[value] for visible buttons, product links, options, navigation, and Buy Now."
                ),
            )

    def reset(self, task: TaskSpec) -> ResetResult:
        self.current_task = task
        self.t = 0
        session_int = task.metadata.get("session_int")
        obs, info = self.env.reset(session=session_int)
        goal = task.goal
        try:
            goal = self.raw_env.get_instruction_text()
        except Exception:
            pass
        return ResetResult(observation=str(obs), goal=goal, info=info or {})

    def available_actions(self):
        try:
            return self.raw_env.get_available_actions()
        except Exception:
            return None

    def step(self, action: str) -> StepOutput:
        self.t += 1
        obs, reward, done, info = self.env.step(action)
        reward_f = float(reward or 0.0)
        if self.t >= self.max_steps:
            done = True
        available = self.available_actions()
        merged_info = info or {}
        if available is not None:
            merged_info["available_actions"] = available
        return StepOutput(
            observation=str(obs),
            reward=reward_f,
            done=bool(done),
            info=merged_info,
            success=bool(done and reward_f >= 1.0),
            progress=reward_f,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
