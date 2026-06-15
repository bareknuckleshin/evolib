"""AgentBoard plugin for EvoLib.

Copy this file into:
    AgentBoard/agentboard/agents/evolib_agent.py

Set:
    export EVOLIB_PROJECT_ROOT=/path/to/evolib_agent_suite

Then set AgentBoard config:
    agent:
      name: EvoLibAgent
      memory_size: 100
      need_goal: True
      library_path: ./results/evolib_agentboard_library.json
      k_skills: 4
      k_insights: 4
      retrieval_similarity_threshold: 0.05
      similarity_merge_threshold: 0.88
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.environ.get("EVOLIB_PROJECT_ROOT")
if ROOT and ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base_agent import BaseAgent  # type: ignore
from common.registry import registry  # type: ignore

from evolib_agent_suite.agents import EvoLibReActAgent
from evolib_agent_suite.evolib import AbstractionExtractor, EvolvingLibrary, SamplingConfig
from evolib_agent_suite.llm.base import BaseLLM
from evolib_agent_suite.schema import StepRecord, TaskSpec, Trajectory


class AgentBoardLLMBridge(BaseLLM):
    """Bridge AgentBoard's llm_model.generate(system, prompt)->(success,text)."""

    def __init__(self, llm_model: Any) -> None:
        self.llm_model = llm_model

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        success, text = self.llm_model.generate(system_prompt, user_prompt)
        if not success:
            # Still return text if AgentBoard provides it; otherwise use a safe fallback.
            return text or "Action: look"
        return text


@registry.register_agent("EvoLibAgent")
class EvoLibAgent(BaseAgent):
    def __init__(
        self,
        llm_model: Any,
        memory_size: int = 100,
        need_goal: bool = True,
        library_path: str = "./results/evolib_agentboard_library.json",
        k_skills: int = 4,
        k_insights: int = 4,
        retrieval_similarity_threshold: float = 0.05,
        similarity_merge_threshold: float = 0.88,
        action_hint: str = "Use one exact executable action accepted by the environment.",
        init_prompt_path: Optional[str] = None,
        instruction: str = "",
        examples: Optional[List[str]] = None,
        system_message: str = "You are a helpful assistant.",
        check_actions: Optional[str] = None,
        check_inventory: Optional[str] = None,
        use_parser: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bridge = AgentBoardLLMBridge(llm_model)
        sampling_cfg = SamplingConfig.from_dict(sampling or kwargs.get("sampling"), default_seed=int(kwargs.get("seed", 0)))
        self.library = EvolvingLibrary(
            path=library_path,
            retrieval_similarity_threshold=retrieval_similarity_threshold,
            similarity_merge_threshold=similarity_merge_threshold,
            seed=int(kwargs.get("seed", 0)),
            sampling_config=sampling_cfg,
        )
        self.extractor = AbstractionExtractor(self.bridge, sampling_config=sampling_cfg)
        self.core = EvoLibReActAgent(
            llm=self.bridge,
            library=self.library,
            memory_size=memory_size,
            action_hint=action_hint,
        )
        self.memory_size = memory_size
        self.need_goal = need_goal
        self.k_skills = k_skills
        self.k_insights = k_insights
        self.action_hint = action_hint
        self.instruction = instruction
        self.examples = examples or []
        self.system_message = system_message
        self.check_actions = check_actions
        self.check_inventory = check_inventory
        self.use_parser = use_parser
        self.current_task: Optional[TaskSpec] = None
        self.current_obs: str = ""
        self.current_trajectory: Optional[Trajectory] = None
        self.last_thought: str = ""
        self.last_raw: str = ""
        self.episode_counter = 0

    def reset(self, goal: str, init_obs: str, init_act: Optional[str] = None):
        self._finalize_previous_episode()
        self.episode_counter += 1
        task = TaskSpec(
            task_id=f"agentboard-{self.episode_counter}",
            goal=goal if self.need_goal else init_obs,
            domain="agentboard",
            action_hint=self._combined_action_hint(),
        )
        entries = self.library.retrieve(
            query=f"agentboard\n{goal}\n{init_obs}",
            k_skills=self.k_skills,
            k_insights=self.k_insights,
            sample=True,
            task_id=task.task_id,
            episode_id=self.episode_counter,
        )
        self.core.reset(task, entries)
        self.current_task = task
        self.current_obs = init_obs
        self.current_trajectory = Trajectory(
            task=task,
            initial_observation=init_obs,
            used_entry_ids=[e.id for e in entries],
        )
        if init_act:
            self.current_trajectory.add_step(
                StepRecord(t=0, observation="", thought="", action=init_act, next_observation=init_obs)
            )

    def run(self, init_prompt_dict: Optional[Dict[str, Any]] = None):
        if self.current_task is None:
            return False, "look"
        decision = self.core.act(self.current_obs, available_actions=None)
        self.last_thought = decision.thought
        self.last_raw = decision.raw_response
        return True, decision.action

    def update(self, action: str, state: str):
        if self.current_trajectory is None:
            return
        step = StepRecord(
            t=len(self.current_trajectory.steps),
            observation=self.current_obs,
            thought=self.last_thought,
            action=action,
            next_observation=state,
            reward=None,
            done=False,
            info={"raw_response": self.last_raw},
        )
        self.current_trajectory.add_step(step)
        self.core.observe_step(step)
        self.current_obs = state

    def _finalize_previous_episode(self):
        traj = self.current_trajectory
        if traj is None or not traj.steps:
            return
        score_info = self.extractor.estimate_score(traj, prefer_env_reward=False)
        score = float(score_info.get("score", 0.0))
        traj.score_estimate = score
        candidates = self.extractor.extract(traj, score)
        new_ids = self.library.add_or_merge_many(
            candidates,
            parents=traj.used_entry_ids,
            task_id=traj.task.task_id,
            score=score,
        )
        self.library.update_after_episode(
            traj.used_entry_ids,
            new_ids,
            score=score,
            success=None,
            task_id=traj.task.task_id,
            episode_id=self.episode_counter,
        )
        self.library.save()
        self.current_trajectory = None

    def _combined_action_hint(self) -> str:
        hints = [self.action_hint]
        if self.check_actions:
            hints.append(f"Use this command when needed to inspect valid actions: {self.check_actions}")
        if self.check_inventory:
            hints.append("Use inventory when needed to check carried objects.")
        return "\n".join(hints)

    @classmethod
    def from_config(cls, llm_model: Any, config: Dict[str, Any]):
        return cls(
            llm_model=llm_model,
            memory_size=config.get("memory_size", 100),
            need_goal=config.get("need_goal", True),
            library_path=config.get("library_path", "./results/evolib_agentboard_library.json"),
            k_skills=config.get("k_skills", 4),
            k_insights=config.get("k_insights", 4),
            retrieval_similarity_threshold=config.get("retrieval_similarity_threshold", 0.05),
            similarity_merge_threshold=config.get("similarity_merge_threshold", 0.88),
            action_hint=config.get("action_hint", "Use one exact executable action accepted by the environment."),
            init_prompt_path=config.get("init_prompt_path"),
            instruction=config.get("instruction", ""),
            examples=config.get("examples", []),
            system_message=config.get("system_message", "You are a helpful assistant."),
            check_actions=config.get("check_actions"),
            check_inventory=config.get("check_inventory"),
            use_parser=config.get("use_parser", True),
            sampling=config.get("sampling"),
            seed=config.get("seed", 0),
        )
