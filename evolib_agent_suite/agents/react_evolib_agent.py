from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from evolib_agent_suite.evolib.library import EvolvingLibrary, LibraryEntry
from evolib_agent_suite.evolib.prompts import ACTION_PROMPT, ACTION_SYSTEM_PROMPT
from evolib_agent_suite.llm.base import BaseLLM
from evolib_agent_suite.schema import ActionDecision, StepRecord, TaskSpec


DEFAULT_ACTION_HINT = "Return one executable text action exactly as the environment expects."


class EvoLibReActAgent:
    """ReAct-style agent augmented with retrieved EvoLib entries."""

    def __init__(
        self,
        llm: BaseLLM,
        library: EvolvingLibrary,
        memory_size: int = 12,
        action_hint: str = DEFAULT_ACTION_HINT,
        max_prompt_chars: int = 18000,
    ) -> None:
        self.llm = llm
        self.library = library
        self.memory_size = memory_size
        self.action_hint = action_hint
        self.max_prompt_chars = max_prompt_chars
        self.task: Optional[TaskSpec] = None
        self.entries: List[LibraryEntry] = []
        self.history: List[StepRecord] = []
        self.last_decision: Optional[ActionDecision] = None

    def reset(self, task: TaskSpec, entries: Sequence[LibraryEntry]) -> None:
        self.task = task
        self.entries = list(entries)
        self.history = []
        self.last_decision = None

    @property
    def used_entry_ids(self) -> List[str]:
        return [e.id for e in self.entries]

    def observe_step(self, step: StepRecord) -> None:
        self.history.append(step)

    def act(self, observation: str, available_actions: Optional[Sequence[str] | dict] = None) -> ActionDecision:
        if self.task is None:
            raise RuntimeError("Agent must be reset before act().")
        library_block = self.library.format_for_prompt(self.entries)
        history = self._format_history()
        available = self._format_available_actions(available_actions)
        action_hint = self.task.action_hint or self.action_hint
        prompt = ACTION_PROMPT.format(
            goal=self.task.goal,
            library_block=library_block,
            action_hint=action_hint,
            history=history,
            observation=self._trim(observation, 5000),
            available_actions=available,
        )
        prompt = self._trim_left(prompt, self.max_prompt_chars)
        raw = self.llm.generate(ACTION_SYSTEM_PROMPT, prompt, max_tokens=384)
        thought, action = self._parse_action(raw)
        decision = ActionDecision(action=action, thought=thought, raw_response=raw, used_entry_ids=self.used_entry_ids)
        self.last_decision = decision
        return decision

    def _format_history(self) -> str:
        if not self.history:
            return "No previous actions."
        lines: List[str] = []
        for s in self.history[-self.memory_size :]:
            lines.append(f"Observation: {self._trim(s.observation, 900)}")
            if s.thought:
                lines.append(f"Thought: {s.thought}")
            lines.append(f"Action: {s.action}")
            lines.append(f"Result: reward={s.reward}, done={s.done}")
            if s.next_observation:
                lines.append(f"Next observation: {self._trim(s.next_observation, 900)}")
        return "\n".join(lines)

    def _format_available_actions(self, available_actions: Optional[Sequence[str] | dict]) -> str:
        if not available_actions:
            return "Not provided. Infer from the observation, but do not invent unsupported commands."
        if isinstance(available_actions, dict):
            parts = []
            if available_actions.get("has_search_bar"):
                parts.append("- search[<query>]")
            for a in available_actions.get("clickables", [])[:80]:
                parts.append(f"- click[{a}]")
            for a in available_actions.get("admissible_commands", [])[:80]:
                parts.append(f"- {a}")
            return "\n".join(parts) if parts else str(available_actions)
        return "\n".join(f"- {a}" for a in list(available_actions)[:120])

    def _parse_action(self, raw: str) -> tuple[str, str]:
        raw = raw or ""
        thought = ""
        action = ""
        m_thought = re.search(r"Thought\s*:\s*(.*)", raw, re.IGNORECASE)
        if m_thought:
            thought = m_thought.group(1).strip().splitlines()[0]
        m_action = re.search(r"Action\s*:\s*(.*)", raw, re.IGNORECASE)
        if m_action:
            action = m_action.group(1).strip().splitlines()[0]
        else:
            # Last non-empty line fallback.
            lines = [x.strip() for x in raw.splitlines() if x.strip()]
            action = lines[-1] if lines else "look"
        action = action.strip().strip("`'\"")
        # Remove common bullet/list prefixes.
        action = re.sub(r"^[-*\d.\s]+", "", action).strip()
        return thought, action

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        text = text or ""
        return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"

    @staticmethod
    def _trim_left(text: str, max_chars: int) -> str:
        text = text or ""
        return text if len(text) <= max_chars else text[-max_chars:]
