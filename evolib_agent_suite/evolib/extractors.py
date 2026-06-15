from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, derive_seed
from evolib_agent_suite.evolib.prompts import (
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    SELF_JUDGE_PROMPT,
    SELF_JUDGE_SYSTEM_PROMPT,
)
from evolib_agent_suite.llm.base import BaseLLM
from evolib_agent_suite.schema import Trajectory
from evolib_agent_suite.utils import clamp, extract_json_block


class AbstractionExtractor:
    def __init__(
        self,
        llm: BaseLLM,
        max_transcript_chars: int = 14000,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> None:
        self.llm = llm
        self.max_transcript_chars = max_transcript_chars
        self.sampling_policy = SamplingPolicy(sampling_config or SamplingConfig(strategy="topk"))
        self.last_sampling_metadata: Dict[str, Any] = {}

    def estimate_score(self, trajectory: Trajectory, prefer_env_reward: bool = False) -> Dict[str, Any]:
        if prefer_env_reward and trajectory.final_reward is not None:
            score = clamp(float(trajectory.final_reward))
            return {"score": score, "progress": trajectory.progress if trajectory.progress is not None else score, "notes": "environment reward"}
        transcript = trajectory.transcript()
        if len(transcript) > self.max_transcript_chars:
            transcript = transcript[-self.max_transcript_chars :]
        prompt = SELF_JUDGE_PROMPT.format(
            subgoals="\n".join(trajectory.task.subgoals) if trajectory.task.subgoals else "None",
            trajectory=transcript,
        )
        try:
            raw = self.llm.generate(SELF_JUDGE_SYSTEM_PROMPT, prompt, max_tokens=512)
            parsed = extract_json_block(raw)
            if isinstance(parsed, dict):
                return {
                    "score": clamp(float(parsed.get("score", 0.0))),
                    "progress": clamp(float(parsed.get("progress", parsed.get("score", 0.0)))),
                    "notes": str(parsed.get("notes", "")),
                    "raw": raw,
                }
        except Exception as exc:
            return self._fallback_score(trajectory, str(exc))
        return self._fallback_score(trajectory, "invalid self-judge JSON")

    def extract(self, trajectory: Trajectory, score: float) -> List[Dict[str, Any]]:
        transcript = trajectory.transcript()
        if len(transcript) > self.max_transcript_chars:
            transcript = transcript[-self.max_transcript_chars :]
        prompt = EXTRACTION_PROMPT.format(score=f"{score:.3f}", trajectory=transcript)
        try:
            raw = self.llm.generate(EXTRACTION_SYSTEM_PROMPT, prompt, max_tokens=1200)
            parsed = extract_json_block(raw)
            if isinstance(parsed, list):
                cleaned = [self._clean_item(x, trajectory.task.domain) for x in parsed if isinstance(x, dict)]
                cleaned = [x for x in cleaned if x.get("content")]
                if cleaned:
                    return self._select_candidates(cleaned, trajectory, 8)
        except Exception:
            pass
        return self._select_candidates(self._heuristic_extract(trajectory, score), trajectory, 8)

    def _clean_item(self, item: Dict[str, Any], domain: str) -> Dict[str, Any]:
        typ = str(item.get("type", "insight")).lower().strip()
        if typ not in {"skill", "insight"}:
            typ = "insight"
        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = [x.strip() for x in tags.split(",") if x.strip()]
        if domain and domain not in tags:
            tags.append(domain)
        return {
            "type": typ,
            "title": str(item.get("title") or typ.title()).strip()[:160],
            "content": str(item.get("content") or item.get("description") or "").strip()[:1800],
            "tags": tags[:12],
        }

    def _fallback_score(self, trajectory: Trajectory, reason: str) -> Dict[str, Any]:
        if trajectory.success is True:
            score = 1.0
        elif trajectory.final_reward is not None:
            score = clamp(float(trajectory.final_reward))
        elif any(s.done for s in trajectory.steps):
            score = 0.5
        else:
            score = min(0.45, 0.05 * len(trajectory.steps))
        return {"score": score, "progress": trajectory.progress if trajectory.progress is not None else score, "notes": f"fallback: {reason}"}

    def _heuristic_extract(self, trajectory: Trajectory, score: float) -> List[Dict[str, Any]]:
        domain = trajectory.task.domain.lower()
        actions = [s.action.lower() for s in trajectory.steps]
        items: List[Dict[str, Any]] = []
        if "webshop" in domain:
            if any(a.startswith("search[") for a in actions):
                items.append(
                    {
                        "type": "skill",
                        "title": "Search with core constraints first",
                        "content": "For shopping tasks, turn the instruction into a concise search query using product type plus the most restrictive attributes, then refine only after reading results.",
                        "tags": [domain, "search", "query-reformulation"],
                    }
                )
            items.append(
                {
                    "type": "insight",
                    "title": "Verify options before buying",
                    "content": "Before clicking Buy, check that the selected item satisfies required attributes, price constraints, and required options such as color, size, quantity, or compatibility.",
                    "tags": [domain, "verification"],
                }
            )
        elif "alfworld" in domain:
            verbs = [v for v in ["pick", "put", "clean", "heat", "cool", "go", "open", "close", "examine"] if any(a.startswith(v) for a in actions)]
            if verbs:
                items.append(
                    {
                        "type": "skill",
                        "title": "Household object workflow",
                        "content": "For household manipulation tasks, first locate and pick up the target object, perform required transformations such as clean/heat/cool, then navigate to the target receptacle and place or examine the object.",
                        "tags": [domain, "workflow"] + verbs[:5],
                    }
                )
            items.append(
                {
                    "type": "insight",
                    "title": "Transform before final placement",
                    "content": "If a goal asks for a cleaned, heated, or cooled object in a location, complete the transformation step before placing the object in the final receptacle.",
                    "tags": [domain, "ordering"],
                }
            )
        else:
            items.append(
                {
                    "type": "skill",
                    "title": "Use observations to choose grounded actions",
                    "content": "At every turn, compare the goal to the latest observation and choose an exact action supported by the environment rather than inventing unsupported commands.",
                    "tags": [domain or "generic", "grounding"],
                }
            )
            if score < 0.6:
                items.append(
                    {
                        "type": "insight",
                        "title": "Recover from low progress",
                        "content": "When progress is low, inspect the environment, list valid actions if possible, and choose the action that directly reduces uncertainty about the next subgoal.",
                        "tags": [domain or "generic", "recovery"],
                    }
                )
        return items

    def _select_candidates(self, candidates: List[Dict[str, Any]], trajectory: Trajectory, k: int) -> List[Dict[str, Any]]:
        context = f"composition_candidates:{trajectory.task.task_id}"
        derived_seed = derive_seed(self.sampling_policy.config.seed, context)
        scores = [self._candidate_score(candidate) for candidate in candidates]
        selected = self.sampling_policy.sample(candidates, scores, min(k, len(candidates)), self.sampling_policy.rng_for(context))
        self.last_sampling_metadata = {
            "composition_candidates": self.sampling_policy.metadata(derived_seed=derived_seed, context=context),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
        }
        return selected

    @staticmethod
    def _candidate_score(candidate: Dict[str, Any]) -> float:
        type_bonus = 1.1 if candidate.get("type") == "skill" else 1.0
        content = str(candidate.get("content", ""))
        return max(1e-6, type_bonus * min(1.0, max(0.1, len(content) / 600.0)))
