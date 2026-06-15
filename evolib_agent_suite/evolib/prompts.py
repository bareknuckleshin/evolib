from __future__ import annotations

ACTION_SYSTEM_PROMPT = """You are a careful text-environment agent. Use prior reusable skills when relevant, but ground every action in the current observation. Return exactly one action for the environment."""

ACTION_PROMPT = """Goal:
{goal}

Reusable EvoLib entries:
{library_block}

Action format / environment rules:
{action_hint}

Current trajectory:
{history}

Latest observation:
{observation}

Available actions, if provided:
{available_actions}

Decide the next step. Keep the thought short. End with a single executable action.
Use this format:
Thought: <one sentence>
Action: <exact action string>
"""

SELF_JUDGE_SYSTEM_PROMPT = """You evaluate an agent trajectory without using hidden ground-truth labels. Estimate how much of the stated goal was achieved from the visible observations and actions only."""

SELF_JUDGE_PROMPT = """Return JSON only with keys: score, progress, notes.
- score: float from 0 to 1 estimating task completion.
- progress: float from 0 to 1 estimating subgoal progress.
- notes: short explanation.

Known subgoals, if any:
{subgoals}

Trajectory:
{trajectory}
"""

EXTRACTION_SYSTEM_PROMPT = """You extract reusable knowledge from agent trajectories. Store abstractions, not raw episode memories."""

EXTRACTION_PROMPT = """Extract 2-6 reusable EvoLib abstractions from this trajectory.
Use two types:
- skill: a reusable procedure/workflow for similar future tasks.
- insight: a concise lesson about mistakes, checks, or corrective strategies.

Rules:
- Do not memorize exact product IDs, object IDs, or episode-specific trivia unless it is a general pattern.
- Prefer domain-general action workflows and verification checks.
- Make each item self-contained.
- Return JSON only: a list of objects with keys type, title, content, tags.

Estimated score: {score}
Trajectory:
{trajectory}
"""

LLM_MERGE_SYSTEM_PROMPT = """You consolidate reusable EvoLib skills and insights. Preserve generally useful knowledge, remove duplicates, and do not invent episode-specific facts. Return JSON only."""

LLM_MERGE_PROMPT = """Merge the existing EvoLib entry with the candidate entry.

Inputs:
Existing {existing_type}:
Title: {existing_title}
Content: {existing_content}
Tags: {existing_tags}

Candidate {candidate_type}:
Title: {candidate_title}
Content: {candidate_content}
Tags: {candidate_tags}

Task context:
{task_context}

Candidate episode score: {score}

Return JSON only with keys:
- title: concise merged title
- content: self-contained merged reusable skill/insight content
- tags: short list of tags
- rationale: one sentence explaining what was retained or changed
"""
