from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolib_agent_suite.agents import EvoLibReActAgent
from evolib_agent_suite.envs import build_env
from evolib_agent_suite.evolib import (
    AbstractionExtractor,
    EvolvingLibrary,
    build_library_storage,
)
from evolib_agent_suite.llm import build_llm
from evolib_agent_suite.schema import StepRecord, TaskSpec, Trajectory
from evolib_agent_suite.utils import append_jsonl, ensure_dir, load_config


def run(config: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    out_dir = ensure_dir(config.get("output_dir", "runs/evolib"))
    result_path = out_dir / "trajectories.jsonl"
    metrics_path = out_dir / "metrics.json"
    library_cfg = config.get("library", {})
    library_path = library_cfg.get("path", str(out_dir / "library.json"))
    storage_cfg = dict(library_cfg.get("storage", {}))
    if "backend" not in storage_cfg and "storage_backend" in library_cfg:
        storage_cfg["backend"] = library_cfg["storage_backend"]
    storage = build_library_storage(library_path, storage_cfg)

    llm = build_llm(config.get("llm", {"provider": "heuristic"}))
    env = build_env(config.get("env", {"backend": "mock"}))
    library = EvolvingLibrary(
        path=library_path,
        storage=storage,
        similarity_merge_threshold=float(
            library_cfg.get("similarity_merge_threshold", 0.88)
        ),
        retrieval_similarity_threshold=float(
            library_cfg.get("retrieval_similarity_threshold", 0.05)
        ),
        seed=int(config.get("seed", 0)),
    )
    extractor = AbstractionExtractor(llm)

    agent_cfg = config.get("agent", {})
    agent = EvoLibReActAgent(
        llm=llm,
        library=library,
        memory_size=int(agent_cfg.get("memory_size", 12)),
        action_hint=agent_cfg.get("action_hint", "Return one exact action string."),
        max_prompt_chars=int(agent_cfg.get("max_prompt_chars", 18000)),
    )

    eval_cfg = config.get("eval", {})
    max_steps = int(eval_cfg.get("max_steps", getattr(env, "max_steps", 20)))
    split = eval_cfg.get("split", "test")
    prefer_env_reward = bool(eval_cfg.get("library_update_uses_env_reward", False))
    k_skills = int(library_cfg.get("k_skills", 4))
    k_insights = int(library_cfg.get("k_insights", 4))
    sample_library = bool(library_cfg.get("sample", True))
    retrieval_policy_config = {
        "k_skills": k_skills,
        "k_insights": k_insights,
        "sample": sample_library,
        "similarity_threshold": library.retrieval_similarity_threshold,
    }
    composition_policy_config = {
        "similarity_merge_threshold": library.similarity_merge_threshold,
        "alpha_ig": library.alpha_ig,
        "beta_future_ig": library.beta_future_ig,
        "ema_decay": library.ema_decay,
    }
    library.record_policy_snapshot(
        {
            "retrieval_policy_config": retrieval_policy_config,
            "composition_policy_config": composition_policy_config,
            "storage_backend": storage.backend,
            "timestamp": time.time(),
        }
    )

    metrics: Dict[str, Any] = {
        "episodes": 0,
        "successes": 0,
        "reward_sum": 0.0,
        "score_estimate_sum": 0.0,
        "progress_sum": 0.0,
        "library_size_start": len(library),
        "library_path": str(library.path),
        "started_at": time.time(),
    }

    for task in env.iter_tasks(limit=limit or eval_cfg.get("limit"), split=split):
        reset = env.reset(task)
        if reset.goal:
            task.goal = reset.goal
        retrieval_trace: Dict[str, Any] = {}
        entries = library.retrieve(
            query=f"{task.domain}\n{task.goal}\n{reset.observation}",
            k_skills=k_skills,
            k_insights=k_insights,
            sample=sample_library,
            trace=retrieval_trace,
        )
        agent.reset(task, entries)
        traj = Trajectory(
            task=task,
            initial_observation=reset.observation,
            used_entry_ids=[e.id for e in entries],
            metadata={"reset_info": _jsonable(reset.info)},
        )
        obs = reset.observation
        final_out = None
        for t in range(max_steps):
            available = env.available_actions()
            decision = agent.act(obs, available_actions=available)
            out = env.step(decision.action)
            step = StepRecord(
                t=t,
                observation=obs,
                thought=decision.thought,
                action=decision.action,
                next_observation=out.observation,
                reward=out.reward,
                done=out.done,
                info=_jsonable(out.info),
            )
            traj.add_step(step)
            agent.observe_step(step)
            obs = out.observation
            final_out = out
            if out.done:
                break

        if final_out is not None:
            traj.final_reward = final_out.reward
            traj.success = final_out.success
            traj.progress = final_out.progress
        score_info = extractor.estimate_score(traj, prefer_env_reward=prefer_env_reward)
        score = float(score_info["score"])
        traj.score_estimate = score
        traj.notes = str(score_info.get("notes", ""))
        if traj.progress is None:
            traj.progress = score_info.get("progress")

        candidates = extractor.extract(traj, score=score)
        merge_events_before = len(library.merge_events)
        new_ids = library.add_or_merge_many(
            candidates,
            parents=traj.used_entry_ids,
            task_id=task.task_id,
            score=score,
        )
        consolidation_decisions = library.merge_events[merge_events_before:]
        ig_event = library.update_after_episode(
            retrieved_ids=traj.used_entry_ids,
            new_ids=new_ids,
            score=score,
            success=traj.success,
        )
        retrieval_event = {
            "task_id": task.task_id,
            "candidate_count": retrieval_trace.get("candidate_count", 0),
            "selected_entry_ids": retrieval_trace.get(
                "selected_entry_ids", traj.used_entry_ids
            ),
            "retrieval_policy_config": retrieval_policy_config,
            "timestamp": time.time(),
        }
        library.record_retrieval_event(retrieval_event)
        library.save()

        record = traj.to_dict()
        record["evolib"] = {
            "retrieval_candidate_count": retrieval_trace.get("candidate_count", 0),
            "retrieved_entry_ids": traj.used_entry_ids,
            "selected_entry_ids": retrieval_trace.get(
                "selected_entry_ids", traj.used_entry_ids
            ),
            "retrieval_policy_config": retrieval_policy_config,
            "composition_policy_config": composition_policy_config,
            "ig_baseline_value": ig_event["baseline_value"],
            "immediate_ig": ig_event["immediate_ig"],
            "propagated_fig_credits": ig_event["propagated_fig_credits"],
            "consolidation_decisions": consolidation_decisions,
            "new_or_updated_entry_ids": new_ids,
            "score_info": _jsonable(score_info),
            "candidate_count": len(candidates),
            "library_size": len(library),
        }
        append_jsonl(result_path, record)

        metrics["episodes"] += 1
        metrics["successes"] += 1 if traj.success else 0
        metrics["reward_sum"] += float(traj.final_reward or 0.0)
        metrics["score_estimate_sum"] += score
        metrics["progress_sum"] += float(traj.progress or 0.0)
        metrics.update(_summarize(metrics, len(library)))
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "episode": metrics["episodes"],
                    "task_id": task.task_id,
                    "success": traj.success,
                    "reward": traj.final_reward,
                    "score_estimate": score,
                    "library_size": len(library),
                },
                ensure_ascii=False,
            )
        )

    env.close()
    library.save()
    metrics["finished_at"] = time.time()
    metrics.update(_summarize(metrics, len(library)))
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def _summarize(metrics: Dict[str, Any], library_size: int) -> Dict[str, Any]:
    n = max(1, int(metrics.get("episodes", 0)))
    return {
        "success_rate": metrics.get("successes", 0) / n,
        "avg_reward": metrics.get("reward_sum", 0.0) / n,
        "avg_score_estimate": metrics.get("score_estimate_sum", 0.0) / n,
        "avg_progress": metrics.get("progress_sum", 0.0) / n,
        "library_size_current": library_size,
    }


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(x) for x in obj]
        return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvoLib agent evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Override number of tasks."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = run(cfg, limit=args.limit)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
