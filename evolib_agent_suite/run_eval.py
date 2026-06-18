from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolib_agent_suite.agents import EvoLibReActAgent
from evolib_agent_suite.embeddings import build_embedding_function
from evolib_agent_suite.envs import build_env
from evolib_agent_suite.evolib import AbstractionExtractor, CompositionConfig, ConsolidationConfig, EvolvingLibrary, IGConfig, RetrievalConfig
from evolib_agent_suite.llm import build_llm
from evolib_agent_suite.schema import StepRecord, TaskSpec, Trajectory
from evolib_agent_suite.utils import append_jsonl, ensure_dir, load_config


def run(config: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    """설정 파일 하나로 EvoLib 에이전트 평가 전체 파이프라인을 실행한다.

    Original WebShop 테스트도 이 함수가 공통으로 담당한다. 설정에서
    `env.backend: original_webshop`을 선택하면 WebShop 어댑터를 만들고, 각
    session을 `TaskSpec`으로 순회하면서 검색(retrieval) → 행동(action) →
    추상화(extraction) → 라이브러리 업데이트(update) 순서로 episode를 처리한다.
    """
    # 실행 산출물: trajectory는 episode별 상세 로그, metrics는 누적 요약 지표,
    # library는 EvoLib가 학습/축적한 skill·insight 저장소다.
    out_dir = ensure_dir(config.get("output_dir", "runs/evolib"))
    result_path = out_dir / "trajectories.jsonl"
    metrics_path = out_dir / "metrics.json"
    library_path = config.get("library", {}).get("path", str(out_dir / "library.json"))
    library_storage_cfg = config.get("library", {}).get("storage", {})
    library_storage_backend = str(library_storage_cfg.get("backend", config.get("library", {}).get("storage_backend", "json")))

    # LLM, 환경, 임베딩 함수는 설정으로 교체 가능하다. Original WebShop 실험에서는
    # env 설정만 바꾸면 동일한 평가 루프를 그대로 재사용한다.
    llm = build_llm(config.get("llm", {"provider": "heuristic"}))
    env = build_env(config.get("env", {"backend": "mock"}))
    library_cfg = config.get("library", {})
    embedding_cfg = config.get("embedding", library_cfg.get("embedding", {}))
    embedding_fn = build_embedding_function(embedding_cfg)
    consolidation_cfg = config.get("consolidation", {})
    ig_cfg = IGConfig(**library_cfg.get("ig", {}))
    library = EvolvingLibrary(
        path=library_path,
        similarity_merge_threshold=float(library_cfg.get("similarity_merge_threshold", 0.88)),
        retrieval_similarity_threshold=float(library_cfg.get("retrieval_similarity_threshold", 0.05)),
        seed=int(config.get("seed", 0)),
        consolidation_config=ConsolidationConfig(
            enabled=bool(consolidation_cfg.get("enabled", True)),
            similarity_threshold=float(consolidation_cfg.get("similarity_threshold", library_cfg.get("similarity_merge_threshold", 0.88))),
            candidate_top_n=int(consolidation_cfg.get("candidate_top_n", 1)),
            merge_strategy=str(consolidation_cfg.get("merge_strategy", "replace_if_longer")),
            score_policy=str(consolidation_cfg.get("score_policy", "ema_score")),
            allow_cross_type_merge=bool(consolidation_cfg.get("allow_cross_type_merge", False)),
            merge_history_limit=int(consolidation_cfg.get("merge_history_limit", 20)),
            ema_decay=float(consolidation_cfg.get("ema_decay", 0.85)),
        ),
        llm=llm,
        ema_decay=ig_cfg.ema_decay,
        ig_config=ig_cfg,
        storage_backend=library_storage_backend,
        embedding_fn=embedding_fn,
    )
    extractor = AbstractionExtractor(llm)

    agent_cfg = config.get("agent", {})
    composition_cfg = _composition_config(agent_cfg.get("composition", {}))
    # EvoLib ReAct agent는 검색된 library entry를 프롬프트 컨텍스트로 받아서
    # thought/action을 생성한다. action_hint는 WebShop처럼 문법이 엄격한 환경에서
    # `search[...]`, `click[...]` 형태를 지키게 하는 중요한 프롬프트 데이터다.
    agent = EvoLibReActAgent(
        llm=llm,
        library=library,
        memory_size=int(agent_cfg.get("memory_size", 12)),
        action_hint=agent_cfg.get("action_hint", "Return one exact action string."),
        max_prompt_chars=int(agent_cfg.get("max_prompt_chars", 18000)),
        composition_config=composition_cfg,
        seed=int(config.get("seed", 0)),
    )

    eval_cfg = config.get("eval", {})
    max_steps = int(eval_cfg.get("max_steps", getattr(env, "max_steps", 20)))
    split = eval_cfg.get("split", "test")
    prefer_env_reward = bool(eval_cfg.get("library_update_uses_env_reward", False))
    k_skills = int(library_cfg.get("k_skills", 4))
    k_insights = int(library_cfg.get("k_insights", 4))
    sample_library = bool(library_cfg.get("sample", True))
    library_cfg = config.get("library", {})
    # RetrievalConfig는 episode 시작 시 어떤 skill/insight를 가져올지 정하는 핵심
    # 알고리즘 파라미터다. 유사도 임계값, 후보 풀 크기, weighted/top-k sampling,
    # temperature/top-p/epsilon 등이 library 탐색과 재사용 방식에 직접 영향을 준다.
    retrieval_config = RetrievalConfig(
        k_skills=int(library_cfg.get("k_skills", 4)),
        k_insights=int(library_cfg.get("k_insights", 4)),
        similarity_threshold=float(
            library_cfg.get("similarity_threshold", library_cfg.get("retrieval_similarity_threshold", 0.05))
        ),
        candidate_pool_multiplier=int(library_cfg.get("candidate_pool_multiplier", 4)),
        sampling_strategy=str(
            library_cfg.get("sampling_strategy", "weighted" if bool(library_cfg.get("sample", True)) else "topk")
        ),
        temperature=float(library_cfg.get("temperature", 1.0)),
        epsilon=float(library_cfg.get("epsilon", 0.1)),
        weight_alpha=float(library_cfg.get("weight_alpha", 1.0)),
        similarity_alpha=float(library_cfg.get("similarity_alpha", 1.0)),
        top_p=float(library_cfg.get("top_p", 0.9)),
        seed=int(config.get("seed", 0)),
        without_replacement=_as_bool(library_cfg.get("without_replacement", True)),
    )

    # 실행 당시의 retrieval/composition/consolidation/IG 정책을 library에 남긴다.
    # 나중에 trajectory만 보더라도 어떤 정책으로 생성된 결과인지 재현·비교할 수 있다.
    library.policy_snapshots.append({
        "created_at": time.time(),
        "retrieval_policy": asdict(retrieval_config),
        "composition_policy": asdict(composition_cfg),
        "consolidation_policy": asdict(library.consolidation_config),
        "ig_policy": asdict(library.ig_config),
        "storage_backend": library_storage_backend,
    })

    metrics: Dict[str, Any] = {
        "episodes": 0,
        "successes": 0,
        "reward_sum": 0.0,
        "score_estimate_sum": 0.0,
        "progress_sum": 0.0,
        "library_size_start": len(library),
        "library_path": str(library.path),
        "library_storage_backend": library_storage_backend,
        "started_at": time.time(),
    }

    for task in env.iter_tasks(limit=limit or eval_cfg.get("limit"), split=split):
        # 1) 환경을 session 단위로 초기화하고, WebShop 원본 instruction이 있으면
        # TaskSpec goal을 최신 값으로 덮어쓴다.
        reset = env.reset(task)
        if reset.goal:
            task.goal = reset.goal
        retrieval_config.context_id = task.task_id
        # 2) 현재 task/goal/초기 관측을 query로 사용해 관련 skill·insight를 검색한다.
        # 이 검색 결과가 episode 동안 에이전트 프롬프트의 장기 기억 역할을 한다.
        retrieved = library.retrieve_with_metadata(
            query=f"{task.domain}\n{task.goal}\n{reset.observation}",
            config=retrieval_config,
        )
        entries = [item.entry for item in retrieved]
        agent.reset(task, entries)
        # 3) composition 단계가 켜져 있으면 검색된 entry들을 조합한 후보 solution을
        # 만들고, 해당 entry id를 trajectory lineage로 기록한다.
        candidate = agent.candidate_solution
        composed_entry_ids = candidate.entry_ids if candidate else []
        traj = Trajectory(
            task=task,
            initial_observation=reset.observation,
            used_entry_ids=composed_entry_ids,
            metadata={
                "reset_info": _jsonable(reset.info),
                "candidate_solution_id": candidate.id if candidate else None,
                "composition_type": candidate.composition_type if candidate else None,
                "composed_entry_ids": composed_entry_ids,
                "retrieval_candidate_count": int(library.last_retrieval_event.get("candidate_count", len(retrieved))),
                "selected_entry_ids": [item.entry.id for item in retrieved],
                "retrieval_policy_config": asdict(retrieval_config),
                "composition_policy_config": asdict(composition_cfg),
            },
        )
        obs = reset.observation
        final_out = None
        for t in range(max_steps):
            # 4) 매 step마다 환경의 available action을 넘긴다. WebShop은 잘못된 문자열
            # 액션에 민감하므로, 가능한 액션 목록은 LLM의 action 형식 오류를 줄인다.
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
            # 5) 환경 reward/progress/success를 trajectory의 최종 성능 지표로 저장한다.
            traj.final_reward = final_out.reward
            traj.success = final_out.success
            traj.progress = final_out.progress
        # 6) EvoLib 논문식 no-external-feedback 설정에서는 LLM 기반 score estimate를
        # 사용할 수 있고, 설정에 따라 WebShop reward를 library update에 사용할 수도 있다.
        score_info = extractor.estimate_score(traj, prefer_env_reward=prefer_env_reward)
        score = float(score_info["score"])
        traj.score_estimate = score
        traj.notes = str(score_info.get("notes", ""))
        if traj.progress is None:
            traj.progress = score_info.get("progress")

        # 7) 완료된 trajectory에서 재사용 가능한 skill/insight 후보를 추출하고,
        # 기존 library entry와 유사하면 merge, 새 지식이면 add한다.
        candidates = extractor.extract(traj, score=score)
        new_ids = library.add_or_merge_many(
            candidates,
            parents=traj.used_entry_ids,
            task_id=task.task_id,
            score=score,
        )
        # 8) Information Gain(IG) 업데이트: episode score와 baseline 차이를 이용해
        # 이번 episode에서 사용/생성된 entry들의 기여도를 library에 전파한다.
        ig_info = library.update_after_episode(
            retrieved_ids=traj.used_entry_ids,
            new_ids=new_ids,
            score=score,
            success=traj.success,
            context={
                "task": task,
                "task_id": task.task_id,
                "domain": task.domain,
                "retrieved_ids": traj.used_entry_ids,
                "retrieved_count": len(traj.used_entry_ids),
            },
        )
        library.save()

        # trajectory JSONL은 디버깅과 후처리의 핵심 데이터다. WebShop 관측/액션뿐 아니라
        # retrieval 선택 이유, sampling seed, composition 정책, IG credit까지 함께 남긴다.
        record = traj.to_dict()
        record["evolib"] = {            
            "candidate_solution_id": traj.metadata.get("candidate_solution_id"),
            "composition_type": traj.metadata.get("composition_type"),
            "composed_entry_ids": traj.metadata.get("composed_entry_ids", []),
            "retrieved_entry_ids": traj.used_entry_ids,
            "retrieval_candidate_count": int(library.last_retrieval_event.get("candidate_count", len(retrieved))),
            "selected_entry_ids": [item.entry.id for item in retrieved],
            "retrieval_policy_config": asdict(retrieval_config),
            "composition_policy_config": asdict(composition_cfg),
            "retrieved_entries": [
                {
                    "id": item.entry.id,
                    "similarity": item.similarity,
                    "retrieval_weight": item.retrieval_weight,
                    "rank": item.rank,
                    "selected_by": item.selected_by,
                    "sampling_seed": item.sampling_seed,
                    "sampling_base_seed": item.sampling_base_seed,
                    "sampling_context_id": item.sampling_context_id,
                }
                for item in retrieved
            ],
            "new_or_updated_entry_ids": new_ids,
            "score_info": _jsonable(score_info),
            "candidate_count": len(candidates),
            "library_size": len(library),
            "lineage_edge_count": len(library.lineage_edges),
            "fig_credit_events": {
                "count": len(library.last_fig_credit_events),
                "credit_total": sum(float(event.get("credit", 0.0)) for event in library.last_fig_credit_events),
                "events": _jsonable(library.last_fig_credit_events),
            },
            "ig_baseline_value": ig_info["baseline"],
            "propagated_fig_credits": _jsonable(library.last_fig_credit_events),
            "consolidation_decisions": _jsonable(library.last_consolidation_decisions),
            "baseline": ig_info["baseline"],
            "score": ig_info["score"],
            "immediate_ig": ig_info["immediate_ig"],
            "baseline_strategy": ig_info["baseline_strategy"],
        }
        append_jsonl(result_path, record)

        metrics["episodes"] += 1
        metrics["successes"] += 1 if traj.success else 0
        metrics["reward_sum"] += float(traj.final_reward or 0.0)
        metrics["score_estimate_sum"] += score
        metrics["progress_sum"] += float(traj.progress or 0.0)
        metrics.setdefault("episode_ig", []).append(
            {
                "episode": metrics["episodes"],
                "task_id": task.task_id,
                "ig_baseline_value": ig_info["baseline"],
                "baseline": ig_info["baseline"],
                "score": ig_info["score"],
                "immediate_ig": ig_info["immediate_ig"],
                "baseline_strategy": ig_info["baseline_strategy"],
            }
        )
        metrics.update(_summarize(metrics, len(library)))
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "episode": metrics["episodes"],
                    "task_id": task.task_id,
                    "success": traj.success,
                    "reward": traj.final_reward,
                    "score_estimate": score,
                    "library_size": len(library),
                    "ig_baseline_value": ig_info["baseline"],
                    "baseline": ig_info["baseline"],
                    "immediate_ig": ig_info["immediate_ig"],
                    "baseline_strategy": ig_info["baseline_strategy"],
                },
                ensure_ascii=False,
            )
        )

    env.close()
    library.save()
    metrics["finished_at"] = time.time()
    metrics.update(_summarize(metrics, len(library)))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _composition_config(data: Dict[str, Any]) -> CompositionConfig:
    return CompositionConfig(
        strategy=data.get("strategy", "all_context"),
        max_candidates=int(data.get("max_candidates", 8)),
        max_skills_per_candidate=int(data.get("max_skills_per_candidate", 4)),
        max_insights_per_candidate=int(data.get("max_insights_per_candidate", 4)),
        include_singletons=_as_bool(data.get("include_singletons", True)),
        include_mixed=_as_bool(data.get("include_mixed", True)),
        score_policy=data.get("score_policy", "sum_weight"),
        sampling_strategy=data.get("sampling_strategy", "weighted"),
        temperature=float(data.get("temperature", 1.0)),
        top_p=float(data.get("top_p", 0.9)),
        epsilon=float(data.get("epsilon", 0.1)),
        seed=int(data.get("seed", 0)),
        without_replacement=_as_bool(data.get("without_replacement", True)),
    )


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
    parser.add_argument("--limit", type=int, default=None, help="Override number of tasks.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = run(cfg, limit=args.limit)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
