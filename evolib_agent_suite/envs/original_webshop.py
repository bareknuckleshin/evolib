from __future__ import annotations

from typing import Any, Iterable, List, Optional

from evolib_agent_suite.envs.base import EnvironmentAdapter
from evolib_agent_suite.schema import ResetResult, StepOutput, TaskSpec


class OriginalWebShopAdapter(EnvironmentAdapter):
    """princeton-nlp/WebShop 원본 텍스트 환경을 EvoLib 공통 인터페이스로 감싸는 어댑터.

    이 클래스는 WebShop의 Gym 환경(`WebAgentTextEnv-v0`)을 직접 생성하고,
    EvoLib 평가 루프가 기대하는 `iter_tasks`, `reset`, `available_actions`, `step`,
    `close` 메서드로 변환한다. 즉, Original WebShop 테스트에서 환경 의존적인
    부분을 이 파일 하나에 모아 두는 역할을 한다.

    Expected external setup:
        git clone https://github.com/princeton-nlp/WebShop webshop
        cd webshop && ./setup.sh -d small
        export PYTHONPATH=/path/to/webshop:$PYTHONPATH
    """

    # EvoLib 라이브러리/trajectory에 기록되는 도메인 이름이다. 다른 환경과 섞여도
    # 검색 및 분석 시 Original WebShop 결과를 구분할 수 있게 한다.
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
        """Original WebShop Gym 환경을 생성하고 평가에 필요한 상태를 초기화한다.

        Args:
            num_products: WebShop에서 로드할 상품 수. 작은 데이터셋 smoke/eval은
                보통 1,000개로 시작한다.
            observation_mode: WebShop 관측 형식. 이 프로젝트는 LLM 에이전트가 읽는
                텍스트 관측을 사용하므로 기본값은 `text`이다.
            human_goals: WebShop 원본 환경의 goal 선택 옵션.
            show_attrs: 상품 속성을 관측에 노출할지 여부.
            max_steps: 한 세션에서 허용하는 최대 액션 수. 원본 환경이 종료하지 않아도
                이 값에 도달하면 어댑터가 강제로 episode를 종료한다.
            sessions: 특정 WebShop session id만 재현 테스트하고 싶을 때 사용하는 목록.
            **kwargs: WebShop Gym 환경으로 그대로 전달되는 추가 옵션.
        """
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
        # 현재 episode에서 수행한 step 수. WebShop 종료 신호와 별도로 max_steps를
        # 적용하기 위한 카운터다.
        self.t = 0

    @property
    def raw_env(self):
        """Gym wrapper 안쪽의 실제 WebShop 환경 객체를 반환한다."""
        return getattr(self.env, "unwrapped", self.env)

    def iter_tasks(self, limit: Optional[int] = None, split: str = "test") -> Iterable[TaskSpec]:
        """WebShop session/goals를 EvoLib `TaskSpec` 스트림으로 변환한다.

        중요한 데이터 흐름:
        - WebShop 원본 서버의 `server.goals`에는 session별 instruction과 메타데이터가
          들어 있다.
        - `sessions`가 지정되면 해당 session만 고정 순서로 평가하여 재현성을 높인다.
        - 지정하지 않으면 `limit` 또는 최대 500개의 앞쪽 session을 평가 대상으로 삼는다.
        """
        raw = self.raw_env
        goals = getattr(getattr(raw, "server", None), "goals", []) or []
        if self.sessions is not None:
            indices = self.sessions
        else:
            n = limit or min(len(goals), 500)
            indices = list(range(n))
        for idx in indices[: limit or len(indices)]:
            goal_text = ""
            # metadata의 session_int는 reset(session=...)에 다시 사용되는 핵심 값이다.
            # goal 원본도 저장해 두면 실패 episode를 사후 분석할 때 유용하다.
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
        """지정된 WebShop session으로 환경을 초기화하고 첫 관측과 goal을 반환한다."""
        self.current_task = task
        self.t = 0
        session_int = task.metadata.get("session_int")
        obs, info = self.env.reset(session=session_int)
        goal = task.goal
        try:
            # 원본 WebShop은 reset 이후 실제 instruction text를 제공한다. TaskSpec의
            # fallback goal보다 원본 instruction을 우선 사용해 평가 프롬프트와 로그를 맞춘다.
            goal = self.raw_env.get_instruction_text()
        except Exception:
            pass
        return ResetResult(observation=str(obs), goal=goal, info=info or {})

    def available_actions(self):
        """현재 화면에서 WebShop이 허용하는 액션 목록을 반환한다.

        일부 WebShop 버전/상태에서는 사용 가능 액션 API가 실패할 수 있으므로, 그 경우
        None을 반환해 에이전트가 관측 텍스트와 action hint만으로 액션을 고르게 한다.
        """
        try:
            return self.raw_env.get_available_actions()
        except Exception:
            return None

    def step(self, action: str) -> StepOutput:
        """에이전트 액션을 WebShop에 적용하고 EvoLib 표준 step 결과로 변환한다.

        알고리즘 요약:
        1. 원본 WebShop `env.step(action)`을 호출한다.
        2. reward를 float로 정규화하고 `max_steps` 초과 시 강제 종료한다.
        3. 가능한 액션 목록을 info에 병합해 trajectory에 기록한다.
        4. WebShop은 최종 구매 성공 시 보통 reward 1.0을 주므로, 종료 상태이면서
           reward가 1.0 이상이면 success로 표시한다.
        """
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
        """WebShop/Gym 환경 리소스를 정리한다."""
        try:
            self.env.close()
        except Exception:
            pass
