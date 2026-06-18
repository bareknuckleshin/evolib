# Original WebShop 테스트 실행 시 EvoLib 동작 정리

이 문서는 `python -m evolib_agent_suite.run_eval --config configs/original_webshop.yaml`로 Original WebShop 테스트를 실행할 때, `evolib_agent_suite` 안의 주요 모듈이 어떤 순서로 작동하고 어떤 설정이 실제로 적용되는지 정리한다.

## 1. 실행 진입점과 전체 루프

- 실행 진입점은 `evolib_agent_suite.run_eval.run()`이다.
- 설정 파일 `configs/original_webshop.yaml`을 읽어 다음 객체를 만든다.
  - LLM: `build_llm(config["llm"])`
  - 환경: `build_env(config["env"])`
  - 임베딩 함수: `build_embedding_function(config["embedding"])`
  - EvoLib 저장소/라이브러리: `EvolvingLibrary(...)`
  - trajectory 평가/추출기: `AbstractionExtractor(llm)`
  - 액션 에이전트: `EvoLibReActAgent(...)`
- 에피소드마다 다음 순서로 진행된다.
  1. WebShop task를 얻고 환경을 reset한다.
  2. 현재 task domain, goal, 최초 observation을 query로 EvoLib library에서 관련 skill/insight를 retrieval한다.
  3. retrieval된 entry들을 candidate solution으로 조합한다.
  4. candidate solution을 prompt에 넣어 ReAct 방식으로 WebShop action을 생성한다.
  5. 에피소드 종료 후 trajectory를 self-judge하거나 환경 reward로 score를 만든다.
  6. trajectory에서 새 reusable skill/insight 후보를 추출한다.
  7. 후보를 기존 library entry와 consolidation/create한다.
  8. information gain 기반으로 새 entry와 사용된 parent entry의 weight를 갱신하고 library를 저장한다.

## 2. Original WebShop 환경 어댑터

Original WebShop 테스트에서는 `env.backend: original_webshop` 설정에 의해 `OriginalWebShopAdapter`가 생성된다.

- `gym.make("WebAgentTextEnv-v0", ...)`로 Princeton WebShop의 text 환경을 감싼다.
- 기본 설정은 다음과 같다.
  - `num_products: 1000`
  - `observation_mode: text`
  - `human_goals: 0`
  - `show_attrs: false`
  - `max_steps: 15`
- `iter_tasks()`는 WebShop raw env의 `server.goals`에서 goal instruction을 읽고, 각 session index를 `TaskSpec.task_id`로 사용한다.
- `TaskSpec.domain`은 `webshop_original`로 고정된다.
- action hint는 WebShop 형식에 맞게 `search[query]`와 `click[value]`를 쓰도록 지정된다.
- `reset()`은 `env.reset(session=session_int)`를 호출하고, 가능한 경우 `raw_env.get_instruction_text()`로 goal text를 다시 읽는다.
- `step()`은 WebShop action을 그대로 `env.step(action)`에 전달하며, reward를 float으로 변환하고 `done and reward >= 1.0`이면 success로 기록한다. `max_steps`에 도달하면 강제로 done 처리한다.

## 3. Retrieval: 어떤 skill/insight를 가져오는가

### 3.1 적용되는 설정

`configs/original_webshop.yaml` 기준 retrieval 관련 기본값은 다음과 같다.

| 항목 | 값 | 의미 |
| --- | --- | --- |
| `k_skills` | `4` | 최대 4개의 skill entry 선택 |
| `k_insights` | `4` | 최대 4개의 insight entry 선택 |
| `retrieval_similarity_threshold` | `0.05` | query-entry cosine similarity가 이 값 이상인 entry만 후보 |
| `sample` | `true` | top-k 고정 선택이 아니라 sampling 기반 선택 사용 |
| `candidate_pool_multiplier` | 기본 `4` | type별로 `k * 4`개까지 후보 pool을 만든 뒤 sampling |
| `sampling_strategy` | 기본 `weighted` | retrieval weight 기반 weighted sampling |
| `temperature` | 기본 `1.0` | softmax/top-p 계열 sampling에서 사용 |
| `epsilon` | 기본 `0.1` | epsilon-greedy sampling에서 사용 |
| `top_p` | 기본 `0.9` | top-p sampling에서 사용 |
| `weight_alpha` | 기본 `1.0` | composite score에서 entry weight 영향 |
| `similarity_alpha` | 기본 `1.0` | composite score에서 similarity 영향 |
| `without_replacement` | 기본 `true` | 같은 type 안에서 중복 없이 sampling |

`run_eval.py`는 매 에피소드마다 `retrieval_config.context_id = task.task_id`로 설정한다. 따라서 weighted sampling을 쓰더라도 같은 seed와 같은 task id에서는 재현 가능한 derived seed가 만들어진다.

### 3.2 query 구성

Retrieval query는 다음 문자열이다.

```text
{task.domain}\n{task.goal}\n{reset.observation}
```

Original WebShop에서는 domain이 `webshop_original`이므로, goal instruction과 첫 observation뿐 아니라 domain tag도 similarity 계산에 들어간다.

### 3.3 scoring과 candidate pool

`EvolvingLibrary.retrieve_with_metadata()`는 library entry마다 다음을 계산한다.

1. query embedding과 entry embedding의 cosine similarity.
2. similarity가 threshold 이상인 entry만 유지.
3. `retrieval_weight = max(1e-6, (0.2 + similarity) * max(entry.weight, 1e-6))` 계산.
4. 정렬용 composite score 계산:

```text
composite = similarity^similarity_alpha * entry.weight^weight_alpha
```

그 다음 entry type별로 분리한다.

- skill group에서 최대 `k_skills`개 선택.
- insight group에서 최대 `k_insights`개 선택.
- 각 group은 composite score, similarity, retrieval weight 순으로 내림차순 정렬한다.
- 정렬된 group에서 상위 `k * candidate_pool_multiplier`개만 sampling pool로 남긴다.

### 3.4 실제 선택 방법

`sampling_strategy`가 `weighted`인 경우, 최종 선택은 composite score가 아니라 `retrieval_weight`를 weight로 사용한다. 즉, 현재 설정에서는 similarity와 entry weight가 모두 큰 entry가 더 뽑히기 쉽다.

지원되는 sampling strategy는 다음과 같다.

- `topk`: pool의 앞에서 k개 선택.
- `uniform`: 균등 random sampling.
- `weighted`: 전달된 weight 기반 sampling.
- `softmax`: score를 temperature softmax로 변환한 뒤 sampling.
- `top_p`: 누적 확률 `top_p` 안의 nucleus에서 sampling.
- `epsilon_greedy`: 확률 `epsilon`으로 탐험, 그 외 최고 score 선택.

Retrieval 결과는 `RetrievedEntry`로 기록되며, trajectory JSONL의 `evolib.retrieved_entries`에 id, similarity, retrieval_weight, rank, selected_by, sampling seed/context가 저장된다.

## 4. Candidate solution 조합 방법

### 4.1 기본 적용 전략

Original WebShop config에는 `agent.composition` 섹션이 없으므로 `CompositionConfig` 기본값이 적용된다.

| 항목 | 기본값 | 의미 |
| --- | --- | --- |
| `strategy` | `all_context` | retrieval된 모든 entry를 하나의 candidate로 묶음 |
| `max_candidates` | `8` | 여러 candidate 생성 전략에서 최대 후보 수 |
| `max_skills_per_candidate` | `4` | bundle 전략에서 skill 최대 수 |
| `max_insights_per_candidate` | `4` | bundle 전략에서 insight 최대 수 |
| `include_singletons` | `true` | singleton 후보 허용 |
| `include_mixed` | `true` | skill+insight 혼합 허용 |
| `score_policy` | `sum_weight` | candidate score는 entry weight 합 |
| `sampling_strategy` | `weighted` | sampled bundle 전략에서 사용 |

현재 기본 전략 `all_context`에서는 retrieval된 entry 전체가 그대로 하나의 candidate solution이 된다. 즉, WebShop 테스트의 기본 경로에서는 별도 pairwise 탐색이나 bundle sampling 없이, retrieval 결과 전체가 prompt의 reusable EvoLib entries 블록으로 들어간다.

### 4.2 다른 조합 전략을 설정했을 때

`agent.composition.strategy`를 설정하면 다음 전략도 가능하다.

- `singletons`: entry 하나당 candidate 하나.
- `pairwise`: skill-skill pair와, `include_mixed=true`일 때 skill-insight pair 생성.
- `mixed_bundle`: 상위 skill N개와 insight N개를 묶음.
- `weighted_sampled_bundle`: skill/insight를 weight 기반 sampling으로 묶음.

`select_candidate()`는 `compose_candidates()` 결과 중 첫 번째 candidate를 사용한다. `all_context`에서는 항상 `composition_type=all_context`이며, `candidate.entry_ids`가 이번 trajectory의 `used_entry_ids`가 된다.

## 5. ReAct agent prompt와 action 생성

`EvoLibReActAgent.reset()`은 retrieval entry들을 candidate solution으로 조합하고 history를 초기화한다. 이후 매 step마다 `act()`가 실행된다.

Prompt에는 다음 정보가 들어간다.

- WebShop goal.
- candidate solution에 포함된 EvoLib entry 목록.
  - 각 entry는 id, type, weight, title, content로 format된다.
  - entry block은 최대 5000자까지 포함된다.
- action hint.
  - Original WebShop task의 hint가 있으면 config의 `agent.action_hint`보다 우선된다.
- 최근 trajectory history.
  - `memory_size: 12`이므로 최근 12 step만 포함된다.
- 최신 observation.
- WebShop에서 제공하는 available actions.
  - search bar가 있으면 `search[<query>]`를 표시한다.
  - clickable이 있으면 `click[...]` 목록을 표시한다.

LLM 응답은 다음 형식을 기대한다.

```text
Thought: <one sentence>
Action: <exact action string>
```

파서는 `Action:` 라인을 우선 사용하고, 없으면 마지막 non-empty line을 action fallback으로 쓴다. 생성된 action은 WebShop env에 그대로 전달된다.

## 6. Score 추정과 reusable abstraction 추출

에피소드 종료 후 `AbstractionExtractor`가 trajectory를 평가하고 새 library 후보를 만든다.

### 6.1 score 추정

`configs/original_webshop.yaml`의 `eval.library_update_uses_env_reward`는 기본 `false`다. 따라서 기본 실행은 WebShop reward를 library update score로 바로 쓰지 않고, LLM self-judge를 사용한다.

- self-judge prompt는 visible trajectory와 subgoal만 보고 `score`, `progress`, `notes` JSON을 요구한다.
- JSON parsing에 실패하거나 LLM 호출이 실패하면 fallback score를 쓴다.
- `library_update_uses_env_reward: true`로 바꾸면 `final_reward`를 clamp해서 score로 사용한다.

### 6.2 abstraction 추출

`extract()`는 trajectory transcript와 estimated score를 LLM에 전달해 2-6개의 reusable abstraction을 JSON list로 요청한다.

- type은 `skill` 또는 `insight`만 허용된다.
- exact product ID나 episode-specific trivia는 피하고, 일반화 가능한 workflow/check를 선호한다.
- WebShop domain tag가 tags에 없으면 자동으로 추가된다.
- LLM extraction이 실패하면 heuristic extraction을 쓴다.
  - WebShop에서 search action이 있었다면 `Search with core constraints first` skill을 생성한다.
  - 항상 `Verify options before buying` insight를 생성한다.

## 7. Consolidation: 새 후보를 library에 어떻게 반영하는가

### 7.1 기본 적용 설정

Original WebShop config에는 별도 `consolidation` 섹션이 없으므로 기본값이 적용된다. 단, `similarity_threshold`는 `library.similarity_merge_threshold` 값인 `0.88`을 이어받는다.

| 항목 | 적용값 | 의미 |
| --- | --- | --- |
| `enabled` | `true` | consolidation 활성화 |
| `similarity_threshold` | `0.88` | candidate와 기존 entry similarity가 이 값 이상이면 merge 후보 |
| `candidate_top_n` | `1` | 가장 유사한 후보 1개만 target으로 사용 |
| `merge_strategy` | `replace_if_longer` | candidate content가 더 길고 제한보다 짧으면 기존 content 대체 |
| `score_policy` | `ema_score` | merge 후 score_ema를 EMA로 갱신 |
| `allow_cross_type_merge` | `false` | skill은 skill끼리, insight는 insight끼리만 merge |
| `merge_history_limit` | `20` | entry metadata에 보관할 merge history 길이 |
| `ema_decay` | `0.85` | score/IG EMA decay |

### 7.2 create vs merge

`library.add_or_merge_many()`는 추출된 후보마다 `LibraryEntry`를 만든 뒤 기존 entry와 비교한다.

- 같은 type이고 embedding cosine similarity가 `0.88` 이상인 기존 entry가 있으면 merge한다.
- merge target이 없으면 새 entry로 create한다.
- merge/create 모두 parent ids로 이번 candidate solution에 포함된 `used_entry_ids`를 연결한다.
- parent-child lineage edge를 남긴다.
- merge decision은 `merge_events`와 `last_consolidation_decisions`에 기록되고, trajectory JSONL에도 `evolib.consolidation_decisions`로 저장된다.

`replace_if_longer` 전략에서는 더 일반적이거나 자세한 candidate를 유지하려는 의도로, candidate content가 기존 content보다 길고 `max_replace_content_chars`보다 짧을 때 기존 content/title을 교체한다. tags, parents, source_task_ids는 합쳐진다.

## 8. Information Gain, Future IG, weight 갱신

Consolidation 이후 `library.update_after_episode()`가 호출된다.

### 8.1 baseline과 immediate IG

기본 `IGConfig`는 `baseline_strategy: global_ema`, `ema_decay: 0.85`다. 따라서 immediate IG는 다음처럼 계산된다.

```text
baseline = library.stats["score_ema"]
immediate_ig = current_score - baseline
```

새로 생성/업데이트된 entry는 다음 값을 갱신한다.

- `ig_ema`
- `score_ema`
- success이면 `wins`
- 최종 `weight`

### 8.2 retrieved parent에 대한 Future IG credit

이번 episode에서 사용된 retrieved entry들, 즉 candidate solution의 `used_entry_ids`는 positive immediate IG가 있을 때 credit을 받는다.

- depth 1 parent: `positive_delta * 1.0`
- depth 2 ancestor: `positive_delta * 0.5`

각 parent entry는 다음을 갱신한다.

- `future_ig_ema`
- `score_ema`
- success이면 `wins`
- 최종 `weight`

Weight 재계산식은 다음과 같다.

```text
usage_bonus = min(0.5, 0.03 * uses)
win_bonus = min(0.5, 0.05 * wins)
value = 1.0 + alpha_ig * ig_ema + beta_future_ig * future_ig_ema
weight = max(0.05, value + usage_bonus + win_bonus)
```

기본 `alpha_ig=1.0`, `beta_future_ig=0.7`이므로, 직접 추출된 abstraction의 immediate IG와 이전 abstraction이 만든 future usefulness가 retrieval 확률에 다시 영향을 준다.

## 9. 저장되는 산출물과 로그

기본 output directory는 `./runs/original_webshop_evolib`이다.

- `trajectories.jsonl`
  - 각 episode trajectory.
  - `evolib` 필드에 candidate/retrieval/consolidation/IG 상세 metadata 저장.
- `metrics.json`
  - episode 수, success rate, reward 평균, score estimate 평균, progress 평균, library size 등 저장.
- `library.json`
  - persistent EvoLib library.
  - entries, lineage_edges, merge_events, ig_events/fig_events, retrieval_events, policy_snapshots 저장.

`policy_snapshots`에는 실행 시작 시점의 retrieval/composition/consolidation/IG/storage 정책이 저장되므로, 나중에 어떤 정책으로 WebShop 테스트가 실행됐는지 재현할 수 있다.

## 10. 기본 Original WebShop 실행에서 핵심 요약

- Retrieval은 `webshop_original + goal + initial observation` query로 embedding cosine similarity를 계산하고, skill 4개/insight 4개를 type별로 weighted sampling한다.
- Weighted retrieval의 weight는 `(0.2 + similarity) * entry.weight`이며, 후보 pool 정렬에는 `similarity * entry.weight` composite score가 쓰인다.
- Candidate solution 조합은 기본적으로 `all_context`라서 retrieval된 entry 전체가 하나의 prompt context가 된다.
- Agent는 WebShop의 `search[query]`, `click[value]` action 규칙과 available actions를 prompt에 넣고 LLM으로 다음 action을 생성한다.
- Library update score는 기본적으로 WebShop reward가 아니라 LLM self-judge score다. 필요하면 `library_update_uses_env_reward: true`로 바꿀 수 있다.
- 새 skill/insight는 LLM extraction으로 만들고, 실패 시 WebShop용 heuristic skill/insight를 만든다.
- Consolidation은 기본적으로 같은 type끼리 cosine similarity `0.88` 이상이면 `replace_if_longer` 전략으로 merge하고, 아니면 create한다.
- Immediate IG와 2-hop Future IG가 entry weight를 바꾸며, 이 weight가 다음 episode retrieval 확률에 다시 반영된다.
