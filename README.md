# EvoLib Agent Suite

This is a practical scaffold for applying **Test-Time Learning with an Evolving Library (EvoLib)** to three evaluation routes:

1. **AgentBoard** via a plugin agent class (`EvoLibAgent`).
2. **Original WebShop** via the `WebAgentTextEnv-v0` Gym text environment.
3. **Original ALFWorld** via the official ALFWorld TextWorld environment.

The code is intentionally modular:

```text
evolib_agent_suite/
  agents/                 ReAct-style action agent with retrieved library entries
  evolib/                 persistent library, retrieval, extraction, self-judging
  envs/                   adapters for mock, original WebShop, original ALFWorld
  integrations/agentboard AgentBoard plugin agent
  llm/                    heuristic, OpenAI-compatible HTTP, and LiteLLM providers
  run_eval.py             standalone evaluator for original WebShop/ALFWorld/mock
configs/                  example configs
scripts/                  helper scripts
```

## 1. Install

```bash
cd evolib_agent_suite
python -m pip install -e .
```

For local smoke test only, no external benchmark is needed:

```bash
python -m evolib_agent_suite.run_eval --config configs/smoke.yaml --limit 2
```

## 2. EvoLib loop implemented here

For each task episode:

```text
retrieve relevant skills/insights from library
→ generate actions with a ReAct-style agent
→ record action-observation trajectory
→ estimate score with self-judge or env reward variant
→ extract new skill/insight abstractions
→ merge/consolidate similar entries
→ update weights via immediate IG + parent Future-IG approximation
→ persist library
```

The default setting follows the no-external-feedback spirit of EvoLib: library updates use LLM self-judging. For reward-aware ablations, set:

```yaml
eval:
  library_update_uses_env_reward: true
```

## 3. Original WebShop

External setup:

```bash
git clone https://github.com/princeton-nlp/WebShop webshop
cd webshop
conda create -n webshop python=3.8.13
conda activate webshop
./setup.sh -d small
export PYTHONPATH=/path/to/webshop:$PYTHONPATH
```

Run:

```bash
export OPENAI_API_KEY=...
python -m evolib_agent_suite.run_eval --config configs/original_webshop.yaml --limit 20
```

The adapter uses:

```python
import gym
from web_agent_site.envs import WebAgentTextEnv
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
```

Actions should be `search[query]` or `click[value]`.

## 4. Original ALFWorld

External setup:

```bash
conda create -n alfworld python=3.9
conda activate alfworld
pip install alfworld[full]
alfworld-download
```

Run:

```bash
export OPENAI_API_KEY=...
python -m evolib_agent_suite.run_eval --config configs/original_alfworld.yaml --limit 20
```

The adapter follows the official `get_environment(...).init_env(batch_size=1)` flow and uses `info['admissible_commands']` when available.

## 5. AgentBoard

External setup:

```bash
git clone https://github.com/hkust-nlp/AgentBoard.git AgentBoard
cd AgentBoard
mkdir data
wget https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz
tar -zxvf data.tar.gz
INSTALL_WEBARENA=false bash ./setup.sh
```

Install the plugin into the AgentBoard checkout:

```bash
cd /path/to/evolib_agent_suite
python scripts/install_into_agentboard.py --agentboard-root /path/to/AgentBoard
export EVOLIB_PROJECT_ROOT=/path/to/evolib_agent_suite
```

Merge `configs/agentboard_evolib_agent_snippet.yaml` into AgentBoard's eval config, then run AgentBoard normally:

```bash
cd /path/to/AgentBoard
python agentboard/eval_main.py \
  --cfg-path eval_configs/main_results_all_tasks.yaml \
  --tasks alfworld,webshop \
  --model gpt-3.5-turbo-0613 \
  --log_path ./results/evolib_agentboard
```

The AgentBoard plugin finalizes the previous episode at the next `reset()` call, because AgentBoard's public custom-agent interface exposes `reset`, `run`, and `update`, but not a universal terminal callback.

## 6. Outputs

Each standalone run writes:

```text
runs/<name>/
  library.json          persistent EvoLib library
  trajectories.jsonl    episode-level traces and extracted entries
  metrics.json          success/rate/reward/progress summary
```

## 7. Notes for fair reporting

Recommended experiment labels:

```text
EvoLib-self-score: library update uses LLM self-judge only; env reward only for final metrics.
EvoLib-env-feedback: library update uses benchmark reward; stronger but not no-external-feedback.
EvoLib-reset: ablation where library is cleared per episode.
```

For WebShop, be careful that a library can overfit to catalog-specific phrases. Prefer reporting whether extracted entries are general workflows such as query reformulation, constraint checking, option verification, and recovery from invalid clicks.

For ALFWorld, reusable workflows are usually more explicit: locate → pick up → transform if needed → navigate → place/examine.
