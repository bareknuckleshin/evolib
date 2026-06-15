# `evolib_agent_suite/evolib/library.py` merge conflict resolution guide

When merging the sampling-policy branch with a newer `main`, prefer keeping the
newer `main` library features and only graft the shared sampling policy into the
places that still perform ad-hoc sampling. In practice, resolve the conflict as
an integration of both sides rather than choosing either side wholesale.

## 1. Imports

Keep `main`'s new modules (`math`, consolidation, IG) and add the sampling policy
imports. Drop the old direct `weighted_sample_without_replacement` import from
`library.py`; `SamplingPolicy` reuses that helper internally.

```python
import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from evolib_agent_suite.evolib.consolidation import ConsolidationConfig, ConsolidationPolicy, LLMMerger
from evolib_agent_suite.evolib.ig import BaselineEstimator, IGConfig
from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, derive_seed
from evolib_agent_suite.utils import clamp, cosine, hashed_embedding
```

## 2. `__init__`

Keep `main`'s IG/consolidation initialization and add `SamplingPolicy` alongside
it. `self.rng` can remain if other main-branch code still needs it, but retrieval
sampling should use `self.sampling_policy`.

```python
self.seed = int(seed)
if sampling_config is None:
    sampling_config = SamplingConfig(seed=self.seed)
elif sampling_config.seed == 0 and self.seed != 0:
    sampling_config.seed = self.seed
self.sampling_policy = SamplingPolicy(sampling_config)

if isinstance(ig_config, IGConfig):
    self.ig_config = ig_config
else:
    ig_data = dict(ig_config or {})
    ig_data.setdefault("ema_decay", ema_decay)
    self.ig_config = IGConfig(**ig_data)
self.rng = random.Random(self.seed)
if consolidation_config is None:
    consolidation_config = ConsolidationConfig(
        similarity_threshold=similarity_merge_threshold,
        ema_decay=ema_decay,
    )
self.consolidation_config = consolidation_config
merger = LLMMerger(llm) if llm is not None and consolidation_config.merge_strategy == "llm_merge" else None
self.consolidation_policy = ConsolidationPolicy(consolidation_config, llm_merger=merger)
```

## 3. Retrieval

Keep `main`'s `RetrievalConfig`, `RetrievedEntry`, `retrieve_with_metadata()`, and
composite scoring API. Add optional `task_id` and `episode_id` to `retrieve()` and
`retrieve_with_metadata()` so derived seeds can include episode/task context.
Then replace `_select_retrieval_group()`'s ad-hoc strategy implementation with
`SamplingPolicy`.

The important shape is:

```python
def retrieve(..., task_id: Optional[str] = None, episode_id: Optional[Any] = None, ...):
    config = RetrievalConfig(...)
    return [item.entry for item in self.retrieve_with_metadata(query, config=config, task_id=task_id, episode_id=episode_id)]


def retrieve_with_metadata(self, query: str, config: Optional[RetrievalConfig] = None, task_id: Optional[str] = None, episode_id: Optional[Any] = None) -> List[RetrievedEntry]:
    ...
    sample_meta: Dict[str, Any] = {}
    for typ, k in [("skill", config.k_skills), ("insight", config.k_insights)]:
        ...
        context = f"retrieval:{task_id or 'unknown-task'}:{episode_id or 'unknown-episode'}:{typ}"
        chosen = self._select_retrieval_group(group, k, config, context=context)
        sample_meta[typ] = self.sampling_policy.metadata(
            derived_seed=derive_seed(self.sampling_policy.config.seed, context),
            context=context,
        )
        ...
    self.stats["last_sampling"] = {**dict(self.stats.get("last_sampling", {})), "retrieval": sample_meta}
    return selected
```

```python
def _select_retrieval_group(..., context: str) -> List[Tuple[LibraryEntry, float, float, float, str]]:
    strategy = config.sampling_strategy.strip().lower()
    policy_config = SamplingConfig(
        strategy=strategy,
        temperature=config.temperature,
        epsilon=config.epsilon,
        seed=self.sampling_policy.config.seed,
        without_replacement=True,
    )
    policy = SamplingPolicy(policy_config)
    scores = [g[3] if strategy in {"topk", "softmax", "epsilon_greedy"} else g[2] for g in group]
    sampled = policy.sample(group, scores, k, policy.rng_for(context))
    return [(*item, strategy) for item in sampled]
```

If `RetrievalConfig` does not already have `top_p`, add it and pass it into
`SamplingConfig(top_p=config.top_p)`.

## 4. `add_or_merge_many()` conflict typo

The provided conflict snippet has a syntax issue: `_record_lineage_edges(` is not
closed before `self.consolidation_policy.merge(` starts. Keep both operations, but
close the lineage call first:

```python
self._record_lineage_edges(
    parents,
    entry.id,
    task_id,
    score,
    edge_type="merge",
    metadata={"merge_similarity": sim, "candidate_id": candidate.id},
)
self.consolidation_policy.merge(
    entry,
    candidate,
    similarity=sim,
    task_id=task_id,
    score=score,
    parents=parents,
    task_context=task_context,
)
```

## 5. `update_after_episode()`

Keep `main`'s `BaselineEstimator` return contract (`-> Dict[str, Any]`) and add
sampling metadata to the `context` dict rather than reverting to the old
`prev_baseline = stats["score_ema"]` logic. This avoids breaking FIG/IG behavior.

```python
def update_after_episode(..., context: Optional[Dict[str, Any]] = None, task_id: Optional[str] = None, episode_id: Optional[Any] = None) -> Dict[str, Any]:
    context = dict(context or {})
    context.setdefault("retrieved_ids", list(retrieved_ids))
    context.setdefault("retrieved_count", len(retrieved_ids))

    baseline_context = (
        f"baseline_bootstrap:{task_id or context.get('task_id', 'unknown-task')}:"
        f"{episode_id or context.get('episode_id', 'unknown-episode')}:{','.join(retrieved_ids)}:{len(new_ids)}"
    )
    baseline_seed = derive_seed(self.sampling_policy.config.seed, baseline_context)
    baseline_entries = self.sampling_policy.sample(
        list(self.entries.values()),
        [entry.weight for entry in self.entries.values()],
        min(8, len(self.entries)),
        self.sampling_policy.rng_for(baseline_context),
    )
    sampling_meta = self.sampling_policy.metadata(derived_seed=baseline_seed, context=baseline_context)
    context["baseline_bootstrap_entry_ids"] = [entry.id for entry in baseline_entries]
    context["baseline_bootstrap_sampling"] = sampling_meta
    self.stats["last_sampling"] = {
        **dict(self.stats.get("last_sampling", {})),
        "baseline_bootstrap": sampling_meta,
        "baseline_entry_ids": context["baseline_bootstrap_entry_ids"],
    }

    ig_info = self.baseline_estimator.compute_immediate_ig(score, context)
    score = float(ig_info["score"])
    immediate_ig = float(ig_info["immediate_ig"])
    prev_baseline = float(ig_info.get("baseline", self.stats.get("score_ema", 0.0)))
    ...
    self.baseline_estimator.update(score, context)
    return ig_info
```

## 6. Sanity checks after editing

After removing all conflict markers, run:

```bash
rg -n "<<<<<<<|=======|>>>>>>>" evolib_agent_suite/evolib/library.py
python -m py_compile evolib_agent_suite/evolib/library.py
pytest -q
```

If `py_compile` reports an import error for `consolidation` or `ig`, verify that
you are resolving against the newer `main` branch that introduced those modules.
