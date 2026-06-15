import random

from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, derive_seed
from evolib_agent_suite.utils import weighted_sample_without_replacement


def test_sampling_policy_reproducible_with_derived_seed():
    cfg = SamplingConfig(strategy="weighted", seed=123)
    policy = SamplingPolicy(cfg)
    context = "task-1"
    first = policy.sample(["a", "b", "c"], [0.1, 0.8, 0.1], 2, policy.rng_for(context))
    second = policy.sample(["a", "b", "c"], [0.1, 0.8, 0.1], 2, policy.rng_for(context))
    assert first == second
    assert derive_seed(123, context) == derive_seed(123, context)


def test_sampling_policy_supports_topk_and_legacy_weighted_sampler():
    cfg = SamplingConfig(strategy="topk", seed=7)
    assert SamplingPolicy(cfg).sample(["low", "high", "mid"], [0.1, 0.9, 0.5], 2) == ["high", "mid"]
    legacy = weighted_sample_without_replacement(["x", "y"], [1.0, 1.0], 2, random.Random(1))
    assert sorted(legacy) == ["x", "y"]
