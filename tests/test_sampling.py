import random

from evolib_agent_suite.evolib.sampling import SamplingConfig, SamplingPolicy, derive_seed


def test_sampling_policy_top_p_limits_to_nucleus():
    policy = SamplingPolicy(SamplingConfig(strategy="top_p", top_p=0.6, seed=7))
    sampled = policy.sample(["a", "b", "c"], [0.7, 0.2, 0.1], 2, random.Random(7))
    assert sampled == ["a"]


def test_derived_seed_is_stable_and_context_sensitive():
    first = derive_seed(123, "episode-1", "skill")
    second = derive_seed(123, "episode-1", "skill")
    other = derive_seed(123, "episode-2", "skill")
    assert first == second
    assert first != other
