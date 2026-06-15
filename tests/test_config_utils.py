from evolib_agent_suite.config_utils import normalize_library_config


def test_flat_library_keys_map_to_nested_policy():
    policy = normalize_library_config(
        {
            "path": "legacy-library.json",
            "k_skills": 7,
            "k_insights": 2,
            "sample": False,
            "retrieval_similarity_threshold": 0.2,
            "similarity_merge_threshold": 0.91,
        },
        default_path="default-library.json",
    )

    assert policy["storage"]["path"] == "legacy-library.json"
    assert policy["retrieval"]["k_skills"] == 7
    assert policy["retrieval"]["k_insights"] == 2
    assert policy["retrieval"]["similarity_threshold"] == 0.2
    assert policy["retrieval"]["sampling_strategy"] == "topk"
    assert policy["retrieval"]["sample"] is False
    assert policy["consolidation"]["similarity_merge_threshold"] == 0.91


def test_flat_overrides_still_work_with_nested_base_config():
    policy = normalize_library_config(
        {
            "path": "override-library.json",
            "k_skills": 9,
            "sample": False,
            "storage": {"path": "nested-library.json"},
            "retrieval": {"k_skills": 4, "sampling_strategy": "weighted"},
        },
        default_path="default-library.json",
    )

    assert policy["storage"]["path"] == "override-library.json"
    assert policy["retrieval"]["k_skills"] == 9
    assert policy["retrieval"]["sampling_strategy"] == "topk"
    assert policy["retrieval"]["sample"] is False
