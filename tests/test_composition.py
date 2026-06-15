from evolib_agent_suite.evolib.composition import CompositionConfig, compose_candidates
from evolib_agent_suite.evolib.library import LibraryEntry


def _entry(entry_id, typ, weight):
    return LibraryEntry(
        id=entry_id, type=typ, title=entry_id, content=f"content {entry_id}", weight=weight
    )


def test_all_context_preserves_retrieval_entries():
    entries = [_entry("s1", "skill", 1.0), _entry("i1", "insight", 2.0)]
    candidates = compose_candidates(entries, CompositionConfig())
    assert len(candidates) == 1
    assert candidates[0].composition_type == "all_context"
    assert candidates[0].entry_ids == ["s1", "i1"]


def test_pairwise_creates_skill_and_mixed_pairs_sorted_by_score():
    entries = [
        _entry("s1", "skill", 1.0),
        _entry("s2", "skill", 2.0),
        _entry("i1", "insight", 3.0),
    ]
    candidates = compose_candidates(entries, CompositionConfig(strategy="pairwise"))
    assert {candidate.composition_type for candidate in candidates} == {
        "skill_skill_pair",
        "skill_insight_pair",
    }
    assert candidates[0].entry_ids == ["s2", "i1"]
