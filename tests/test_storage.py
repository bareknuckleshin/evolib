import json
import sqlite3

from evolib_agent_suite.evolib import EvolvingLibrary, JsonLibraryStorage, LibraryStorage, SQLiteLibraryStorage


def test_json_storage_preserves_legacy_library_shape_and_adds_schema_version(tmp_path):
    path = tmp_path / "library.json"
    legacy_payload = {
        "stats": {"episodes": 1, "score_ema": 0.5, "score_sum": 0.5},
        "entries": [],
    }
    path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    library = EvolvingLibrary(path)
    library.save()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["stats"]["episodes"] == 1
    assert saved["entries"] == []
    assert "lineage_edges" in saved
    assert "merge_events" in saved
    assert "ig_events" in saved
    assert "retrieval_events" in saved
    assert "policy_snapshots" in saved


def test_sqlite_storage_round_trips_canonical_payload(tmp_path):
    path = tmp_path / "library.sqlite"
    storage: LibraryStorage = SQLiteLibraryStorage(path)
    payload = {"schema_version": 2, "stats": {"episodes": 2}, "entries": []}

    storage.save(payload)

    assert storage.load() == payload
    with sqlite3.connect(path) as conn:
        table_count = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'library_state'"
        ).fetchone()[0]
    assert table_count == 1


def test_json_storage_round_trips_payload(tmp_path):
    path = tmp_path / "library.json"
    storage = JsonLibraryStorage(path)
    payload = {"schema_version": 2, "stats": {}, "entries": []}

    storage.save(payload)

    assert storage.load() == payload
