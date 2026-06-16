import json
from pathlib import Path

from evolib_agent_suite.run_eval import run
from evolib_agent_suite.utils import load_config


def test_smoke_run(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    cfg["output_dir"] = str(tmp_path / "runs")
    cfg["library"]["path"] = str(tmp_path / "library.json")
    metrics = run(cfg, limit=2)
    assert metrics["episodes"] == 2
    assert metrics["library_size_current"] > 0

    payload = json.loads(Path(cfg["library"]["path"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["entries"]
    assert "ig_events" in payload
    assert "ig_events" not in payload["stats"]

    record = json.loads((Path(cfg["output_dir"]) / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()[0])
    evolib = record["evolib"]
    assert "retrieval_candidate_count" in evolib
    assert "selected_entry_ids" in evolib
    assert "retrieval_policy_config" in evolib
    assert "composition_policy_config" in evolib
    assert "ig_baseline_value" in evolib
    assert "immediate_ig" in evolib
    assert "propagated_fig_credits" in evolib
    assert "consolidation_decisions" in evolib
