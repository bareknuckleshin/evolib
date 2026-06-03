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
