from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install EvoLibAgent plugin into an AgentBoard checkout.")
    parser.add_argument("--agentboard-root", required=True, help="Path to cloned hkust-nlp/AgentBoard")
    args = parser.parse_args()
    root = Path(args.agentboard_root).expanduser().resolve()
    target_dir = root / "agentboard" / "agents"
    if not target_dir.exists():
        raise SystemExit(f"Could not find {target_dir}. Is this an AgentBoard checkout?")
    source = Path(__file__).resolve().parents[1] / "evolib_agent_suite" / "integrations" / "agentboard" / "evolib_agent.py"
    target = target_dir / "evolib_agent.py"
    shutil.copy2(source, target)
    print(f"Copied {source} -> {target}")
    print("Next:")
    print(f"  export EVOLIB_PROJECT_ROOT={Path(__file__).resolve().parents[1]}")
    print("  Edit AgentBoard eval config: agent.name: EvoLibAgent")


if __name__ == "__main__":
    main()
