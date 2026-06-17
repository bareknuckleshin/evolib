from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


IMPORT_LINE = "from .evolib_agent import EvoLibAgent"


def patch_agents_init(init_path: Path, dry_run: bool = False) -> bool:
    text = init_path.read_text(encoding="utf-8")
    updated = text
    changed = False

    if IMPORT_LINE not in updated:
        imports = re.findall(r"from \.\w+ import [^\n]+", updated)
        if imports:
            last_import = imports[-1]
            updated = updated.replace(last_import, f"{last_import}\n{IMPORT_LINE}", 1)
        else:
            updated = f"{IMPORT_LINE}\n{updated}"
        changed = True

    if "__all__" in updated and '"EvoLibAgent"' not in updated and "'EvoLibAgent'" not in updated:
        all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", updated, flags=re.DOTALL)
        if all_match:
            existing = all_match.group(1).strip()
            replacement_items = f"{existing}, \"EvoLibAgent\"" if existing else '"EvoLibAgent"'
            updated = updated[: all_match.start(1)] + replacement_items + updated[all_match.end(1) :]
            changed = True

    if changed and not dry_run:
        init_path.write_text(updated, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Install EvoLibAgent plugin into an AgentBoard checkout.")
    parser.add_argument("--agentboard-root", required=True, help="Path to cloned hkust-nlp/AgentBoard")
    parser.add_argument("--dry-run", action="store_true", help="Show intended changes without writing files.")
    args = parser.parse_args()

    root = Path(args.agentboard_root).expanduser().resolve()
    target_dir = root / "agentboard" / "agents"
    init_path = target_dir / "__init__.py"
    if not target_dir.exists() or not init_path.exists():
        raise SystemExit(f"Could not find {target_dir} and {init_path}. Is this an AgentBoard checkout?")

    source = Path(__file__).resolve().parents[1] / "evolib_agent_suite" / "integrations" / "agentboard" / "evolib_agent.py"
    target = target_dir / "evolib_agent.py"

    if args.dry_run:
        print(f"Would copy {source} -> {target}")
    else:
        shutil.copy2(source, target)
        print(f"Copied {source} -> {target}")

    changed = patch_agents_init(init_path, dry_run=args.dry_run)
    action = "Would patch" if args.dry_run and changed else "Patched" if changed else "Already configured"
    print(f"{action}: {init_path}")

    print("Next:")
    print(f"  export EVOLIB_PROJECT_ROOT={Path(__file__).resolve().parents[1]}")
    print("  Merge configs/agentboard_evolib_agent_snippet.yaml into AgentBoard/eval_configs/main_results_all_tasks.yaml")
    print("  cd /path/to/AgentBoard/agentboard/environment/WebShop && bash ./run_dev.sh")
    print("  cd /path/to/AgentBoard && python agentboard/eval_main.py --cfg-path eval_configs/main_results_all_tasks.yaml --tasks webshop --model gpt-3.5-turbo-0613 --log_path ./results/evolib_agentboard")


if __name__ == "__main__":
    main()
