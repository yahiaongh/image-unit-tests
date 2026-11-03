from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent
from src.clients import get_clients
from src.config import load_config

SCENARIOS = {
    "Apples": "a bowl with exactly 8 green apples, no other fruit",
    "Spatial": "a golden retriever sitting on the left half of the image, and a blue ball on the right half",
    "Absence": "a landscape with exactly 3 trees, and no people anywhere",
}


def _find_run_dir(runs_dir: Path, prompt: str, previous_dirs: set[Path]) -> Path:
    candidates = []
    for path in runs_dir.iterdir():
        if not path.is_dir() or path in previous_dirs:
            continue
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("brief") == prompt:
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No completed run directory found for {prompt!r}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Capture only this scenario; by default all scenarios are captured.",
    )
    args = parser.parse_args()

    config = load_config()
    mock_config = config.setdefault("mock", {})
    mock_config.update({"enabled": False, "spark": False, "image": False, "sam": False})
    spark, image, sam = get_clients(config)

    runs_dir = Path(config.get("runs_dir", "runs"))
    demos_dir = runs_dir / "demos"
    demos_dir.mkdir(parents=True, exist_ok=True)

    scenarios = (
        {args.scenario: SCENARIOS[args.scenario]}
        if args.scenario
        else SCENARIOS
    )
    for name, prompt in scenarios.items():
        target_dir = demos_dir / name

        print(f"=== {name} ===", flush=True)
        print(f"PROMPT: {prompt}", flush=True)
        previous_dirs = {
            path for path in runs_dir.iterdir()
            if path.is_dir() and path != demos_dir
        }
        try:
            result = Agent(spark, image, sam, config).run(prompt)
            rounds = result["rounds"]
            print(f"ROUNDS: {len(rounds)}", flush=True)
            print(f"PASSED: {result['final_passed']}", flush=True)
            for round_index, round_data in enumerate(rounds, start=1):
                print(f"  Round {round_index}:", flush=True)
                for verdict in round_data["verdicts"]:
                    status = "PASS" if verdict["passed"] else "FAIL"
                    print(f"    {status}  {verdict['reason']}", flush=True)
                if round_data.get("fix_instruction"):
                    print(
                        f"    FIX  {round_data['fix_instruction']}",
                        flush=True,
                    )

            run_dir = _find_run_dir(runs_dir, prompt, previous_dirs)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(run_dir, target_dir)
            print(f"COPIED: {run_dir} -> {target_dir}", flush=True)
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
