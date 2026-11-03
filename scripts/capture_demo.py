from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent
from src.clients import get_clients
from src.config import load_config

HARD_PROMPTS = [
    "a bowl with exactly 8 green apples, no other fruit",
]


def _newest_run_dir(runs_dir: Path, existing: set[Path]) -> Path | None:
    new_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and path not in existing and (path / "summary.json").is_file()
    ]
    return max(new_dirs, key=lambda path: path.stat().st_mtime, default=None)


def main() -> None:
    config = load_config()
    spark, image, sam = get_clients(config)
    runs_dir = Path(config.get("runs_dir", "runs"))
    demo_dir = runs_dir / "demo"

    for prompt in HARD_PROMPTS:
        print(f"PROMPT: {prompt}", flush=True)
        existing_dirs = {path for path in runs_dir.iterdir() if path.is_dir()}
        try:
            result = Agent(spark, image, sam, config).run(prompt)
        except Exception as exc:
            print(f"INCOMPLETE: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            return

        rounds = result["rounds"]
        print(f"ROUNDS: {len(rounds)}", flush=True)
        print(f"PASSED: {result['final_passed']}", flush=True)
        for round_index, round_data in enumerate(rounds, start=1):
            print(f"  Round {round_index}:", flush=True)
            for verdict in round_data["verdicts"]:
                status = "PASS" if verdict["passed"] else "FAIL"
                print(f"    {status}  {verdict['reason']}", flush=True)

        if len(rounds) == 1 and result["final_passed"]:
            print("MODEL TOO GOOD: this prompt passed round 1", flush=True)
            return
        if len(rounds) <= 1 or not result["final_passed"]:
            print("FAILED: No successful fail-then-pass demo was captured.", flush=True)
            return

        run_dir = _newest_run_dir(runs_dir, existing_dirs)
        if run_dir is None:
            raise RuntimeError(f"Could not identify the run directory for {prompt!r}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("brief") != prompt:
            raise RuntimeError(f"Run summary does not match prompt: {run_dir}")
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
        shutil.copytree(run_dir, demo_dir)
        print(f"RUN DIR: {run_dir}", flush=True)
        print("SUCCESS: Captured a fail-then-pass demo.", flush=True)
        return


if __name__ == "__main__":
    main()
