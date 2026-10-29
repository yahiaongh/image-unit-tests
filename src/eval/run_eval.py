from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.clients import get_clients
from src.config import load_config
from src.json_utils import parse_constraints
from src.verifier import filter_masks, verify_all


def load_briefs(briefs_path: str | Path) -> list[dict]:
    path = Path(briefs_path)
    briefs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                briefs.append(json.loads(stripped))
    return briefs


def estimate_cost(mode: str, prompt_count: int, rounds: int, config: dict) -> float:
    eval_cfg = config.get("eval", {})
    image_cost = float(eval_cfg.get("cost_per_image_usd", 0.01))
    sam_cost = float(eval_cfg.get("cost_per_sam_call_usd", 0.0025))
    if mode == "one-shot":
        return image_cost + (sam_cost * max(1, prompt_count))
    return (image_cost + (sam_cost * max(1, prompt_count))) * max(1, rounds)


def run_one_shot(brief: dict, config: dict) -> tuple[bool, int, float, list[str]]:
    spark, image, sam = get_clients(config)
    constraints = parse_constraints(json.dumps(spark.generate_constraints(brief["brief"])))
    image_bytes = image.generate(brief["brief"])
    image_shape = Image.open(__import__("io").BytesIO(image_bytes)).size
    unique_prompts = sorted({c.object_prompt for c in constraints})
    masks_by_prompt = {}
    for prompt in unique_prompts:
        masks_by_prompt[prompt] = filter_masks(sam.segment(image_bytes, prompt), float(config.get("agent", {}).get("iou_threshold", 0.5)))
    verdicts = verify_all(constraints, masks_by_prompt, image_shape)
    passed = all(item["passed"] for item in verdicts)
    failed = [item["constraint_id"] for item in verdicts if not item["passed"]]
    cost = estimate_cost("one-shot", len(unique_prompts), 1, config)
    return passed, 1, cost, failed


def run_loop(brief: dict, config: dict) -> tuple[bool, int, float, list[str]]:
    spark, image, sam = get_clients(config)
    from src.agent import Agent

    agent = Agent(spark, image, sam, config)
    result = agent.run(brief["brief"])
    rounds_used = len(result["rounds"])
    passed = bool(result["final_passed"])
    failed = []
    if result["rounds"]:
        failed = [item["constraint_id"] for item in result["rounds"][-1]["verdicts"] if not item["passed"]]
    estimated_prompt_count = len({constraint.object_prompt for constraint in result["constraints"]})
    cost = estimate_cost("loop", estimated_prompt_count, rounds_used, config)
    return passed, rounds_used, cost, failed


def _write_results(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["brief_id", "mode", "passed", "rounds", "cost_usd", "failed_constraints"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(output_dir: Path, rows: list[dict]) -> None:
    modes = sorted({row["mode"] for row in rows})
    lines = [
        "| Mode | Pass rate | Avg rounds | Avg cost per image |",
        "|------|-----------|------------|---------------------|",
    ]
    for mode in modes:
        entries = [row for row in rows if row["mode"] == mode]
        pass_rate = sum(1 for row in entries if row["passed"]) / len(entries) if entries else 0.0
        avg_rounds = sum(float(row["rounds"]) for row in entries) / len(entries) if entries else 0.0
        avg_cost = sum(float(row["cost_usd"]) for row in entries) / len(entries) if entries else 0.0
        lines.append(f"| {mode} | {pass_rate:.2f} | {avg_rounds:.2f} | {avg_cost:.2f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image generation briefs.")
    parser.add_argument("--mode", choices=["one-shot", "loop"], default="loop")
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--live", action="store_true", help="Use real clients instead of mock mode.")
    parser.add_argument("--briefs", type=str, default="src/eval/briefs.jsonl")
    args = parser.parse_args()

    config = load_config()
    config.setdefault("mock", {})
    config["mock"]["enabled"] = not args.live
    max_cost_default = float(config.get("eval", {}).get("default_max_cost_usd", 2.00))
    max_cost = args.max_cost if args.max_cost is not None else max_cost_default

    briefs = load_briefs(args.briefs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("runs") / f"eval_{timestamp}"

    rows: list[dict] = []
    spent = 0.0
    for brief in briefs:
        estimated_prompt_count = 1
        cost_estimate = estimate_cost(args.mode, estimated_prompt_count, 1 if args.mode == "one-shot" else 1, config)
        if spent + cost_estimate > max_cost:
            print(f"Warning: skipping brief {brief['id']} because it would exceed --max-cost={max_cost:.2f} USD.")
            break

        if args.mode == "one-shot":
            passed, rounds_used, cost_usd, failed_constraints = run_one_shot(brief, config)
        else:
            passed, rounds_used, cost_usd, failed_constraints = run_loop(brief, config)

        spent += cost_usd
        rows.append({
            "brief_id": brief["id"],
            "mode": args.mode,
            "passed": bool(passed),
            "rounds": rounds_used,
            "cost_usd": round(cost_usd, 6),
            "failed_constraints": ";".join(failed_constraints),
        })

    _write_results(output_dir, rows)
    _write_summary(output_dir, rows)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
