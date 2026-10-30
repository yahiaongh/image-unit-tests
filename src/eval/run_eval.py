from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.clients import get_clients
from src.config import load_config
from src.json_utils import parse_constraints
from src.schema import Constraint
from src.verifier import filter_masks, verify_all

EVAL_MODES = ("one-shot", "random_retry", "pass@4", "loop")


def _as_bool(value) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


@dataclass
class EvalResult:
    passed: bool
    rounds: int
    attempts: int
    images_generated: int
    cost_usd: float
    failed_constraints: list[str]


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
    image_count = {"one-shot": 1, "pass@4": 4}.get(mode, max(1, rounds))
    return image_count * (image_cost + sam_cost * prompt_count)


def _verdicts_for_image(image_bytes, constraints, sam, config):
    with Image.open(BytesIO(image_bytes)) as image:
        image_shape = image.size
    unique_prompts = sorted({constraint.object_prompt for constraint in constraints})
    masks_by_prompt = {}
    iou_threshold = float(config.get("agent", {}).get("iou_threshold", 0.5))
    for prompt in unique_prompts:
        masks_by_prompt[prompt] = filter_masks(
            sam.segment(image_bytes, prompt), iou_threshold
        )
    return verify_all(constraints, masks_by_prompt, image_shape), unique_prompts


def run_one_shot(brief: dict, config: dict, clients, constraints: list[Constraint]) -> EvalResult:
    _, image, sam = clients
    image_bytes = image.generate(brief["brief"])
    verdicts, unique_prompts = _verdicts_for_image(image_bytes, constraints, sam, config)
    failed = [item["constraint_id"] for item in verdicts if not item["passed"]]
    return EvalResult(
        passed=not failed,
        rounds=1,
        attempts=1,
        images_generated=1,
        cost_usd=estimate_cost("one-shot", len(unique_prompts), 1, config),
        failed_constraints=failed,
    )


def run_pass_at_four(
    brief: dict,
    config: dict,
    clients,
    constraints: list[Constraint],
) -> EvalResult:
    _, image, sam = clients
    unique_prompts = sorted({constraint.object_prompt for constraint in constraints})
    first_passing_attempt = None
    last_failed_constraints = []

    for attempt in range(1, 5):
        image_bytes = image.generate(brief["brief"])
        verdicts, _ = _verdicts_for_image(image_bytes, constraints, sam, config)
        failed = [item["constraint_id"] for item in verdicts if not item["passed"]]
        if not failed and first_passing_attempt is None:
            first_passing_attempt = attempt
        if failed:
            last_failed_constraints = failed

    attempts_to_pass = first_passing_attempt or 4
    return EvalResult(
        passed=first_passing_attempt is not None,
        rounds=attempts_to_pass,
        attempts=attempts_to_pass,
        images_generated=4,
        cost_usd=estimate_cost("pass@4", len(unique_prompts), 4, config),
        failed_constraints=[] if first_passing_attempt is not None else last_failed_constraints,
    )


def run_random_retry(
    brief: dict,
    config: dict,
    clients,
    constraints: list[Constraint],
) -> EvalResult:
    _, image, sam = clients
    unique_prompts = sorted({constraint.object_prompt for constraint in constraints})
    failed_constraints = []

    for attempt in range(1, 5):
        image_bytes = image.generate(brief["brief"])
        verdicts, _ = _verdicts_for_image(image_bytes, constraints, sam, config)
        failed_constraints = [
            item["constraint_id"] for item in verdicts if not item["passed"]
        ]
        if not failed_constraints:
            break

    return EvalResult(
        passed=not failed_constraints,
        rounds=attempt,
        attempts=attempt,
        images_generated=attempt,
        cost_usd=estimate_cost("random_retry", len(unique_prompts), attempt, config),
        failed_constraints=failed_constraints,
    )


def run_loop(brief: dict, config: dict, clients, constraints: list[Constraint]) -> EvalResult:
    from src.agent import Agent

    spark, image, sam = clients
    result = Agent(spark, image, sam, config).run(brief["brief"], constraints=constraints)
    rounds_used = len(result["rounds"])
    failed = []
    if result["rounds"]:
        failed = [
            item["constraint_id"]
            for item in result["rounds"][-1]["verdicts"]
            if not item["passed"]
        ]
    prompt_count = len({constraint.object_prompt for constraint in constraints})
    return EvalResult(
        passed=bool(result["final_passed"]),
        rounds=rounds_used,
        attempts=rounds_used,
        images_generated=rounds_used,
        cost_usd=estimate_cost("loop", prompt_count, rounds_used, config),
        failed_constraints=failed,
    )


def _write_results(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "brief_id",
                "mode",
                "passed",
                "rounds",
                "attempts",
                "images_generated",
                "cost_usd",
                "failed_constraints",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in EVAL_MODES if any(row["mode"] == mode for row in rows)]
    lines = [
        "| Mode | Pass Rate | Avg Rounds (iterative modes) | Avg Cost per Image* |",
        "|------|-----------|------------------------------|---------------------|",
    ]
    for mode in modes:
        entries = [row for row in rows if row["mode"] == mode]
        evaluated_entries = [row for row in entries if not row.get("error")]
        pass_rate = (
            sum(_as_bool(row["passed"]) for row in evaluated_entries) / len(evaluated_entries)
            if evaluated_entries
            else 0.0
        )
        if mode in ("random_retry", "loop"):
            avg_rounds = (
                sum(float(row["rounds"]) for row in evaluated_entries) / len(evaluated_entries)
                if evaluated_entries
                else 0.0
            )
            rounds_text = f"{avg_rounds:.2f}"
        else:
            rounds_text = "—"
        image_count = sum(int(row["images_generated"]) for row in evaluated_entries)
        total_cost = sum(float(row["cost_usd"]) for row in evaluated_entries)
        avg_cost = total_cost / image_count if image_count else 0.0
        lines.append(
            f"| {mode} | {pass_rate:.2f} | {rounds_text} | ${avg_cost:.4f} |"
        )
    error_counts = [
        f"{mode}: {sum(bool(row.get('error')) for row in rows if row['mode'] == mode)}"
        for mode in modes
    ]
    lines.extend([
        "",
        "*Configured Image/SAM cost estimates; Spark costs are excluded. Rows with API errors are excluded from pass-rate, round, and per-image cost averages.",
        f"API error counts — {', '.join(error_counts)}.",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image generation briefs.")
    parser.add_argument("--mode", choices=EVAL_MODES, default="loop")
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--live", action="store_true", help="Use real clients instead of mock mode.")
    parser.add_argument("--briefs", type=str, default="src/eval/briefs.jsonl")
    args = parser.parse_args()

    config = load_config()
    use_mock = not args.live
    mock_config = config.setdefault("mock", {})
    mock_config.update({
        "enabled": use_mock,
        "spark": use_mock,
        "image": use_mock,
        "sam": use_mock,
    })
    max_cost_default = float(config.get("eval", {}).get("default_max_cost_usd", 2.00))
    max_cost = args.max_cost if args.max_cost is not None else max_cost_default
    if max_cost <= 0:
        parser.error("--max-cost must be greater than zero")

    briefs = load_briefs(args.briefs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("runs") / f"eval_{timestamp}"
    clients = get_clients(config)
    spark, _, _ = clients

    rows = []
    spent = 0.0
    reserved_spend = 0.0
    max_rounds = int(config.get("agent", {}).get("max_rounds", 4))
    print(
        "Cost guard uses configured Image/SAM rates and excludes Spark request costs.",
        flush=True,
    )
    for brief in briefs:
        try:
            constraints = parse_constraints(
                json.dumps(spark.generate_constraints(brief["brief"]))
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            rows.append({
                "brief_id": brief["id"],
                "mode": args.mode,
                "passed": False,
                "rounds": 0,
                "attempts": 0,
                "images_generated": 0,
                "cost_usd": 0.0,
                "failed_constraints": "",
                "error": message,
            })
            print(f"{brief['id']}: ERROR during constraint generation: {message}", flush=True)
            _write_results(output_dir, rows)
            _write_summary(output_dir, rows)
            continue
        prompt_count = len({constraint.object_prompt for constraint in constraints})
        reserved_mode = "loop" if args.mode == "loop" else args.mode
        reserved_rounds = (
            max_rounds if args.mode in ("loop", "random_retry") else 1
        )
        reserve = estimate_cost(reserved_mode, prompt_count, reserved_rounds, config)
        if reserved_spend + reserve > max_cost:
            print(
                f"Warning: stopping before brief {brief['id']}; worst-case configured "
                f"Image/SAM cost would exceed --max-cost=${max_cost:.2f}.",
                flush=True,
            )
            break
        reserved_spend += reserve

        try:
            if args.mode == "one-shot":
                result = run_one_shot(brief, config, clients, constraints)
            elif args.mode == "random_retry":
                result = run_random_retry(brief, config, clients, constraints)
            elif args.mode == "pass@4":
                result = run_pass_at_four(brief, config, clients, constraints)
            else:
                result = run_loop(brief, config, clients, constraints)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            max_images = (
                4 if args.mode in ("pass@4", "random_retry")
                else max_rounds if args.mode == "loop"
                else 1
            )
            spent += reserve
            rows.append({
                "brief_id": brief["id"],
                "mode": args.mode,
                "passed": False,
                "rounds": 0,
                "attempts": 0,
                "images_generated": max_images,
                "cost_usd": round(reserve, 6),
                "failed_constraints": "",
                "error": message,
            })
            print(
                f"{brief['id']}: ERROR during {args.mode}: {message}; "
                f"reserved configured cost=${reserve:.4f}",
                flush=True,
            )
            _write_results(output_dir, rows)
            _write_summary(output_dir, rows)
            continue

        spent += result.cost_usd
        rows.append({
            "brief_id": brief["id"],
            "mode": args.mode,
            "passed": result.passed,
            "rounds": result.rounds,
            "attempts": result.attempts,
            "images_generated": result.images_generated,
            "cost_usd": round(result.cost_usd, 6),
            "failed_constraints": ";".join(result.failed_constraints),
            "error": "",
        })
        print(
            f"{brief['id']}: {'PASS' if result.passed else 'FAIL'} "
            f"attempts={result.attempts} images={result.images_generated} "
            f"configured_cost=${result.cost_usd:.4f}",
            flush=True,
        )
        _write_results(output_dir, rows)
        _write_summary(output_dir, rows)

    print(
        f"Processed {len(rows)}/{len(briefs)} briefs; successful/failed evaluation "
        f"requests estimated at ${spent:.4f}, worst-case budget reserved "
        f"${reserved_spend:.4f}/${max_cost:.2f}. Wrote results to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
