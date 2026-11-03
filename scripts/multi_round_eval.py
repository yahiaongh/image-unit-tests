from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from scripts.analyze_human_audit import load_human_verified_rows
from src.agent import Agent
from src.clients import get_clients
from src.config import load_config
from src.schema import Constraint
from src.verifier import filter_masks, verify_all

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "runs" / "single_step" / "results.csv"
AUDIT_CSV = ROOT / "runs" / "audit" / "master_ground_truth.csv"
AUDIT_DIR = ROOT / "runs" / "audit"
BRIEFS_PATH = ROOT / "src" / "eval" / "single_step_briefs.jsonl"
OUTPUT_DIR = ROOT / "runs" / "multi_round"
TRAJECTORY_FIELDS = (
    "start_id",
    "condition",
    "round",
    "sam_count",
    "expected_count",
)
CONDITIONS = ("Numeric Loop", "Generic Loop", "Pass@4")


def _start_id(brief_id: str, seed: str) -> str:
    return f"{brief_id}_seed_{seed}"


def _condition_dir(condition: str) -> str:
    if condition == "Pass@4":
        return "pass_4"
    return condition.lower().replace(" ", "_")


def _make_constraint(start: dict[str, object]) -> Constraint:
    expected = int(start["expected_count"])
    object_prompt = str(start["object_prompt"])
    return Constraint(
        id="audited_count",
        type="count",
        description=f"Exactly {expected} {object_prompt} are visible",
        object_prompt=object_prompt,
        expected_count=expected,
    )


def select_human_overcount_cohort(
    results_path: Path = RESULTS_PATH,
    audit_csv: Path = AUDIT_CSV,
    audit_dir: Path = AUDIT_DIR,
) -> list[dict[str, object]]:
    _, raw_starts = load_human_verified_rows(results_path, audit_csv, audit_dir)
    with results_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    with BRIEFS_PATH.open(encoding="utf-8") as handle:
        briefs = {
            entry["id"]: entry["brief"]
            for line in handle
            if line.strip()
            for entry in [json.loads(line)]
        }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(row["brief_id"], row["seed"])].append(row)

    prompt_map_path = audit_dir / "master_prompt_map.json"
    prompt_map = json.loads(prompt_map_path.read_text(encoding="utf-8"))
    audit_counts = {}
    with audit_csv.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    for row in audit_rows:
        audit_image = (audit_dir / row["filename"]).resolve()
        digest = hashlib.sha256(audit_image.read_bytes()).hexdigest()
        audit_counts[digest] = (row, prompt_map[row["filename"]])

    selected = []
    for start in raw_starts:
        key = (str(start["brief_id"]), str(start["seed"]))
        expected = int(start["expected_count"])
        if start["failure_type"] != "over" or int(start["human_count"]) <= expected:
            continue
        digest = hashlib.sha256(Path(start["image_path"]).read_bytes()).hexdigest()
        audit_row, object_prompt = audit_counts[digest]
        start_rows = grouped[key]
        if len(start_rows) != 4:
            raise ValueError(f"{key} has {len(start_rows)} outcomes; expected 4.")
        selected.append({
            **start,
            "start_id": _start_id(*key),
            "brief": briefs[str(start["brief_id"])],
            "object_prompt": object_prompt,
            "image_path": Path(start["image_path"]),
            "initial_human_count": int(audit_row["human_count"]),
        })

    selected.sort(key=lambda item: str(item["start_id"]))
    if len(selected) != 19:
        raise ValueError(
            f"Expected 19 pre-labeled overcount starts confirmed by humans, found "
            f"{len(selected)}."
        )
    return selected


def _save_png(image_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.convert("RGB").save(path, format="PNG")


def _sam_count(verdicts: list[dict], constraint_id: str) -> int:
    for verdict in verdicts:
        if verdict["constraint_id"] == constraint_id:
            return len(verdict.get("masks_used") or [])
    raise ValueError(f"Verifier omitted constraint {constraint_id!r}.")


def _write_trajectory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _completed_ids(path: Path) -> set[tuple[str, str]]:
    complete = set()
    trajectory_path = path / "trajectory.csv"
    trajectory_keys = set()
    if trajectory_path.is_file():
        with trajectory_path.open(newline="", encoding="utf-8") as handle:
            trajectory_keys = {
                (row["start_id"], row["condition"])
                for row in csv.DictReader(handle)
            }
    for start_dir in path.glob("*/*"):
        marker = start_dir / "complete.json"
        if marker.is_file():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            key = (payload["start_id"], payload["condition"])
            if key in trajectory_keys:
                complete.add(key)
    return complete


def _run_loop_arm(
    start: dict[str, object],
    condition: str,
    clients,
    config: dict,
) -> list[dict[str, object]]:
    brief = str(start["brief"])
    initial = Path(start["image_path"]).read_bytes()
    constraint = _make_constraint(start)
    arm_config = dict(config)
    arm_config["agent"] = {
        **config.get("agent", {}),
        "max_rounds": 5,
    }
    arm_config["runs_dir"] = str(
        OUTPUT_DIR / _condition_dir(condition) / str(start["start_id"]) / "agent"
    )
    agent = Agent(*clients, arm_config)
    outcome = agent.run(
        brief,
        constraints=[constraint],
        numeric_feedback=condition == "Numeric Loop",
        initial_image=initial,
    )

    run_dir = Path(agent.runs_dir) / Path(outcome["run_dir"]).name
    arm_dir = OUTPUT_DIR / _condition_dir(condition) / str(start["start_id"])
    arm_dir.mkdir(parents=True, exist_ok=True)
    trajectory = []
    for agent_round in outcome["rounds"]:
        internal_round = int(agent_round["round"])
        public_round = internal_round - 1
        image_path = run_dir / f"round_{internal_round}.png"
        if image_path.is_file():
            shutil.copyfile(image_path, arm_dir / f"round_{public_round}.png")
        trajectory.append({
            "start_id": start["start_id"],
            "condition": condition,
            "round": public_round,
            "sam_count": _sam_count(agent_round["verdicts"], constraint.id),
            "expected_count": int(start["expected_count"]),
        })
    return trajectory


def _run_pass_at_four(
    start: dict[str, object],
    clients,
    config: dict,
) -> list[dict[str, object]]:
    _, image, sam = clients
    brief = str(start["brief"])
    constraint = _make_constraint(start)
    output_dir = OUTPUT_DIR / "pass_4" / str(start["start_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = []
    for attempt in range(1, 5):
        image_bytes = image.generate(brief)
        image_shape = Image.open(io.BytesIO(image_bytes)).size
        raw_masks = sam.segment(image_bytes, constraint.object_prompt)
        masks = filter_masks(raw_masks, float(config.get("agent", {}).get("iou_threshold", 0.5)))
        verdicts = verify_all(
            [constraint],
            {constraint.object_prompt: masks},
            image_shape,
            ambiguity_masks_by_prompt={constraint.object_prompt: raw_masks},
        )
        count = _sam_count(verdicts, constraint.id)
        _save_png(image_bytes, output_dir / f"round_{attempt}.png")
        trajectory.append({
            "start_id": start["start_id"],
            "condition": "Pass@4",
            "round": attempt,
            "sam_count": count,
            "expected_count": int(start["expected_count"]),
        })
        if all(item["passed"] for item in verdicts):
            break
    return trajectory


def run_experiment(
    cohort: list[dict[str, object]],
    clients,
    config: dict,
    output_dir: Path = OUTPUT_DIR,
) -> list[dict[str, object]]:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    trajectory_path = output_dir / "trajectory.csv"
    rows = []
    if trajectory_path.is_file():
        with trajectory_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            for row in rows:
                row["round"] = int(row["round"])
                row["sam_count"] = int(row["sam_count"])
                row["expected_count"] = int(row["expected_count"])
    completed = _completed_ids(output_dir)
    for start in cohort:
        for condition, runner in (
            ("Numeric Loop", _run_loop_arm),
            ("Generic Loop", _run_loop_arm),
            ("Pass@4", _run_pass_at_four),
        ):
            start_id = str(start["start_id"])
            if (start_id, condition) in completed:
                continue
            if runner is _run_loop_arm:
                new_rows = runner(start, condition, clients, config)
            else:
                new_rows = runner(start, clients, config)
            rows = [
                row for row in rows
                if (row["start_id"], row["condition"]) != (start_id, condition)
            ]
            rows.extend(new_rows)
            _write_trajectory(trajectory_path, rows)
            arm_dir = output_dir / _condition_dir(condition) / start_id
            arm_dir.mkdir(parents=True, exist_ok=True)
            (arm_dir / "complete.json").write_text(
                json.dumps({"start_id": start_id, "condition": condition}, indent=2),
                encoding="utf-8",
            )
            completed.add((start_id, condition))
            print(f"Completed {condition}: {start_id}", flush=True)
    return rows


def _print_success_rates(rows: list[dict[str, object]], cohort_size: int) -> None:
    print("| Condition | Reached error 0 by round 4 | N |")
    print("|-----------|----------------------------|---|")
    for condition in CONDITIONS:
        by_start: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if row["condition"] == condition:
                by_start[str(row["start_id"])].append(row)
        successes = sum(
            any(
                abs(int(row["sam_count"]) - int(row["expected_count"])) == 0
                and 0 <= int(row["round"]) <= 4
                for row in trials
            )
            for trials in by_start.values()
        )
        print(f"| {condition} | {successes}/{cohort_size} ({100 * successes / cohort_size:.1f}%) | {cohort_size} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run human-confirmed overcount starts through numeric/generic loops and Pass@4."
    )
    parser.add_argument("--max-cost", type=float, default=3.00)
    parser.add_argument("--live", action="store_true", help="Use real image and SAM clients.")
    args = parser.parse_args()
    if args.max_cost <= 0:
        parser.error("--max-cost must be greater than zero")

    config = load_config()
    if not args.live:
        config["mock"].update({"enabled": True, "spark": True, "image": True, "sam": True})
    else:
        config["mock"].update({"enabled": False, "spark": False, "image": False, "sam": False})
    config["agent"] = {**config.get("agent", {}), "max_rounds": 5}
    cohort = select_human_overcount_cohort()
    image_cost = float(config.get("eval", {}).get("cost_per_image_usd", 0.01))
    sam_cost = float(config.get("eval", {}).get("cost_per_sam_call_usd", 0.0025))
    prompt_count = 1
    worst_case = len(cohort) * (
        2 * (4 * image_cost + 5 * sam_cost * prompt_count)
        + 4 * (image_cost + sam_cost * prompt_count)
    )
    if worst_case > args.max_cost:
        parser.error(
            f"Configured worst-case image/SAM estimate ${worst_case:.2f} exceeds "
            f"--max-cost=${args.max_cost:.2f}; Spark charges are not included."
        )
    print(
        f"Running {len(cohort)} human-confirmed, pre-labeled overcount starts. "
        f"Configured maximum Image/SAM estimate: ${worst_case:.2f}; Spark excluded.",
        flush=True,
    )
    clients = get_clients(config)
    rows = run_experiment(cohort, clients, config)
    _print_success_rates(rows, len(cohort))


if __name__ == "__main__":
    main()
