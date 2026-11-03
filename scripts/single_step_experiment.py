from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import shutil
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clients import get_clients
from src.config import load_config
from src.json_utils import parse_constraints
from src.schema import Constraint
from src.verifier import filter_masks, verify_all

CONDITIONS = (
    "numeric_edit",
    "generic_edit",
    "marked_edit",
    "control_edit",
)
GENERIC_INSTRUCTION = "The image failed the constraint. Please fix the image to match the brief."
CONTROL_INSTRUCTION = "Please edit the image."
RESULT_COLUMNS = (
    "brief_id",
    "seed",
    "failure_type",
    "condition",
    "expected_count",
    "sam_before",
    "sam_after",
    "human_before",
    "human_after",
    "visible_artifact",
    "chosen_mask_id",
)
AUDIT_COLUMNS = (
    "before_filename",
    "after_filename",
    "brief",
    "human_before_count",
    "human_after_count",
    "visible_artifact (Y/N)",
)


def _object_names(object_prompt: str, count: int) -> tuple[str, str]:
    irregulars = {
        "people": ("person", "people"),
        "children": ("child", "children"),
    }
    if object_prompt.lower() in irregulars:
        singular, plural = irregulars[object_prompt.lower()]
        return (singular if count == 1 else plural, singular)
    lower_name = object_prompt.lower()
    if lower_name.endswith(("ches", "shes", "xes", "zes", "sses")):
        singular = object_prompt[:-2]
        plural = object_prompt
    elif lower_name.endswith("ies"):
        singular = object_prompt[:-3] + "y"
        plural = object_prompt
    elif lower_name.endswith("s") and not lower_name.endswith("ss"):
        singular = object_prompt[:-1]
        plural = object_prompt
    else:
        singular = object_prompt
        if lower_name.endswith("y") and len(object_prompt) > 1:
            plural = f"{object_prompt[:-1]}ies"
        elif lower_name.endswith(("s", "x", "z", "ch", "sh")):
            plural = f"{object_prompt}es"
        else:
            plural = f"{object_prompt}s"
    return (singular if count == 1 else plural, singular)


def _numeric_instruction(found: int, expected: int, object_prompt: str) -> str:
    difference = abs(found - expected)
    found_name, _ = _object_names(object_prompt, found)
    delta_name, _ = _object_names(object_prompt, difference)
    if found > expected:
        return (
            f"The image contains {found} {found_name}, which is {difference} too many. "
            f"Remove exactly {difference} {delta_name}."
        )
    return (
        f"The image contains {found} {found_name}, which is {difference} too few. "
        f"Add exactly {difference} more {delta_name}."
    )


def _save_png(image_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(path, format="PNG")


def _mask_center(mask_value) -> tuple[float, float] | None:
    mask = np.asarray(mask_value, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _mark_count_target(
    image_bytes: bytes,
    masks: list[dict],
    failure_type: str,
    expected_count: int,
) -> tuple[bytes, str]:
    with Image.open(BytesIO(image_bytes)) as image:
        marked = image.convert("RGB")
    width, height = marked.size
    draw = ImageDraw.Draw(marked)

    if failure_type == "over":
        centers = [
            (index, _mask_center(entry.get("mask")))
            for index, entry in enumerate(masks, start=1)
        ]
        valid_centers = [
            (mask_id, center)
            for mask_id, center in centers
            if center is not None
        ]
        if len(valid_centers) <= expected_count:
            raise ValueError(
                "Overcount marking requires more valid masks than the expected count"
            )
        chosen_mask_id, (center_x, center_y) = max(
            valid_centers,
            key=lambda item: item[1][0],
        )
        mask = np.asarray(masks[chosen_mask_id - 1]["mask"], dtype=bool)
        ys, xs = np.where(mask)
        padding = max(6, min(width, height) // 100)
        box = (
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(width - 1, int(xs.max()) + padding),
            min(height - 1, int(ys.max()) + padding),
        )
        draw.rectangle(box, outline=(255, 0, 0), width=10)
        return _encoded_png(marked), f"mask_{chosen_mask_id:03d}"

    if failure_type != "under":
        raise ValueError(f"Unsupported failure type: {failure_type}")
    marker_size = max(30, min(width, height) // 12)
    center_x, center_y = width // 2, height // 2
    draw.rectangle(
        (
            max(0, center_x - marker_size),
            max(0, center_y - marker_size),
            min(width - 1, center_x + marker_size),
            min(height - 1, center_y + marker_size),
        ),
        outline=(255, 0, 0),
        width=10,
    )
    return _encoded_png(marked), ""


def _encoded_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _evaluate(
    image_bytes: bytes,
    constraints: list[Constraint],
    sam,
    config: dict,
) -> tuple[list[dict], dict[str, list[dict]]]:
    with Image.open(BytesIO(image_bytes)) as image:
        image_shape = image.size
    iou_threshold = float(config.get("agent", {}).get("iou_threshold", 0.5))
    filtered_masks: dict[str, list[dict]] = {}
    raw_masks: dict[str, list[dict]] = {}
    for prompt in sorted({constraint.object_prompt for constraint in constraints}):
        raw = sam.segment(image_bytes, prompt)
        raw_masks[prompt] = raw
        filtered_masks[prompt] = filter_masks(raw, iou_threshold)
    verdicts = verify_all(
        constraints,
        filtered_masks,
        image_shape,
        ambiguity_masks_by_prompt=raw_masks,
    )
    return verdicts, filtered_masks


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_blinded_audit(
    records: list[dict],
    audit_dir: Path,
    ground_truth_path: Path,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    shuffled_records = list(records)
    random.SystemRandom().shuffle(shuffled_records)
    mapping = {}
    ground_truth_rows = []
    for record in shuffled_records:
        filename_token = secrets.token_hex(6)
        before_filename = f"before_{filename_token}.png"
        after_filename = f"after_{filename_token}.png"
        before_path = audit_dir / before_filename
        after_path = audit_dir / after_filename
        shutil.copyfile(record["before_path"], before_path)
        shutil.copyfile(record["after_path"], after_path)
        metadata = {
            "brief_id": record["brief_id"],
            "condition": record["condition"],
            "failure_type": record["failure_type"],
            "seed": record["seed"],
        }
        mapping[before_filename] = metadata
        mapping[after_filename] = metadata
        ground_truth_rows.append({
            "before_filename": f"single_step/{before_filename}",
            "after_filename": f"single_step/{after_filename}",
            "brief": record["brief"],
            "human_before_count": "",
            "human_after_count": "",
            "visible_artifact (Y/N)": "",
        })

    (audit_dir / "mapping.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )
    _write_csv(ground_truth_path, AUDIT_COLUMNS, ground_truth_rows)


def _write_summary(rows: list[dict]) -> None:
    print("\n| Condition | Failure Type | Mean |Error| Reduction | N (Images) |")
    print("|-----------|--------------|----------------------------|------------:|")
    for condition in CONDITIONS:
        for failure_type, label in (("over", "Overcount"), ("under", "Undercount")):
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["failure_type"] == failure_type
                and row["sam_after"] != ""
            ]
            reductions = [
                abs(int(row["sam_before"]) - int(row["expected_count"]))
                - abs(int(row["sam_after"]) - int(row["expected_count"]))
                for row in selected
            ]
            mean_reduction = (
                f"{sum(reductions) / len(reductions):.3f}"
                if reductions
                else "n/a"
            )
            print(
                f"| {condition.removesuffix('_edit').title()} | {label} | "
                f"{mean_reduction} | {len(selected)} |"
            )

    print("\n| Condition | Perfect fixes | N (Images) |")
    print("|-----------|---------------:|------------:|")
    for condition in CONDITIONS:
        selected = [
            row
            for row in rows
            if row["condition"] == condition and row["sam_after"] != ""
        ]
        perfect = sum(
            abs(int(row["sam_after"]) - int(row["expected_count"])) == 0
            for row in selected
        )
        percentage = 100 * perfect / len(selected) if selected else 0.0
        print(f"| {condition.removesuffix('_edit').title()} | {percentage:.1f}% | {len(selected)} |")


def run_experiment(
    briefs_path: Path,
    runs_dir: Path,
    audit_dir: Path,
    ground_truth_path: Path,
    seeds: tuple[int, ...] = (1, 2),
) -> list[dict]:
    config = load_config()
    config.setdefault("mock", {}).update({
        "enabled": False,
        "spark": False,
        "image": False,
        "sam": False,
    })
    spark, image, sam = get_clients(config)
    with briefs_path.open(encoding="utf-8") as handle:
        briefs = [json.loads(line) for line in handle if line.strip()]

    result_rows = []
    error_rows = []
    audit_records = []
    for brief_item in briefs:
        brief_id = brief_item["id"]
        brief = brief_item["brief"]
        try:
            constraints = parse_constraints(
                json.dumps(spark.generate_constraints(brief))
            )
            count_constraints = [
                constraint
                for constraint in constraints
                if constraint.type == "count"
                and constraint.expected_count is not None
            ]
            if not count_constraints:
                raise ValueError("Spark did not return a count constraint")
            count_constraint = count_constraints[0]
        except Exception as exc:
            error_rows.append({
                "brief_id": brief_id,
                "seed": "",
                "stage": "constraints",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{brief_id}: ERROR during constraint extraction: {exc}")
            continue

        for seed in seeds:
            try:
                original = image.generate(brief)
                initial_verdicts, initial_masks = _evaluate(
                    original,
                    constraints,
                    sam,
                    config,
                )
                count_verdict = next(
                    verdict
                    for verdict in initial_verdicts
                    if verdict["constraint_id"] == count_constraint.id
                )
                prompt_masks = initial_masks.get(count_constraint.object_prompt, [])
                sam_before = len(prompt_masks)
                expected_count = int(count_constraint.expected_count)

                if count_verdict.get("status", "FAIL") != "FAIL":
                    label = count_verdict.get("status", "PASS")
                    print(
                        f"{brief_id} seed={seed}: skipped "
                        f"(count verdict {label}; SAM={sam_before}, expected={expected_count})"
                    )
                    continue
                if sam_before == expected_count:
                    print(
                        f"{brief_id} seed={seed}: skipped "
                        "(count is uncertain, not a directional over/under-count)"
                    )
                    continue

                failure_type = "over" if sam_before > expected_count else "under"
                failure_label = "OVERCOUNT" if failure_type == "over" else "UNDERCOUNT"
                print(
                    f"{brief_id} seed={seed}: {failure_label}, "
                    f"SAM={sam_before}, expected={expected_count}"
                )
                replicate_dir = runs_dir / brief_id / f"seed_{seed}"
                original_path = replicate_dir / "original.png"
                marked_path = replicate_dir / "marked_input.png"
                _save_png(original, original_path)
                marked_input, chosen_mask_id = _mark_count_target(
                    original,
                    prompt_masks,
                    failure_type,
                    expected_count,
                )
                _save_png(marked_input, marked_path)

                numeric_instruction = _numeric_instruction(
                    sam_before,
                    expected_count,
                    count_constraint.object_prompt,
                )
                marked_instruction = (
                    "Remove the object marked with a red box."
                    if failure_type == "over"
                    else "Add exactly "
                    f"{expected_count - sam_before} more "
                    f"{_object_names(count_constraint.object_prompt, expected_count - sam_before)[1]} "
                    "in the region marked with a red box."
                )
                instructions = {
                    "numeric_edit": numeric_instruction,
                    "generic_edit": GENERIC_INSTRUCTION,
                    "marked_edit": marked_instruction,
                    "control_edit": CONTROL_INSTRUCTION,
                }

                for condition in CONDITIONS:
                    input_bytes = (
                        marked_input if condition == "marked_edit" else original
                    )
                    try:
                        edited = image.edit(input_bytes, instructions[condition])
                        after_path = replicate_dir / f"{condition}.png"
                        _save_png(edited, after_path)
                        edited_verdicts, edited_masks = _evaluate(
                            edited,
                            constraints,
                            sam,
                            config,
                        )
                        edited_count_verdict = next(
                            verdict
                            for verdict in edited_verdicts
                            if verdict["constraint_id"] == count_constraint.id
                        )
                        sam_after = len(
                            edited_masks.get(count_constraint.object_prompt, [])
                        )
                        row = {
                            "brief_id": brief_id,
                            "seed": seed,
                            "failure_type": failure_type,
                            "condition": condition,
                            "expected_count": expected_count,
                            "sam_before": sam_before,
                            "sam_after": sam_after,
                            "human_before": "",
                            "human_after": "",
                            "visible_artifact": "",
                            "chosen_mask_id": (
                                chosen_mask_id if condition == "marked_edit" else ""
                            ),
                        }
                        result_rows.append(row)
                        audit_records.append({
                            "brief_id": brief_id,
                            "brief": brief,
                            "seed": seed,
                            "condition": condition,
                            "failure_type": failure_type,
                            "before_path": original_path,
                            "after_path": after_path,
                        })
                        print(
                            f"  {condition}: SAM after={sam_after}, "
                            f"|error| reduction="
                            f"{abs(sam_before - expected_count) - abs(sam_after - expected_count)}"
                        )
                    except Exception as exc:
                        error_rows.append({
                            "brief_id": brief_id,
                            "seed": seed,
                            "stage": condition,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        print(f"  {condition}: ERROR: {type(exc).__name__}: {exc}")
                    _write_csv(runs_dir / "results.csv", RESULT_COLUMNS, result_rows)
                    _write_csv(
                        runs_dir / "errors.csv",
                        ("brief_id", "seed", "stage", "error"),
                        error_rows,
                    )
            except Exception as exc:
                error_rows.append({
                    "brief_id": brief_id,
                    "seed": seed,
                    "stage": "generation_or_initial_verification",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"{brief_id} seed={seed}: ERROR: {type(exc).__name__}: {exc}")

    _write_csv(runs_dir / "results.csv", RESULT_COLUMNS, result_rows)
    _write_csv(
        runs_dir / "errors.csv",
        ("brief_id", "seed", "stage", "error"),
        error_rows,
    )
    _prepare_blinded_audit(audit_records, audit_dir, ground_truth_path)
    _write_summary(result_rows)
    print(
        f"\nCompleted {len(briefs)} prompts x {len(seeds)} replicates; "
        f"{len(audit_records)} edited images included in blinded audit. "
        f"API errors: {len(error_rows)}. "
        "The numeric seed values label independent repetitions; the image API "
        "does not accept an explicit seed parameter."
    )
    print(f"Results: {runs_dir / 'results.csv'}")
    print(f"Blinded audit images and mapping: {audit_dir}")
    print(f"Human review sheet: {ground_truth_path}")
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired single-step image edit experiment."
    )
    parser.add_argument(
        "--briefs",
        type=Path,
        default=Path("src/eval/single_step_briefs.jsonl"),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/single_step"))
    parser.add_argument("--audit-dir", type=Path, default=Path("runs/audit/single_step"))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("runs/audit/ground_truth.csv"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Independent generation replicate labels; API generation is not seed-controlled.",
    )
    args = parser.parse_args()
    run_experiment(
        args.briefs,
        args.runs_dir,
        args.audit_dir,
        args.ground_truth,
        tuple(args.seeds),
    )


if __name__ == "__main__":
    main()
