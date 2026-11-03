from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

CONDITIONS = (
    ("numeric_edit", "Numeric Feedback"),
    ("generic_edit", "Generic Feedback"),
    ("marked_edit", "Marked Image"),
    ("control_edit", "Control"),
)
RESULT_FIELDS = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_audit_path(audit_dir: Path, filename: str) -> Path:
    path = (audit_dir / filename).resolve()
    if not path.is_relative_to(audit_dir.resolve()):
        raise ValueError(f"Audit filename escapes the audit directory: {filename}")
    if not path.is_file():
        raise FileNotFoundError(f"Audited image is missing: {path}")
    return path


def load_human_verified_rows(
    results_path: Path,
    audit_csv: Path,
    audit_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with results_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    with audit_csv.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))

    human_by_hash: dict[str, dict[str, str]] = {}
    for row in audit_rows:
        if not row.get("human_count", "").strip():
            raise ValueError(f"Human count is missing for {row.get('filename')!r}.")
        filename = row["filename"]
        image_hash = _sha256(_safe_audit_path(audit_dir, filename))
        if image_hash in human_by_hash:
            raise ValueError(f"Duplicate audited image content for {filename!r}.")
        human_by_hash[image_hash] = row

    source_by_hash: dict[str, tuple[str, str, Path]] = {}
    starts: dict[tuple[str, str], dict[str, str | Path | int]] = {}
    grouped_results: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped_results[(row["brief_id"], row["seed"])].append(row)

    for (brief_id, seed), rows in grouped_results.items():
        replicate_dir = results_path.parent / brief_id / f"seed_{seed}"
        if not replicate_dir.is_dir():
            replicate_dir = results_path.parent / brief_id
        start_path = replicate_dir / "original.png"
        if not start_path.is_file():
            raise FileNotFoundError(f"Starting image is missing: {start_path}")
        start_hash = _sha256(start_path)
        source_by_hash[start_hash] = (brief_id, "start", start_path)
        starts[(brief_id, seed)] = {
            "brief_id": brief_id,
            "seed": seed,
            "failure_type": rows[0]["failure_type"].strip().lower(),
            "expected_count": int(rows[0]["expected_count"]),
            "image_path": start_path,
            "image_hash": start_hash,
        }

        for row in rows:
            condition = row["condition"].strip().lower()
            image_path = replicate_dir / f"{condition}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Outcome image is missing: {image_path}")
            image_hash = _sha256(image_path)
            if image_hash in source_by_hash:
                previous = source_by_hash[image_hash]
                raise ValueError(
                    f"Image content is not unique: {previous[2]} and {image_path}"
                )
            source_by_hash[image_hash] = (brief_id, condition, image_path)

    if len(human_by_hash) != len(audit_rows):
        raise ValueError("Audit filenames must uniquely identify audited images.")
    if set(human_by_hash) != set(source_by_hash):
        missing_audits = set(source_by_hash) - set(human_by_hash)
        missing_sources = set(human_by_hash) - set(source_by_hash)
        raise ValueError(
            "Audit/source image hash sets do not match: "
            f"{len(missing_audits)} source images lack human verdicts, "
            f"{len(missing_sources)} audited images have no source artifact."
        )

    human_start_counts = {
        key: int(human_by_hash[start["image_hash"]]["human_count"])
        for key, start in starts.items()
    }
    joined_rows = []
    for row in result_rows:
        brief_id, seed = row["brief_id"], row["seed"]
        replicate_dir = results_path.parent / brief_id / f"seed_{seed}"
        if not replicate_dir.is_dir():
            replicate_dir = results_path.parent / brief_id
        before_path = replicate_dir / "original.png"
        after_path = replicate_dir / f"{row['condition'].strip().lower()}.png"
        before_audit = human_by_hash[_sha256(before_path)]
        after_audit = human_by_hash[_sha256(after_path)]
        joined_rows.append({
            **row,
            "human_before": before_audit["human_count"],
            "human_after": after_audit["human_count"],
            "visible_artifact": after_audit.get("visible_artifact", ""),
        })

    for start in starts.values():
        start["human_count"] = human_start_counts[
            (str(start["brief_id"]), str(start["seed"]))
        ]
    return joined_rows, list(starts.values())


def _human_summaries(
    joined_rows: list[dict[str, str]],
    starts: list[dict[str, str | Path | int]],
) -> tuple[list[dict[str, str | int]], dict[str, int]]:
    cohort = {
        (str(start["brief_id"]), str(start["seed"]))
        for start in starts
        if start["failure_type"] == "over"
        and int(start["human_count"]) > int(start["expected_count"])
    }
    if len(cohort) != 19:
        raise ValueError(
            f"Expected 19 pre-labeled overcount starts confirmed by humans, found "
            f"{len(cohort)}."
        )
    groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    false_passes = sam_passes = 0
    for row in joined_rows:
        key = (row["brief_id"], row["seed"])
        expected = int(row["expected_count"])
        sam_after = int(row["sam_after"])
        human_after = int(row["human_after"])
        if sam_after == expected:
            sam_passes += 1
            false_passes += human_after != expected
        if key in cohort:
            groups[row["condition"].strip().lower()].append(
                (int(row["human_before"]), human_after, expected)
            )

    summaries = []
    for condition, label in CONDITIONS:
        trials = groups.get(condition, [])
        if len(trials) != len(cohort):
            raise ValueError(
                f"{label} has {len(trials)} human-verified overcount trials; "
                f"expected {len(cohort)}."
            )
        perfect = improved = unchanged = worse = 0
        reductions = []
        for before, after, expected in trials:
            before_error = abs(before - expected)
            after_error = abs(after - expected)
            reductions.append(before_error - after_error)
            if after_error == 0:
                perfect += 1
            elif after < expected:
                worse += 1
            elif after_error < before_error:
                improved += 1
            elif after_error == before_error:
                unchanged += 1
            else:
                worse += 1
        summaries.append({
            "condition": label,
            "n": len(trials),
            "mean_reduction": f"{sum(reductions) / len(reductions):+.2f}",
            "perfect": f"{100 * perfect / len(trials):.1f}%",
            "improved": f"{100 * improved / len(trials):.1f}%",
            "unchanged": f"{100 * unchanged / len(trials):.1f}%",
            "worse": f"{100 * worse / len(trials):.1f}%",
        })
    return summaries, {"false_passes": false_passes, "sam_passes": sam_passes}


def _uncertain_catch_summary(
    joined_rows: list[dict[str, str]],
) -> dict[str, int] | None:
    false_pass_rows = [
        row
        for row in joined_rows
        if int(row["sam_after"]) == int(row["expected_count"])
        and int(row["human_after"]) != int(row["expected_count"])
    ]
    if not false_pass_rows:
        return {"caught": 0, "total": 0}
    status_key = next(
        (key for key in ("verdict_status", "status") if key in false_pass_rows[0]),
        None,
    )
    if status_key is None or any(
        not row.get(status_key, "").strip() for row in false_pass_rows
    ):
        return None
    return {
        "caught": sum(
            row[status_key].strip().upper() == "UNCERTAIN"
            for row in false_pass_rows
        ),
        "total": len(false_pass_rows),
    }


def print_markdown(summaries: list[dict[str, str | int]]) -> None:
    print(
        "| Condition | N | Mean Human Error Reduction | Perfect Fix | "
        "Improved | Unchanged | Worse / Overshoot |"
    )
    print(
        "|-----------|---|----------------------------|-------------|"
        "----------|-----------|-------------------|"
    )
    for row in summaries:
        print(
            f"| {row['condition']} | {row['n']} | {row['mean_reduction']} | "
            f"{row['perfect']} | {row['improved']} | {row['unchanged']} | "
            f"{row['worse']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join blinded human counts to experiment outcomes and report human-only statistics."
    )
    parser.add_argument("--results", type=Path, default=Path("runs/single_step/results.csv"))
    parser.add_argument("--audit-csv", type=Path, default=Path("runs/audit/master_ground_truth.csv"))
    parser.add_argument("--audit-dir", type=Path, default=Path("runs/audit"))
    parser.add_argument(
        "--joined-output",
        type=Path,
        default=Path("runs/single_step/human_verified_results.csv"),
    )
    args = parser.parse_args()

    joined, starts = load_human_verified_rows(args.results, args.audit_csv, args.audit_dir)
    summaries, false_pass = _human_summaries(joined, starts)
    uncertain_catch = _uncertain_catch_summary(joined)
    args.joined_output.parent.mkdir(parents=True, exist_ok=True)
    with args.joined_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(joined)

    print("Human-verified N=19 overcount cohort (human counts determine all outcomes):")
    print_markdown(summaries)
    rate = (
        100 * false_pass["false_passes"] / false_pass["sam_passes"]
        if false_pass["sam_passes"]
        else 0.0
    )
    print(
        f"\nCount-based SAM false passes: {false_pass['false_passes']} / "
        f"{false_pass['sam_passes']} SAM-count passes ({rate:.1f}%)."
    )
    if uncertain_catch is None:
        print(
            "UNCERTAIN catch rate: unavailable. Verdict status was not recorded "
            "for the false-pass rows, so the count cannot be reconstructed."
        )
    else:
        caught = uncertain_catch["caught"]
        total = uncertain_catch["total"]
        catch_rate = 100 * caught / total if total else 0.0
        print(
            f"UNCERTAIN catch rate: {caught} / {total} SAM false passes "
            f"({catch_rate:.1f}%)."
        )
    human_overcounts = [
        start for start in starts
        if int(start["human_count"]) > int(start["expected_count"])
    ]
    cohort_size = int(summaries[0]["n"]) if summaries else 0
    print(
        f"Human-confirmed overcounts: {len(human_overcounts)} total; "
        f"{cohort_size} are reported in the pre-labeled N=19 cohort."
    )


if __name__ == "__main__":
    main()
