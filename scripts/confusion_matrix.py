from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("runs/single_step/results.csv")
AUDIT_CSV_PATH = Path("runs/audit/master_ground_truth.csv")
AUDIT_DIR = Path("runs/audit")

CONDITION_LABELS = {
    "numeric_edit": "Numeric",
    "generic_edit": "Generic",
    "marked_edit": "Marked",
    "control_edit": "Control",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_records(results_path: Path) -> dict[str, dict[str, object]]:
    with results_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    if not result_rows:
        raise ValueError(f"No experiment rows found in {results_path}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(row["brief_id"], row["seed"])].append(row)

    records: dict[str, dict[str, object]] = {}

    def add_record(
        image_path: Path,
        *,
        image_type: str,
        condition: str,
        expected: int,
        sam_count: int,
    ) -> None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Experiment image is missing: {image_path}")
        image_hash = _sha256(image_path)
        if image_hash in records:
            raise ValueError(f"Duplicate source image content: {image_path}")
        records[image_hash] = {
            "image_type": image_type,
            "condition": condition,
            "expected_count": expected,
            "sam_count": sam_count,
        }

    for (brief_id, seed), rows in grouped.items():
        replicate_dir = results_path.parent / brief_id / f"seed_{seed}"
        if not replicate_dir.is_dir():
            replicate_dir = results_path.parent / brief_id
        expected_values = {int(row["expected_count"]) for row in rows}
        before_values = {int(row["sam_before"]) for row in rows}
        if len(expected_values) != 1 or len(before_values) != 1:
            raise ValueError(
                f"Inconsistent starting counts for {brief_id} seed {seed}"
            )
        expected = expected_values.pop()
        sam_before = before_values.pop()
        add_record(
            replicate_dir / "original.png",
            image_type="start",
            condition="start",
            expected=expected,
            sam_count=sam_before,
        )
        for row in rows:
            condition = row["condition"].strip().lower()
            if condition not in CONDITION_LABELS:
                raise ValueError(f"Unknown experiment condition: {condition!r}")
            add_record(
                replicate_dir / f"{condition}.png",
                image_type="edit",
                condition=condition,
                expected=int(row["expected_count"]),
                sam_count=int(row["sam_after"]),
            )
    return records


def _audit_records(
    audit_csv_path: Path,
    audit_dir: Path,
) -> dict[str, dict[str, object]]:
    with audit_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records: dict[str, dict[str, object]] = {}
    for row in rows:
        filename = row.get("filename", "").strip()
        human_count = row.get("human_count", "").strip()
        artifact_value = row.get("visible_artifact", "").strip().lower()
        if not filename or not human_count:
            raise ValueError(f"Audit row is missing filename or human count: {row}")
        if artifact_value in {"yes", "y"}:
            has_artifact = True
        elif artifact_value in {"no", "n"}:
            has_artifact = False
        else:
            raise ValueError(
                f"Unrecognized visible_artifact value {artifact_value!r} "
                f"for {filename}"
            )
        image_path = (audit_dir / filename).resolve()
        if not image_path.is_relative_to(audit_dir.resolve()):
            raise ValueError(f"Audit filename escapes the audit directory: {filename}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Audited image is missing: {image_path}")
        image_hash = _sha256(image_path)
        if image_hash in records:
            raise ValueError(f"Duplicate audited image content: {filename}")
        records[image_hash] = {
            "human_count": int(human_count),
            "visible_artifact": has_artifact,
        }
    return records


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("A confidence interval requires a non-zero denominator.")
    if not 0 <= successes <= total:
        raise ValueError("Successes must be between zero and total.")
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5)
        / denominator
    )
    return (max(0.0, center - margin) * 100, min(1.0, center + margin) * 100)


def analyze(
    results_path: Path = RESULTS_PATH,
    audit_csv_path: Path = AUDIT_CSV_PATH,
    audit_dir: Path = AUDIT_DIR,
) -> dict[str, object]:
    source_records = _source_records(results_path)
    audited_records = _audit_records(audit_csv_path, audit_dir)
    if source_records.keys() != audited_records.keys():
        missing_audits = source_records.keys() - audited_records.keys()
        missing_sources = audited_records.keys() - source_records.keys()
        raise ValueError(
            "Source and audit image hashes do not match: "
            f"{len(missing_audits)} source images lack human verdicts, "
            f"{len(missing_sources)} audited images have no source artifact."
        )

    merged = []
    for image_hash, source in source_records.items():
        human = audited_records[image_hash]
        merged.append({**source, **human})

    sam_passes = [row for row in merged if row["sam_count"] == row["expected_count"]]
    sam_fails = [row for row in merged if row["sam_count"] != row["expected_count"]]
    false_passes = sum(
        row["human_count"] != row["expected_count"] for row in sam_passes
    )
    false_fails = sum(
        row["human_count"] == row["expected_count"] for row in sam_fails
    )

    starts = [row for row in merged if row["image_type"] == "start"]
    edits = [row for row in merged if row["image_type"] == "edit"]

    def artifact_summary(rows: list[dict[str, object]]) -> dict[str, int | float]:
        if not rows:
            raise ValueError("Cannot calculate artifact rate for an empty group.")
        artifacts = sum(bool(row["visible_artifact"]) for row in rows)
        return {
            "artifacts": artifacts,
            "total": len(rows),
            "percent": 100 * artifacts / len(rows),
        }

    artifacts_by_condition = {
        label: artifact_summary(
            [row for row in edits if row["condition"] == condition]
        )
        for condition, label in CONDITION_LABELS.items()
    }
    numeric = [row for row in edits if row["condition"] == "numeric_edit"]
    clean_numeric_successes = sum(
        row["human_count"] == row["expected_count"]
        and not row["visible_artifact"]
        for row in numeric
    )
    return {
        "image_count": len(merged),
        "sam_false_passes": {
            "count": false_passes,
            "total": len(sam_passes),
            "ci": wilson_interval(false_passes, len(sam_passes)),
        },
        "sam_false_fails": {
            "count": false_fails,
            "total": len(sam_fails),
            "ci": wilson_interval(false_fails, len(sam_fails)),
        },
        "starts_artifacts": artifact_summary(starts),
        "edits_artifacts": artifact_summary(edits),
        "artifacts_by_condition": artifacts_by_condition,
        "clean_numeric_success": {
            "count": clean_numeric_successes,
            "total": len(numeric),
            "percent": 100 * clean_numeric_successes / len(numeric),
        },
    }


def _print_rate(name: str, values: dict[str, object]) -> None:
    low, high = values["ci"]
    count, total = values["count"], values["total"]
    percent = 100 * count / total
    print(
        f"- {name}: {count}/{total} = {percent:.1f}% "
        f"(95% Wilson CI: {low:.1f}%–{high:.1f}%; denominator: {total} "
        "SAM predictions in this class)"
    )


def _print_artifact(name: str, values: dict[str, int | float]) -> None:
    print(
        f"- {name}: {values['artifacts']}/{values['total']} "
        f"= {values['percent']:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join blinded human audit counts with SAM experiment results."
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV_PATH)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    args = parser.parse_args()

    summary = analyze(args.results, args.audit_csv, args.audit_dir)
    print(f"Audited unique images: {summary['image_count']}")
    print("\nSAM vs. human confusion matrix (count equality as PASS):")
    _print_rate("SAM false passes", summary["sam_false_passes"])
    _print_rate("SAM false fails", summary["sam_false_fails"])

    print("\nVisible artifact rate (human audit):")
    _print_artifact("Starting images", summary["starts_artifacts"])
    _print_artifact("Edited images", summary["edits_artifacts"])
    for label, values in summary["artifacts_by_condition"].items():
        _print_artifact(label, values)

    clean = summary["clean_numeric_success"]
    print(
        "\nAccepted Clean Image (Numeric outcomes): "
        f"{clean['count']}/{clean['total']} = {clean['percent']:.1f}% "
        "(human count equals expected and no visible artifact)"
    )


if __name__ == "__main__":
    main()
