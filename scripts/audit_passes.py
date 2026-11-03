from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def _read_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_loop_results(runs_dir: Path) -> Path:
    candidates = []
    for results_path in runs_dir.glob("eval*/results.csv"):
        rows = _read_rows(results_path)
        if not rows or any(row.get("mode") != "loop" for row in rows):
            continue
        clean_rows = [row for row in rows if not row.get("error")]
        candidates.append((len(clean_rows), results_path.stat().st_mtime, results_path))
    if not candidates:
        raise FileNotFoundError(f"No loop evaluation results found under {runs_dir}")
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def _find_passed_run(
    runs_dir: Path,
    brief: str,
) -> tuple[Path, dict]:
    matches = []
    for summary_path in runs_dir.glob("*/summary.json"):
        run_dir = summary_path.parent
        if run_dir.name.startswith("eval") or run_dir.name in {"demos", "audit"}:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("brief") == brief and summary.get("final_passed") is True:
            matches.append((summary_path.stat().st_mtime, run_dir, summary))
    if not matches:
        raise FileNotFoundError(f"No successful saved loop run found for brief: {brief}")
    _, run_dir, summary = max(matches, key=lambda match: match[0])
    return run_dir, summary


def export_passes(results_path: Path, runs_dir: Path, audit_dir: Path) -> int:
    rows = _read_rows(results_path)
    if not rows or any(row.get("mode") != "loop" for row in rows):
        raise ValueError(f"{results_path} is not a loop evaluation results file")

    briefs_by_id = {}
    default_briefs = Path("src/eval/briefs.jsonl")
    if default_briefs.is_file():
        with default_briefs.open(encoding="utf-8") as handle:
            briefs_by_id = {
                item["id"]: item["brief"]
                for line in handle
                if line.strip()
                for item in [json.loads(line)]
            }

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    passed_rows = [
        row for row in rows
        if not row.get("error") and row.get("passed", "").lower() == "true"
    ]
    random.SystemRandom().shuffle(passed_rows)
    mapping = {}
    for image_index, row in enumerate(passed_rows, start=1):
        brief = row.get("brief") or briefs_by_id.get(row["brief_id"])
        if not brief:
            raise ValueError(
                f"Brief text is unavailable for passing result {row['brief_id']}"
            )
        run_dir, summary = _find_passed_run(runs_dir, brief)
        round_index = int(summary["rounds"])
        image_path = run_dir / f"round_{round_index}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Final round image is missing: {image_path}")
        filename = f"img_{image_index:03d}.png"
        shutil.copyfile(image_path, audit_dir / filename)
        verdict_path = run_dir / f"round_{round_index}_verdicts.json"
        if not verdict_path.is_file():
            raise FileNotFoundError(f"Final round verdicts are missing: {verdict_path}")
        sam_verdicts = json.loads(verdict_path.read_text(encoding="utf-8"))
        mapping[filename] = {
            "brief_id": row["brief_id"],
            "mode": row["mode"],
            "sam_verdict": {
                "passed": True,
                "constraints": sam_verdicts,
            },
        }

        constraints = summary.get("constraints", [])
        if not constraints:
            audit_rows.append({
                "filename": filename,
                "brief": brief,
                "constraint_type": "",
                "expected_count": "",
                "human_verdict": "",
            })
        for constraint in constraints:
            expected_count = constraint.get("expected_count")
            audit_rows.append({
                "filename": filename,
                "brief": brief,
                "constraint_type": constraint.get("type", ""),
                "expected_count": "" if expected_count is None else expected_count,
                "human_verdict": "",
            })

    mapping_path = audit_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    csv_path = audit_dir / "ground_truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "brief",
                "constraint_type",
                "expected_count",
                "human_verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    return len(passed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export final images from a loop evaluation for human audit."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Evaluation directory containing results.csv (defaults to latest loop run).",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    results_path = (
        args.run_dir / "results.csv"
        if args.run_dir
        else _latest_loop_results(args.runs_dir)
    )
    count = export_passes(results_path, args.runs_dir, args.runs_dir / "audit")
    print(f"Audited {count} passing prompts from {results_path}")
    print(f"Human review sheet: {args.runs_dir / 'audit' / 'ground_truth.csv'}")
    print(f"Private filename mapping: {args.runs_dir / 'audit' / 'mapping.json'}")


if __name__ == "__main__":
    main()
