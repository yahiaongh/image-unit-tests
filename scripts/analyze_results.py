from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

EXPECTED_N = 19
T_CRITICAL_95_DF18 = 2.100922
CONDITIONS = (
    ("numeric_edit", "Numeric"),
    ("generic_edit", "Generic"),
    ("marked_edit", "Marked"),
    ("control_edit", "Control"),
)


def load_overcount_rows(path: Path) -> dict[str, list[dict[str, int]]]:
    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            failure_type = row.get("failure_type", "").strip().lower()
            if failure_type not in {"over", "overcount"}:
                continue
            expected = int(row["expected_count"])
            before = int(row["sam_before"])
            after = int(row["sam_after"])
            before_error = abs(before - expected)
            if before_error == 0:
                continue
            grouped[row["condition"].strip().lower()].append({
                "before_error": before_error,
                "after_error": abs(after - expected),
            })
    return grouped


def summarize(
    grouped: dict[str, list[dict[str, int]]],
) -> list[dict[str, str | int]]:
    summaries = []
    for condition, label in CONDITIONS:
        rows = grouped.get(condition, [])
        if len(rows) != EXPECTED_N:
            raise ValueError(
                f"{label} has N={len(rows)} valid overcount rows; expected "
                f"N={EXPECTED_N} after excluding non-failures."
            )
        reductions = [
            row["before_error"] - row["after_error"]
            for row in rows
        ]
        mean = sum(reductions) / EXPECTED_N
        variance = sum((value - mean) ** 2 for value in reductions) / (EXPECTED_N - 1)
        margin = T_CRITICAL_95_DF18 * math.sqrt(variance / EXPECTED_N)
        perfect = sum(row["after_error"] == 0 for row in rows)
        improved_nonperfect = sum(
            row["after_error"] < row["before_error"]
            and row["after_error"] != 0
            for row in rows
        )
        unchanged = sum(row["after_error"] == row["before_error"] for row in rows)
        worse = sum(row["after_error"] > row["before_error"] for row in rows)
        summaries.append({
            "condition": label,
            "mean": f"{mean:+.2f}",
            "ci": f"[{mean - margin:+.2f}, {mean + margin:+.2f}]",
            "perfect": f"{100 * perfect / EXPECTED_N:.1f}%",
            "improved": f"{100 * improved_nonperfect / EXPECTED_N:.1f}%",
            "unchanged": f"{100 * unchanged / EXPECTED_N:.1f}%",
            "worse": f"{100 * worse / EXPECTED_N:.1f}%",
        })
    return summaries


def print_markdown(summaries: list[dict[str, str | int]]) -> None:
    print(
        "| Condition | Mean Error Reduction | 95% CI | Perfect Fix | "
        "Improved | Unchanged | Worse |"
    )
    print(
        "|-----------|----------------------|--------|-------------|"
        "----------|-----------|-------|"
    )
    for row in summaries:
        print(
            f"| {row['condition']} | {row['mean']} | {row['ci']} | "
            f"{row['perfect']} | {row['improved']} | "
            f"{row['unchanged']} | {row['worse']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the overcount-only single-step experiment."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("runs/single_step/results.csv"),
    )
    args = parser.parse_args()
    print_markdown(summarize(load_overcount_rows(args.results)))


if __name__ == "__main__":
    main()
