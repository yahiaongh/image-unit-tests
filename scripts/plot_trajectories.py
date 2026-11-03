from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_trajectories(
    trajectory_path: Path,
    output_path: Path,
) -> None:
    grouped: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition"]
            if condition not in {"Numeric Loop", "Generic Loop"}:
                continue
            start_id = row["start_id"]
            round_index = int(row["round"])
            error = abs(int(row["sam_count"]) - int(row["expected_count"]))
            grouped[condition][start_id].append((round_index, error))

    if not grouped:
        raise ValueError(f"No Numeric Loop or Generic Loop rows in {trajectory_path}")
    expected_starts = set(grouped["Numeric Loop"]) | set(grouped["Generic Loop"])

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, condition in zip(axes, ("Numeric Loop", "Generic Loop")):
        starts = grouped.get(condition, {})
        for start_id in sorted(starts):
            points = sorted(starts[start_id])
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                alpha=0.55,
                linewidth=1,
                label=start_id,
            )
        axis.set_title(f"{condition} (N={len(starts)})")
        axis.set_xlabel("Round")
        axis.set_xlim(0, 4)
        axis.set_xticks(range(5))
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Absolute Error (|SAM count - expected count|)")
    if len(expected_starts) > 20:
        axes[1].legend([], [], frameon=False)
    else:
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="x-small")
    figure.suptitle("Count-Error Trajectories by Feedback Condition")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-round SAM count trajectories.")
    parser.add_argument("--trajectory", type=Path, default=Path("runs/multi_round/trajectory.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/multi_round/spaghetti_chart.png"))
    args = parser.parse_args()
    plot_trajectories(args.trajectory, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
