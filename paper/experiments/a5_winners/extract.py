"""Find the fastest fully successful algorithm in every parameter cell."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from paper.experiments.common import ALGORITHM_DATA_DIR, RESULTS_DIR, aggregate_statistics, load_all_algorithms, write_csv

OUTPUT_DIRECTORY = RESULTS_DIR / "winners"
METHOD_FIELDS = (
    "problem", "algorithm", "n", "k", "r", "num_requested", "num_successful",
    "mean_seconds", "stddev_seconds", "maximum_seconds", "num_unexpected",
    "num_timeouts", "num_memory_limited", "num_errors", "num_generation_errors",
    "complete",
)
WINNER_FIELDS = (
    "problem", "n", "k", "r", "winner", "mean_seconds", "runner_up",
    "runner_up_mean_seconds", "speed_ratio", "num_eligible_algorithms",
)


def extract(
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> list[dict[str, Any]]:
    methods = aggregate_statistics(load_all_algorithms(algorithm_directory))
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in methods:
        if cell["complete"] and cell["mean_seconds"] is not None:
            groups[(cell["problem"], cell["n"], cell["k"])].append(cell)
    winners = []
    for (problem, n, k), choices in sorted(groups.items()):
        ordered = sorted(choices, key=lambda cell: cell["mean_seconds"])
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        winners.append({
            "problem": problem, "n": n, "k": k, "r": n - k,
            "winner": winner["algorithm"], "mean_seconds": winner["mean_seconds"],
            "runner_up": runner_up["algorithm"] if runner_up else "",
            "runner_up_mean_seconds": runner_up["mean_seconds"] if runner_up else "",
            "speed_ratio": runner_up["mean_seconds"] / winner["mean_seconds"] if runner_up and winner["mean_seconds"] else "",
            "num_eligible_algorithms": len(ordered),
        })
    write_csv(output_directory / "by_method.csv", methods, METHOD_FIELDS)
    write_csv(output_directory / "by_cell.csv", winners, WINNER_FIELDS)
    return winners


if __name__ == "__main__":
    extract()
