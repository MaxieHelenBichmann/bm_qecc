"""Pair invariant timings with the fastest valid backend in each cell."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from paper.experiments.common import (
    ALGORITHM_DATA_DIR,
    COLLECTED_DATA_DIR,
    RESULTS_DIR,
    aggregate_statistics,
    load_all_algorithms,
    read_statistics,
    write_csv,
)

INVARIANT_INPUT = COLLECTED_DATA_DIR / "invariant_timings.csv"
OUTPUT = RESULTS_DIR / "invariant_cost" / "by_cell.csv"
FIELDS = (
    "problem", "invariant", "n", "k", "r", "invariant_mean_seconds",
    "invariant_stddev_seconds", "backend_algorithm", "backend_mean_seconds",
    "relative_runtime", "num_invariant_requested", "num_invariant_successful",
)


def extract(
    invariant_input: Path = INVARIANT_INPUT,
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    output_file: Path = OUTPUT,
) -> list[dict[str, Any]]:
    invariant_cells = aggregate_statistics(read_statistics(invariant_input))
    backend_cells = aggregate_statistics(load_all_algorithms(algorithm_directory))
    candidates: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in backend_cells:
        if cell["complete"] and cell["mean_seconds"] is not None:
            candidates[(cell["problem"], cell["n"], cell["k"])].append(cell)

    output = []
    for invariant_cell in invariant_cells:
        key = (
            invariant_cell["problem"],
            invariant_cell["n"],
            invariant_cell["k"],
        )
        choices = candidates.get(key, [])
        if (
            not choices
            or not invariant_cell["complete"]
            or invariant_cell["mean_seconds"] is None
        ):
            continue
        backend = min(choices, key=lambda cell: cell["mean_seconds"])
        problem = invariant_cell["problem"]
        output.append({
            "problem": problem,
            "invariant": invariant_cell["algorithm"].removeprefix(f"{problem}_"),
            "n": invariant_cell["n"], "k": invariant_cell["k"],
            "r": invariant_cell["r"],
            "invariant_mean_seconds": invariant_cell["mean_seconds"],
            "invariant_stddev_seconds": invariant_cell["stddev_seconds"],
            "backend_algorithm": backend["algorithm"],
            "backend_mean_seconds": backend["mean_seconds"],
            "relative_runtime": invariant_cell["mean_seconds"] / backend["mean_seconds"] if backend["mean_seconds"] else "",
            "num_invariant_requested": invariant_cell["num_requested"],
            "num_invariant_successful": invariant_cell["num_successful"],
        })
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
