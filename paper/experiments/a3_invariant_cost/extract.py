"""Pair invariant timings with the fastest valid backend in each cell."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from paper.experiments.common import (
    ALGORITHM_DATA_DIR,
    COLLECTED_DATA_DIR,
    RESULTS_DIR,
    aggregate_statistics,
    as_bool,
    as_float,
    load_all_algorithms,
    read_csv,
    write_csv,
)

INVARIANT_INPUT = COLLECTED_DATA_DIR / "invariant_timings.csv"
OUTPUT = RESULTS_DIR / "invariant_cost" / "by_cell.csv"
FIELDS = (
    "problem", "invariant", "n", "k", "r", "invariant_mean_seconds",
    "invariant_stddev_seconds", "backend_algorithm", "backend_mean_seconds",
    "relative_runtime", "num_invariant_requested", "num_invariant_successful",
)
EXPECTED_SEEDS_PER_POLARITY = 5


def read_invariant_cells(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(
        path,
        (
            "problem",
            "invariant",
            "seed",
            "n",
            "k",
            "positive",
            "runtime_seconds",
            "status",
        ),
    )
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in ("problem", "invariant", "n", "k", "positive", "seed")
        )
        latest[key] = row

    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in latest.values():
        grouped[
            (row["problem"], row["invariant"], int(row["n"]), int(row["k"]))
        ].append(row)

    cells = []
    for (problem, invariant, n, k), group in sorted(grouped.items()):
        positive = [row for row in group if as_bool(row["positive"])]
        negative = [row for row in group if not as_bool(row["positive"])]
        complete = (
            len(positive) == EXPECTED_SEEDS_PER_POLARITY
            and len(negative) == EXPECTED_SEEDS_PER_POLARITY
            and all(row["status"] == "success" for row in group)
        )
        if not complete:
            continue
        runtimes = [as_float(row["runtime_seconds"]) for row in group]
        if any(runtime is None for runtime in runtimes):
            continue
        values = [runtime for runtime in runtimes if runtime is not None]
        cells.append(
            {
                "problem": problem,
                "invariant": invariant,
                "n": n,
                "k": k,
                "r": n - k,
                "mean_seconds": mean(values),
                "stddev_seconds": stdev(values) if len(values) > 1 else 0.0,
                "num_requested": len(group),
                "num_successful": len(values),
            }
        )
    return cells


def extract(
    invariant_input: Path = INVARIANT_INPUT,
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    output_file: Path = OUTPUT,
) -> list[dict[str, Any]]:
    invariant_cells = read_invariant_cells(invariant_input)
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
            or invariant_cell["mean_seconds"] is None
        ):
            continue
        backend = min(choices, key=lambda cell: cell["mean_seconds"])
        problem = invariant_cell["problem"]
        output.append({
            "problem": problem,
            "invariant": invariant_cell["invariant"],
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
