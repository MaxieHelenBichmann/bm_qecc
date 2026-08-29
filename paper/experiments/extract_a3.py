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
from paper.experiments.extract_a5 import select_winners

INVARIANT_INPUT = COLLECTED_DATA_DIR / "invariant_timings.csv"
OUTPUT = RESULTS_DIR / "a3" / "by_cell.csv"
FIELDS = (
    "problem", "invariant", "n", "k", "r", "invariant_mean_seconds",
    "invariant_stddev_seconds", "backend_algorithm", "backend_mean_seconds",
    "backend_selection", "backend_num_timeouts", "relative_runtime",
    "num_invariant_requested", "num_invariant_successful",
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
    backend_statistics = aggregate_statistics(load_all_algorithms(algorithm_directory))
    # The comparison baseline is A5's winner, including its timeout fallback: a
    # backend that only finishes by timing out still bounds what the invariant
    # has to beat, and dropping those cells would silently hide the region
    # where invariants matter most.
    backends = {
        (winner["problem"], winner["n"], winner["k"]): winner
        for winner in select_winners(backend_statistics)
    }

    output = []
    for invariant_cell in invariant_cells:
        key = (
            invariant_cell["problem"],
            invariant_cell["n"],
            invariant_cell["k"],
        )
        backend = backends.get(key)
        if (
            backend is None
            or not backend["mean_seconds"]
            or invariant_cell["mean_seconds"] is None
        ):
            continue
        output.append({
            "problem": invariant_cell["problem"],
            "invariant": invariant_cell["invariant"],
            "n": invariant_cell["n"], "k": invariant_cell["k"],
            "r": invariant_cell["r"],
            "invariant_mean_seconds": invariant_cell["mean_seconds"],
            "invariant_stddev_seconds": invariant_cell["stddev_seconds"],
            "backend_algorithm": backend["winner"],
            "backend_mean_seconds": backend["mean_seconds"],
            "backend_selection": backend["selection"],
            "backend_num_timeouts": backend["winner_num_timeouts"],
            "relative_runtime": invariant_cell["mean_seconds"] / backend["mean_seconds"],
            "num_invariant_requested": invariant_cell["num_requested"],
            "num_invariant_successful": invariant_cell["num_successful"],
        })
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
