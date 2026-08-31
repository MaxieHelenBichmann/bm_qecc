"""Aggregate raw A7 SAT encoding and decision measurements."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from paper.experiments.common import (
    COLLECTED_DATA_DIR,
    RESULTS_DIR,
    as_bool,
    as_float,
    as_int,
    read_csv,
    write_csv,
)

INPUT = COLLECTED_DATA_DIR / "a7_sat_css_structure.csv"
OUTPUT = RESULTS_DIR / "a7" / "by_cell.csv"

EXPERIMENT1 = "permutation_survival"
EXPERIMENT2 = "row_mixing"
CONDITION_ORDER = ("A", "B1", "B2", "C", "clean", "mixed")

REQUIRED = (
    "experiment",
    "condition",
    "sample",
    "seed",
    "n",
    "k",
    "r",
    "rx",
    "rz",
    "measurement",
    "probe",
    "result",
    "timed_out",
    "solve_seconds",
    "decisions",
)
FIELDS = (
    "experiment",
    "condition",
    "n",
    "k",
    "r",
    "rx",
    "rz",
    "base_runs",
    "base_completed",
    "base_timeouts",
    "median_base_decisions",
    "median_base_seconds",
    "invalid_mapping_attempts",
    "invalid_mappings_proven",
    "invalid_mapping_automorphisms",
    "invalid_mapping_timeouts",
    "median_invalid_mapping_decisions",
    "median_invalid_mapping_seconds",
)


def _median(values: Sequence[int | float]) -> int | float | str:
    return median(values) if values else ""


def _deduplicate(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "experiment",
                "condition",
                "sample",
                "seed",
                "n",
                "k",
                "measurement",
                "probe",
            )
        )
        latest[key] = row
    return list(latest.values())


def extract(
    input_file: Path = INPUT,
    output_file: Path = OUTPUT,
) -> list[dict[str, Any]]:
    rows = _deduplicate(read_csv(input_file, REQUIRED))
    cells = sorted(
        {
            (
                row["experiment"],
                row["condition"],
                as_int(row["n"]),
                as_int(row["k"]),
            )
            for row in rows
        },
        key=lambda cell: (
            cell[0],
            cell[2],
            CONDITION_ORDER.index(cell[1]),
        ),
    )
    output: list[dict[str, Any]] = []
    for experiment, condition, n, k in cells:
        selected = [
            row
            for row in rows
            if row["experiment"] == experiment
            and row["condition"] == condition
            and as_int(row["n"]) == n
            and as_int(row["k"]) == k
        ]
        base = [row for row in selected if row["measurement"] == "base"]
        completed_base = [row for row in base if row["result"] == "sat"]
        mappings = [
            row for row in selected if row["measurement"] == "invalid_mapping"
        ]
        proven = [row for row in mappings if row["result"] == "unsat"]
        r = n - k
        rx = r // 2
        output.append(
            {
                "experiment": experiment,
                "condition": condition,
                "n": n,
                "k": k,
                "r": r,
                "rx": rx,
                "rz": r - rx,
                "base_runs": len(base),
                "base_completed": len(completed_base),
                "base_timeouts": sum(as_bool(row["timed_out"]) for row in base),
                "median_base_decisions": _median(
                    [as_int(row["decisions"]) for row in completed_base]
                ),
                "median_base_seconds": _median(
                    [as_float(row["solve_seconds"]) or 0.0 for row in completed_base]
                ),
                "invalid_mapping_attempts": len(mappings),
                "invalid_mappings_proven": len(proven),
                "invalid_mapping_automorphisms": sum(
                    row["result"] == "sat" for row in mappings
                ),
                "invalid_mapping_timeouts": sum(
                    as_bool(row["timed_out"]) for row in mappings
                ),
                "median_invalid_mapping_decisions": _median(
                    [as_int(row["decisions"]) for row in proven]
                ),
                "median_invalid_mapping_seconds": _median(
                    [as_float(row["solve_seconds"]) or 0.0 for row in proven]
                ),
            }
        )
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=INPUT)
    parser.add_argument("--output-file", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    extract(arguments.input_file, arguments.output_file)
