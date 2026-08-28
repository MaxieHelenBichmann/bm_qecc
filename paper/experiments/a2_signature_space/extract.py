"""Normalize and aggregate A2 signature metrics from typical random codes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "signature_space.csv"
OUTPUT = RESULTS_DIR / "signature_space" / "by_cell.csv"
FIELDS = (
    "problem", "n", "k", "r", "num_requested", "num_valid",
    "mean_distinct_pair_fraction", "stddev_distinct_pair_fraction",
    "num_censored",
)


def distinct_pair_fraction(q_pairs: float, n: int) -> float:
    """Normalize q_pairs from its n-dependent [1/n, 1] range to [0, 1]."""
    minimum = 1 / n
    tolerance = 1e-12
    if q_pairs < minimum - tolerance or q_pairs > 1 + tolerance:
        raise ValueError(
            f"q_pairs={q_pairs} is outside the theoretical [{minimum}, 1] "
            f"range for n={n}"
        )
    return min(1.0, max(0.0, (q_pairs - minimum) / (1 - minimum)))


def extract(input_file: Path = INPUT, output_file: Path = OUTPUT) -> list[dict[str, Any]]:
    rows = read_csv(input_file, ("problem", "n", "k", "q_pairs", "status"))
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["problem"], int(row["n"]), int(row["k"]))].append(row)
    cells = []
    for (problem, n, k), group in sorted(groups.items()):
        values = [
            distinct_pair_fraction(float(row["q_pairs"]), n)
            for row in group
            if row["status"] == "success"
        ]
        cells.append({
            "problem": problem, "n": n, "k": k, "r": n - k,
            "num_requested": len(group), "num_valid": len(values),
            "mean_distinct_pair_fraction": mean(values) if values else "",
            "stddev_distinct_pair_fraction": (
                stdev(values) if len(values) > 1 else (0.0 if values else "")
            ),
            "num_censored": len(group) - len(values),
        })
    write_csv(output_file, cells, FIELDS)
    return cells


if __name__ == "__main__":
    extract()
