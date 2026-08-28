"""Aggregate A2 positive and negative raw signature metrics per cell."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "signature_space.csv"
OUTPUT = RESULTS_DIR / "signature_space" / "by_cell.csv"
FIELDS = ("problem", "n", "k", "r", "num_requested", "num_valid", "mean_q_pairs", "stddev_q_pairs", "num_censored")


def extract(input_file: Path = INPUT, output_file: Path = OUTPUT) -> list[dict[str, Any]]:
    rows = read_csv(input_file, ("problem", "n", "k", "q_pairs", "status"))
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["problem"], int(row["n"]), int(row["k"]))].append(row)
    cells = []
    for (problem, n, k), group in sorted(groups.items()):
        values = [float(row["q_pairs"]) for row in group if row["status"] == "success"]
        cells.append({
            "problem": problem, "n": n, "k": k, "r": n - k,
            "num_requested": len(group), "num_valid": len(values),
            "mean_q_pairs": mean(values) if values else "",
            "stddev_q_pairs": stdev(values) if len(values) > 1 else (0.0 if values else ""),
            "num_censored": len(group) - len(values),
        })
    write_csv(output_file, cells, FIELDS)
    return cells


if __name__ == "__main__":
    extract()
