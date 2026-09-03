"""Select and aggregate the three graph-representation algorithm files for A4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper.experiments.common import ALGORITHM_DATA_DIR, RESULTS_DIR, aggregate_statistics, load_algorithm, write_csv

ALGORITHMS = ("pm_stb_graph_iso", "pm_css_matroid", "lc_stb_graph_iso")
OUTPUT = RESULTS_DIR / "a4" / "by_cell.csv"
FIELDS = (
    "problem", "algorithm", "n", "k", "r", "num_requested", "num_successful",
    "mean_total_seconds", "stddev_total_seconds", "maximum_total_seconds",
    "num_timeouts", "num_memory_limited", "num_errors", "num_unexpected",
    "num_generation_errors",
)


def extract(
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    output_file: Path = OUTPUT,
) -> list[dict[str, Any]]:
    raw = [
        row
        for algorithm in ALGORITHMS
        for row in load_algorithm(algorithm, algorithm_directory)
    ]
    cells = aggregate_statistics(raw)
    output = [
        {
            **cell,
            "mean_total_seconds": cell["mean_seconds"],
            "stddev_total_seconds": cell["stddev_seconds"],
            "maximum_total_seconds": cell["maximum_seconds"],
        }
        for cell in cells
    ]
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
