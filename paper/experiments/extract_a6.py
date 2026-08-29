"""Combine normal SAT files with the extra PM-STB-on-CSS collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper.experiments.common import ALGORITHM_DATA_DIR, COLLECTED_DATA_DIR, RESULTS_DIR, aggregate_statistics, load_algorithm, read_statistics, write_csv

EXTRA_INPUT = COLLECTED_DATA_DIR / "pm_stb_sat_on_css.csv"
OUTPUT = RESULTS_DIR / "a6" / "by_cell.csv"
VARIANTS = (
    ("pm_stb_sat_on_stabilizer", "pm_stb_sat", "stabilizer"),
    ("pm_css_sat_on_css", "pm_css_sat", "css"),
    ("pm_stb_sat_on_css", "pm_stb_sat_on_css", "css"),
)
FIELDS = (
    "variant", "algorithm", "code_family", "n", "k", "r", "num_requested",
    "num_successful", "mean_seconds", "stddev_seconds", "maximum_seconds",
    "num_timeouts", "num_memory_limited", "num_errors", "num_unexpected",
    "num_generation_errors",
)


def extract(
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    extra_input: Path = EXTRA_INPUT,
    output_file: Path = OUTPUT,
) -> list[dict[str, Any]]:
    sources = {
        "pm_stb_sat": load_algorithm("pm_stb_sat", algorithm_directory),
        "pm_css_sat": load_algorithm("pm_css_sat", algorithm_directory),
        "pm_stb_sat_on_css": read_statistics(extra_input),
    }
    output = []
    for variant, algorithm, family in VARIANTS:
        for cell in aggregate_statistics(sources[algorithm]):
            output.append({**cell, "variant": variant, "code_family": family})
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
