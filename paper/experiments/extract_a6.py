"""Combine normal SAT files with the extra PM-STB-on-CSS collection.

For CSS cells measured by both encodings, the PM-STB row also records their
absolute distance on A6's shared logarithmic runtime scale::

    100 * abs(log(PM-CSS mean) - log(PM-STB mean)) / log(scale_max / scale_min)

The scale covers all displayed A6 cells and extends to the timeout cap, exactly
like the figure's colorbar. The visualization averages these per-cell
percentages to summarize how different its two CSS panels look.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from paper.experiments.common import ALGORITHM_DATA_DIR, COLLECTED_DATA_DIR, RESULTS_DIR, aggregate_statistics, load_algorithm, read_statistics, write_csv

EXTRA_INPUT = COLLECTED_DATA_DIR / "pm_stb_sat_on_css.csv"
OUTPUT = RESULTS_DIR / "a6" / "by_cell.csv"
NMAX = 25
TIMEOUT_SECONDS = 5_400.0
VARIANTS = (
    ("pm_stb_sat_on_stabilizer", "pm_stb_sat", "stabilizer"),
    ("pm_css_sat_on_css", "pm_css_sat", "css"),
    ("pm_stb_sat_on_css", "pm_stb_sat_on_css", "css"),
)
FIELDS = (
    "variant", "algorithm", "code_family", "n", "k", "r", "num_requested",
    "num_successful", "mean_seconds", "stddev_seconds", "maximum_seconds",
    "pm_stb_log_scale_difference_percentage",
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
    aggregated = {
        algorithm: aggregate_statistics(rows)
        for algorithm, rows in sources.items()
    }
    output = []
    for variant, algorithm, family in VARIANTS:
        for cell in aggregated[algorithm]:
            output.append(
                {
                    **cell,
                    "variant": variant,
                    "code_family": family,
                    "pm_stb_log_scale_difference_percentage": "",
                }
            )

    displayed_means = [
        float(row["mean_seconds"])
        for row in output
        if row["n"] <= NMAX
        and row["mean_seconds"] is not None
        and row["mean_seconds"] > 0
        and row["num_successful"]
    ]
    scale_min = min(displayed_means)
    scale_max = max(max(displayed_means), TIMEOUT_SECONDS)
    log_span = math.log(scale_max / scale_min)
    css_means = {
        (row["n"], row["k"]): float(row["mean_seconds"])
        for row in output
        if row["variant"] == "pm_css_sat_on_css"
        and row["mean_seconds"] is not None
        and row["mean_seconds"] > 0
    }
    for row in output:
        if row["variant"] != "pm_stb_sat_on_css":
            continue
        css_mean = css_means.get((row["n"], row["k"]))
        stb_mean = row["mean_seconds"]
        if css_mean is not None and stb_mean is not None and stb_mean > 0:
            row["pm_stb_log_scale_difference_percentage"] = (
                100.0 * abs(math.log(css_mean) - math.log(stb_mean)) / log_span
            )
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
