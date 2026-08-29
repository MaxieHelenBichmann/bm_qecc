"""Render A6 as three full-grid SAT runtime maps.

Run with no arguments::

    python3 -m paper.visualizations.visualize_a6

The panels show PM-STB SAT on stabilizer inputs, PM-CSS SAT on CSS inputs, and
PM-STB SAT on the same CSS population. Positive and negative cases are
aggregated into one runtime-observation-weighted mean per cell. Completed runs
and capped timeouts contribute to the color; memory and execution failures do
not. Resource failures remain explicit. Exactly one PNG is written.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper.visualizations.common import (
    RESULTS_DIR,
    RUNTIME_CMAP,
    aggregate_cells,
    decimal_ticks,
    failure_marks,
    load_rows,
    mark_timeout,
    parameter_axis,
    partition_cell,
    runtime_norm,
    save_png,
    scalar_mappable,
    WIDE_TEXT_SCALE,
    use_style,
)

INPUT = RESULTS_DIR / "a6" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a6" / "a6.png"
NMAX = 25
PANELS = (
    ("pm_stb_sat_on_stabilizer", "Tableau Encoding\non General Stabilizer Codes"),
    ("pm_css_sat_on_css", "Check-Matrix Encoding\non CSS Codes"),
    ("pm_stb_sat_on_css", "Tableau Encoding\non CSS Codes"),
)


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    required = (
        "variant",
        "n",
        "r",
        "mean_seconds",
        "num_successful",
        "num_timeouts",
        "num_memory_limited",
        "num_errors",
    )
    rows = load_rows(input_file, required)
    rows = [row for row in rows if int(row["n"]) <= NMAX]
    for row in rows:
        row["num_runtime_samples"] = str(
            int(row["num_successful"])
            + int(row["num_timeouts"])
            + int(row.get("num_unexpected", 0) or 0)
        )
    aggregated = {
        variant: aggregate_cells(
            [row for row in rows if row["variant"] == variant],
            "mean_seconds",
            "num_runtime_samples",
        )
        for variant, _ in PANELS
    }
    norm = runtime_norm(
        float(cell["mean_value"])
        for cells in aggregated.values()
        for cell in cells.values()
        if int(cell["num_successful"])
    )

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.5))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.08, top=0.82, wspace=0.18)
    for ax, (variant, title) in zip(axes, PANELS):
        parameter_axis(ax, title, nmax=NMAX)
        for (n, r), cell in aggregated[variant].items():
            if int(cell["num_successful"]):
                partition_cell(
                    ax,
                    n,
                    r,
                    0,
                    1,
                    RUNTIME_CMAP(norm(float(cell["mean_value"]))),
                )
            failure_marks(ax, n, r, cell)
    figure.suptitle("Performance of SAT Encodings on Stabilizer and CSS inputs", fontsize=12 * WIDE_TEXT_SCALE)
    bar = figure.colorbar(
        scalar_mappable(RUNTIME_CMAP, norm), ax=axes, fraction=0.018, pad=0.015
    )
    decimal_ticks(bar)
    mark_timeout(bar)
    bar.set_label("Mean runtime [s]")
    return save_png(figure, output)


if __name__ == "__main__":
    render()
