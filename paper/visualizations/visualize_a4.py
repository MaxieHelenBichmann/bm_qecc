"""Render A4 as three full-grid graph-representation runtime maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper.visualizations.common import (
    RESULTS_DIR,
    RUNTIME_CMAP,
    aggregate_cells,
    decimal_ticks,
    failure_legend,
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

INPUT = RESULTS_DIR / "a4" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a4" / "a4.png"
PANELS = (
    ("pm_stb_graph_iso", "Permutation Equivalence\nfor Stabilizer Codes", 25),
    ("lc_stb_graph_iso", "Local-Clifford Equivalence\nfor Stabilizer Codes", 25),
    ("pm_css_matroid", "Permutation Equivalence\nfor CSS Codes (Matroid)", 35),
)


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    required = (
        "algorithm",
        "n",
        "r",
        "mean_total_seconds",
        "num_successful",
        "num_timeouts",
        "num_memory_limited",
        "num_errors",
        "num_unexpected",
    )
    rows = load_rows(input_file, required)
    for row in rows:
        row["num_runtime_samples"] = str(
            int(row["num_successful"])
            + int(row["num_timeouts"])
            + int(row["num_unexpected"])
        )
    aggregated = {
        algorithm: aggregate_cells(
            [
                row
                for row in rows
                if row["algorithm"] == algorithm and int(row["n"]) <= nmax
            ],
            "mean_total_seconds",
            "num_runtime_samples",
        )
        for algorithm, _, nmax in PANELS
    }
    norm = runtime_norm(
        float(cell["mean_value"])
        for cells in aggregated.values()
        for cell in cells.values()
        if int(cell["num_successful"])
    )

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.5))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.08, top=0.79, wspace=0.18)
    for ax, (algorithm, title, nmax) in zip(axes, PANELS):
        parameter_axis(ax, title, nmax=nmax)
        for (n, r), cell in aggregated[algorithm].items():
            if int(cell["num_successful"]):
                partition_cell(
                    ax,
                    n,
                    r,
                    0,
                    1,
                    RUNTIME_CMAP(norm(float(cell["mean_value"]))),
                )
            memory_cell = {
                **cell,
                "num_memory_limited": (
                    int(cell["num_memory_limited"]) + int(cell["num_errors"])
                ),
                "num_errors": 0,
            }
            failure_marks(ax, n, r, memory_cell)
    figure.suptitle(
        "Search Cost using Graph Representations",
        fontsize=12 * WIDE_TEXT_SCALE,
    )
    # The (n, r) triangle leaves the upper left of every panel empty, so the key
    # sits there next to the marks it explains instead of in its own strip.
    axes[0].legend(
        handles=failure_legend()[:1],
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=0.8,
        handletextpad=0.4,
    )
    bar = figure.colorbar(
        scalar_mappable(RUNTIME_CMAP, norm), ax=axes, fraction=0.018, pad=0.015
    )
    decimal_ticks(bar)
    mark_timeout(bar)
    bar.set_label("Mean runtime [s]")
    return save_png(figure, output)


if __name__ == "__main__":
    render()
