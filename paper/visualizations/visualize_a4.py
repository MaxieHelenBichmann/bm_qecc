"""Render A4 as two full-grid graph-isomorphism runtime maps.

Run with no arguments::

    python3 -m paper.visualizations.visualize_a4

Positive and negative cases are aggregated into one runtime-observation-
weighted mean per parameter cell. Completed runs and capped timeouts contribute
to the color; memory failures do not. In these legacy graph-isomorphism
measurements, both memory-limit and general error counts denote memory failures
and are shown as blue crosses. Exactly one PNG is written.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper.visualizations.common import (
    RESULTS_DIR,
    RUNTIME_CMAP,
    aggregate_cells,
    failure_legend,
    failure_marks,
    load_plot_rows,
    mark_synthetic,
    parameter_axis,
    partition_cell,
    runtime_norm,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a4" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a4" / "a4.png"
NMAX = 25
PANELS = (
    ("pm_stb_graph_iso", "Permutation graph isomorphism"),
    ("lc_stb_graph_iso", "Local-Clifford graph isomorphism"),
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
    rows, synthetic = load_plot_rows(input_file, required)
    rows = [row for row in rows if int(row["n"]) <= NMAX]
    for row in rows:
        row["num_runtime_samples"] = str(
            int(row["num_successful"])
            + int(row["num_timeouts"])
            + int(row["num_unexpected"])
        )
    aggregated = {
        algorithm: aggregate_cells(
            [row for row in rows if row["algorithm"] == algorithm],
            "mean_total_seconds",
            "num_runtime_samples",
        )
        for algorithm, _ in PANELS
    }
    norm = runtime_norm(
        float(cell["mean_value"])
        for cells in aggregated.values()
        for cell in cells.values()
        if int(cell["num_successful"])
    )

    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.84, wspace=0.18)
    for ax, (algorithm, title) in zip(axes, PANELS):
        parameter_axis(ax, title, nmax=NMAX)
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
    figure.suptitle("Graph representation and search cost", fontsize=12)
    figure.legend(
        handles=failure_legend()[:1],
        loc="lower center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    bar = figure.colorbar(
        scalar_mappable(RUNTIME_CMAP, norm), ax=axes, fraction=0.025, pad=0.02
    )
    bar.set_label("mean runtime [s], positive + negative aggregate, logarithmic")
    mark_synthetic(figure, synthetic)
    return save_png(figure, output)


if __name__ == "__main__":
    render()
