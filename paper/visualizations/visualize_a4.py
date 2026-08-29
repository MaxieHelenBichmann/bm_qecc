"""Render A4 as two full-grid graph-isomorphism runtime maps."""

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
    use_style,
)

INPUT = RESULTS_DIR / "a4" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a4" / "a4.png"
NMAX = 25
PANELS = (
    ("pm_stb_graph_iso", "Permutation Equivalence"),
    ("lc_stb_graph_iso", "Local-Clifford Equivalence"),
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
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.08, top=0.84, wspace=0.18)
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
    figure.suptitle("Search Cost using Graph Representations", fontsize=12)
    # The (n, r) triangle leaves the upper left of every panel empty, so the key
    # sits there next to the marks it explains instead of in its own strip.
    axes[0].legend(handles=failure_legend()[:1], loc="upper left", frameon=False, fontsize=10,
        handlelength=0.8, handletextpad=0.4)
    bar = figure.colorbar(
        scalar_mappable(RUNTIME_CMAP, norm), ax=axes, fraction=0.025, pad=0.02
    )
    decimal_ticks(bar)
    mark_timeout(bar)
    bar.set_label("Mean runtime [s]")
    return save_png(figure, output)


if __name__ == "__main__":
    render()
