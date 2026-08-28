"""Render A2 as two full-grid signature-space maps.

Run with no arguments::

    python3 -m paper.visualizations.plot_signature_space

Independently seeded random codes are aggregated into one mean per parameter
cell. Color shows the fraction of distinct ordered qubit pairs that remain in
the same signature class, normalized to one shared linear ``0..1`` scale.
Exactly one PNG is written.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from paper.visualizations.common import (
    RESULTS_DIR,
    SIGNATURE_CMAP,
    aggregate_cells,
    load_plot_rows,
    mark_synthetic,
    parameter_axis,
    partition_cell,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "signature_space" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "signature_space" / "signature_space.png"


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    value_field = "mean_distinct_pair_fraction"
    required = ("problem", "n", "r", "num_valid", value_field)
    rows, synthetic = load_plot_rows(input_file, required)
    aggregated = {
        problem: aggregate_cells(
            [row for row in rows if row["problem"] == problem],
            value_field,
            "num_valid",
        )
        for problem in ("pm_stb", "pm_css")
    }
    norm = Normalize(vmin=0.0, vmax=1.0)

    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.12, top=0.84, wspace=0.18)
    for ax, problem, title in zip(
        axes,
        ("pm_stb", "pm_css"),
        ("Stabilizer signatures", "CSS signatures"),
    ):
        parameter_axis(ax, title)
        for (n, r), cell in aggregated[problem].items():
            if int(cell["num_successful"]):
                partition_cell(
                    ax,
                    n,
                    r,
                    0,
                    1,
                    SIGNATURE_CMAP(norm(float(cell["mean_value"]))),
                )
    figure.suptitle("Typical random-code signature partitions", fontsize=12)
    bar = figure.colorbar(
        scalar_mappable(SIGNATURE_CMAP, norm), ax=axes, fraction=0.025, pad=0.02
    )
    bar.set_label(
        "mean fraction of distinct qubit pairs in the same signature class\n"
        "random-code aggregate (0 = complete, 1 = no refinement)"
    )
    mark_synthetic(figure, synthetic)
    return save_png(figure, output)


if __name__ == "__main__":
    render()
