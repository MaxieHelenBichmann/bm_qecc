"""Render A2 as two full-grid signature-space maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from paper.visualizations.common import (
    RESULTS_DIR,
    SIGNATURE_CMAP,
    aggregate_cells,
    load_rows,
    parameter_axis,
    partition_cell,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a2" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a2" / "a2.png"


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    value_field = "mean_pairwise_refinement"
    required = ("problem", "n", "r", "num_valid", "num_censored", value_field)
    rows = load_rows(input_file, required)
    aggregated = {
        problem: aggregate_cells(
            [row for row in rows if row["problem"] == problem],
            value_field,
            "num_valid",
        )
        for problem in ("pm_stb", "pm_css")
    }
    censored = {
        problem: {
            (int(row["n"]), int(row["r"]))
            for row in rows
            if row["problem"] == problem and int(row["num_censored"])
        }
        for problem in ("pm_stb", "pm_css")
    }
    norm = Normalize(vmin=0.0, vmax=1.0)

    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.12, top=0.84, wspace=0.18)
    for ax, problem, title in zip(
        axes,
        ("pm_stb", "pm_css"),
        ("Stabilizer-Code Signatures", "CSS-Code Signatures"),
    ):
        parameter_axis(ax, title)
        for (n, r), cell in aggregated[problem].items():
            # A timeout can depend on the partition itself, so a successful-only
            # mean may be biased. Leave every censored cell uncolored.
            if int(cell["num_successful"]) and (n, r) not in censored[problem]:
                partition_cell(
                    ax,
                    n,
                    r,
                    0,
                    1,
                    SIGNATURE_CMAP(norm(float(cell["mean_value"]))),
                )
    figure.suptitle("Pairwise Refinement Induced by Permutation Signatures", fontsize=12)
    bar = figure.colorbar(
        scalar_mappable(SIGNATURE_CMAP, norm), ax=axes, fraction=0.025, pad=0.02
    )
    bar.set_label(
        "Fraction of qubit pairs distinguished by signature\n"
        "(0 = no refinement, 1 = complete refinement)"
    )
    return save_png(figure, output)


if __name__ == "__main__":
    render()
