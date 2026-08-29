"""Render A3 as three invariant-versus-backend cost-ratio maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper.visualizations.common import (
    COLOR_PAPER_DARK_BLUE,
    COLOR_PAPER_DARK_RED,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_WHITE,
    RELATIVE_CMAP,
    RESULTS_DIR,
    WIDE_TEXT_SCALE,
    half_cell_key,
    load_rows,
    number,
    outline_partition,
    parameter_axis,
    partition_cell,
    ratio_ticks,
    relative_norm,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a3" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a3" / "a3.png"

PANELS = (
    ("linear_dependency", "Linear Column Dependencies", ("pm_stb", "pm_css")),
    ("signatures", "Signatures", ("pm_stb", "pm_css")),
    ("local_invariant", "Local Invariant", ("lc_stb",)),
)


def draw_panel(ax, rows: Sequence[dict[str, str]], problems: Sequence[str], norm) -> int:
    """Paint one panel, splitting every cell across the problems it holds."""
    painted = 0
    for index, problem in enumerate(problems):
        for row in rows:
            if row["problem"] != problem:
                continue
            ratio = number(row["relative_runtime"])
            if ratio is None or ratio <= 0:
                continue
            n, r = int(row["n"]), int(row["r"])
            partition_cell(ax, n, r, index, len(problems), RELATIVE_CMAP(norm(ratio)))
            if row["backend_selection"] == "timeout_fallback":
                outline_partition(
                    ax, n, r, index, len(problems), COLOR_PAPER_GRAY_VERY_VERY_DARK
                )
            painted += 1
    return painted


def legend_handles() -> list[Line2D | Patch]:
    gray = COLOR_PAPER_GRAY_VERY_VERY_DARK
    return [
        half_cell_key("left", gray, "Stabilizer codes", size=8),
        half_cell_key("right", gray, "CSS codes", size=8),
        Patch(facecolor=COLOR_PAPER_DARK_BLUE, edgecolor="none", label="Invariant is cheaper"),
        Patch(facecolor=COLOR_PAPER_DARK_RED, edgecolor="none", label="Invariant costs more"),
        Patch(
            facecolor=COLOR_PAPER_WHITE,
            edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK,
            linewidth=0.9,
            label="Backend timed out",
        ),
    ]


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    required = (
        "problem", "invariant", "n", "r", "invariant_mean_seconds",
        "backend_algorithm", "backend_mean_seconds", "backend_selection",
        "relative_runtime",
    )
    rows = load_rows(input_file, required)
    norm = relative_norm()

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.22, top=0.80, wspace=0.18)

    for ax, (invariant, title, problems) in zip(axes, PANELS):
        parameter_axis(ax, title)
        panel_rows = [row for row in rows if row["invariant"] == invariant]
        if not draw_panel(ax, panel_rows, problems, norm):
            ax.text(
                0.5, 0.62, "not yet collected", transform=ax.transAxes,
                ha="center", va="center", fontsize=9 * WIDE_TEXT_SCALE,
                color=COLOR_PAPER_GRAY_VERY_VERY_DARK,
            )
            print(f"warning: no A3 rows for {invariant}", flush=True)

    figure.suptitle(
        "Invariant Cost Relative to the Best-Performing Backend",
        fontsize=12 * WIDE_TEXT_SCALE,
        y=0.96,
    )
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.015),
    )

    bar = figure.colorbar(
        scalar_mappable(RELATIVE_CMAP, norm), ax=axes, fraction=0.025, pad=0.02,
        extend="both",
    )
    ratio_ticks(bar)
    bar.set_label("$T_{\\mathrm{invariant}} \\,/\\, T_{\\mathrm{backend}}$")
    return save_png(figure, output)


if __name__ == "__main__":
    render()
