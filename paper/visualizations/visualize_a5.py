"""Render A5 as three categorical winner maps.

Run with no arguments::

    python3 -m paper.visualizations.visualize_a5

Each method keeps the same colour in every panel. A forward-slash hatch in the
runner-up's colour marks methods whose runtimes differ by at most five percent.
Exactly one PNG is written.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from paper.visualizations.common import (
    COLOR_PAPER_BLUE,
    COLOR_PAPER_CYAN,
    COLOR_PAPER_DARK_GREEN,
    COLOR_PAPER_DARK_LILA,
    COLOR_PAPER_DARK_PINK,
    COLOR_PAPER_DARK_RED,
    COLOR_PAPER_GRAY_VERY_DARK,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_LIGHT_CYAN,
    COLOR_PAPER_SALMON,
    COLOR_PAPER_ZX_GREEN_DARK,
    COLOR_PAPER_DARK_GREEN,
    COLOR_PAPER_GREEN,
    COLOR_PAPER_ZX_ORANGE,
    EMPTY,
    RESULTS_DIR,
    load_rows,
    parameter_axis,
    partition_cell,
    save_png,
    WIDE_TEXT_SCALE,
    use_style,
)

INPUT = RESULTS_DIR / "a5" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a5" / "a5.png"
NEAR_TIE_RATIO = 1.05

PANELS = (
    ("pm_stb", "Permutation Equivalence\nfor Stabilizer Codes"),
    ("pm_css", "Permutation Equivalence\nfor CSS Codes"),
    ("lc_stb", "Local Clifford Equivalence\nfor Stabilizer Codes"),
)

METHODS = (
    ("sat", "SAT", COLOR_PAPER_CYAN),
    ("bruteforce", "Brute force", COLOR_PAPER_ZX_ORANGE),
    ("classical", "Classical Approaches", COLOR_PAPER_DARK_LILA),
    ("graph_iso", "Graph Isomorphism", COLOR_PAPER_GREEN),
    ("matroid", "Matroid Isomorphism", COLOR_PAPER_SALMON),
    ("kls", "KLS Orbit", COLOR_PAPER_DARK_PINK),
    ("lse", "Graph-State LSE", COLOR_PAPER_DARK_RED),
)


def method(algorithm: str) -> tuple[str, str]:
    for suffix, label, color in METHODS:
        if algorithm.endswith(f"_{suffix}"):
            return label, color
    return algorithm, COLOR_PAPER_GRAY_VERY_VERY_DARK


def overlay_hatch(
    ax,
    n: int,
    r: int,
    hatch: str,
    color: str,
) -> None:
    ax.add_patch(
        Rectangle(
            (n - 0.5, r - 0.5),
            1,
            1,
            facecolor="none",
            edgecolor=color,
            linewidth=0,
            hatch=hatch,
            zorder=3,
        )
    )


def legend(rows: list[dict[str, str]]) -> list[Patch]:
    algorithms = {row["winner"] for row in rows}
    algorithms.update(row["runner_up"] for row in rows if row["runner_up"])
    present = {method(algorithm)[0] for algorithm in algorithms}
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in METHODS
        if label in present
    ]
    return handles


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    required = (
        "problem",
        "n",
        "r",
        "winner",
        "runner_up",
        "speed_ratio",
        "num_eligible_algorithms",
        "selection",
    )
    rows = load_rows(input_file, required)

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.80, wspace=0.18)

    for ax, (problem, title) in zip(axes, PANELS):
        parameter_axis(ax, title)
        for row in rows:
            if row["problem"] != problem:
                continue
            n, r = int(row["n"]), int(row["r"])
            _, winner_color = method(row["winner"])
            partition_cell(ax, n, r, 0, 1, winner_color)

            ratio = float(row["speed_ratio"]) if row["speed_ratio"] else None
            if (
                row["selection"] == "completed"
                and ratio is not None
                and ratio <= NEAR_TIE_RATIO
                and row["runner_up"]
            ):
                _, runner_color = method(row["runner_up"])
                overlay_hatch(ax, n, r, "///", runner_color)

    figure.suptitle(
        "Best-Performing Prototype per Parameter Setting",
        fontsize=12 * WIDE_TEXT_SCALE,
        y=0.96,
    )
    figure.legend(
        handles=legend(rows),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.025),
    )
    return save_png(figure, output)


if __name__ == "__main__":
    render()
