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
    EMPTY,
    RESULTS_DIR,
    load_plot_rows,
    mark_synthetic,
    parameter_axis,
    partition_cell,
    save_png,
    use_style,
)

INPUT = RESULTS_DIR / "a5" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a5" / "a5.png"
NEAR_TIE_RATIO = 1.05

PANELS = (
    ("pm_stb", "Permutation equivalence\nstabilizer codes"),
    ("pm_css", "Permutation equivalence\nCSS codes"),
    ("lc_stb", "Local-Clifford equivalence\nstabilizer codes"),
)

# The same algorithmic method has the same colour in every problem panel.
METHODS = (
    ("sat", "SAT", "#0072B2"),
    ("bruteforce", "Brute force", "#CC79A7"),
    ("classical", "Classical reduction", "#E69F00"),
    ("graph_iso", "Graph isomorphism", "#56B4E9"),
    ("matroid", "Matroid isomorphism", "#009E73"),
    ("kls", "KLS", "#7B61A8"),
    ("lse", "LSE / graph state", "#B07C00"),
)


def method(algorithm: str) -> tuple[str, str]:
    for suffix, label, color in METHODS:
        if algorithm.endswith(f"_{suffix}"):
            return label, color
    return algorithm, "#666666"


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
    handles.extend(
        [
            Patch(
                facecolor="#777777",
                edgecolor="#222222",
                hatch="///",
                label="runner-up within 5%",
            ),
            Patch(
                facecolor=EMPTY,
                edgecolor="#D0D0D0",
                label="no eligible winner / unmeasured",
            ),
        ]
    )
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
    rows, synthetic = load_plot_rows(input_file, required)

    use_style()
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
        "Preferred algorithm per parameter regime",
        fontsize=12,
        y=0.96,
    )
    figure.legend(
        handles=legend(rows),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    mark_synthetic(figure, synthetic)
    return save_png(figure, output)


if __name__ == "__main__":
    render()
