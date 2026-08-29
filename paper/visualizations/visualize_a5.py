"""Render A5 as three categorical winner maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Polygon, Rectangle

from paper.visualizations.common import (
    COLOR_PAPER_CYAN_STRONG,
    COLOR_PAPER_DARK_CYAN,
    COLOR_PAPER_ZX_BLUE,
    COLOR_PAPER_LILA,
    COLOR_PAPER_DARK_PINK,
    COLOR_PAPER_DARK_RED,
    COLOR_PAPER_GRAY_VERY_DARK,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_LIGHT_RED,
    COLOR_PAPER_GREEN_DEEP,
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
    ("sat", "SAT", COLOR_PAPER_DARK_CYAN),
    ("bruteforce", "Brute force", COLOR_PAPER_DARK_RED),
    ("classical", "Classical Approaches", COLOR_PAPER_DARK_PINK),
    ("graph_iso", "Graph Isomorphism", COLOR_PAPER_GREEN_DEEP),
    ("matroid", "Matroid Isomorphism", COLOR_PAPER_LIGHT_RED),
    ("kls", "KLS Orbit", COLOR_PAPER_ZX_BLUE),
    ("lse", "Graph-State LSE", COLOR_PAPER_LILA),
)


def method(algorithm: str) -> tuple[str, str]:
    for suffix, label, color in METHODS:
        if algorithm.endswith(f"_{suffix}"):
            return label, color
    return algorithm, COLOR_PAPER_GRAY_VERY_VERY_DARK


def overlay_runner_up(
    ax,
    n: int,
    r: int,
    color: str,
) -> None:
    """Paint the runner-up into the lower triangle of an already-filled cell.

    The split runs bottom-left to top-right, so the winner keeps the whole top
    edge and the runner-up the whole bottom edge; which method leads stays
    readable at a glance without a second visual channel.
    """
    x, y = n - 0.5, r - 0.5
    ax.add_patch(
        Polygon(
            [(x, y), (x + 1, y), (x + 1, y + 1)],
            closed=True,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )


class _SplitCellHandler(HandlerBase):
    """Legend key drawn as the same diagonally split cell."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x, y = -xdescent, -ydescent
        top = Polygon([(x, y), (x, y + height), (x + width, y + height)], closed=True,
                      facecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK, edgecolor="none", transform=trans)
        bottom = Polygon([(x, y), (x + width, y), (x + width, y + height)], closed=True,
                         facecolor=COLOR_PAPER_GRAY_VERY_DARK, edgecolor="none", transform=trans)
        return [top, bottom]


class _SplitCellKey:
    """Marker object selecting :class:`_SplitCellHandler` in the legend."""

    def get_label(self) -> str:
        return "top/bottom: within 5%"


def legend(rows: list[dict[str, str]]) -> list[Patch]:
    algorithms = {row["winner"] for row in rows}
    algorithms.update(row["runner_up"] for row in rows if row["runner_up"])
    present = {method(algorithm)[0] for algorithm in algorithms}
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in METHODS
        if label in present
    ]
    handles.append(_SplitCellKey())
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
                overlay_runner_up(ax, n, r, runner_color)

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
        handler_map={_SplitCellKey: _SplitCellHandler()},
    )
    return save_png(figure, output)


if __name__ == "__main__":
    render()
