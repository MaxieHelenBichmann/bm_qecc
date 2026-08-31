"""Render A8 as a runtime-colored hybrid attribution table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from paper.experiments.extract_a8 import ALGORITHMS, CODE_ORDER
from paper.visualizations.common import (
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_LIGHT,
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_WHITE,
    RESULTS_DIR,
    RUNTIME_CMAP,
    TIMEOUT_SECONDS,
    decimal_ticks,
    load_rows,
    mark_timeout,
    runtime_norm,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a8" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a8" / "a8.png"

ALGORITHM_LABELS = {
    "pm_stb_hybrid": "PM-STB",
    "pm_css_hybrid": "PM-CSS",
    "lc_stb_hybrid": "LC-STB",
}
COMPONENT_LEGEND = (
    ("CI", "cheap invariants"),
    ("EI", "expensive invariants"),
    ("S", "signatures"),
    ("BF", "brute force"),
    ("MI", "matroid isomorphism"),
    ("GI", "graph isomorphism"),
    ("SAT", "SAT solver"),
    ("LSE", "graph-state LSE"),
)

REQUIRED = (
    "algorithm",
    "code",
    "code_label",
    "positive",
    "mean_seconds",
    "primary_decider",
    "num_timeouts",
    "num_memory_limited",
    "num_errors",
)


def _boolean(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _text_color(color: Any) -> str:
    """Use white lettering only on the darkest runtime cells."""
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return COLOR_PAPER_WHITE if luminance < 0.46 else "#202020"


def render() -> Path:
    rows = load_rows(INPUT, REQUIRED)
    rows = [row for row in rows if row["algorithm"] in ALGORITHMS]
    if not rows:
        raise ValueError(f"{INPUT} contains no A8 hybrid rows")

    cells = {
        (row["code"], row["algorithm"], _boolean(row["positive"])): row
        for row in rows
    }
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(row["code"], row["code_label"])
    codes = sorted(
        labels,
        key=lambda code: (CODE_ORDER.get(code, len(CODE_ORDER)), code),
    )
    runtimes = [
        float(row["mean_seconds"])
        for row in rows
        if row["mean_seconds"].strip() and float(row["mean_seconds"]) > 0
    ]
    timeout_values = [
        float(row["timeout_seconds"])
        for row in rows
        if row.get("timeout_seconds", "").strip()
    ]
    timeout = max(timeout_values, default=TIMEOUT_SECONDS)
    norm = runtime_norm(runtimes, timeout=timeout)

    use_style()
    figure_height = max(6.2, 0.36 * len(codes) + 1.8)
    figure = plt.figure(figsize=(10.6, figure_height))
    ax = figure.add_axes([0.035, 0.055, 0.76, 0.875])

    code_width = 3.0
    cell_width = 1.0
    header_rows = 2
    total_width = code_width + 2 * len(ALGORITHMS) * cell_width
    total_height = header_rows + len(codes)

    # Two-level header: problem families above their +/- columns.
    ax.add_patch(
        Rectangle(
            (0, 0), code_width, header_rows,
            facecolor=COLOR_PAPER_GRAY_DARK,
            edgecolor=COLOR_PAPER_WHITE,
            linewidth=0.8,
        )
    )
    ax.text(0.10, 1.0, "Code", ha="left", va="center", fontsize=9, fontweight="bold")
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        x = code_width + 2 * algorithm_index * cell_width
        ax.add_patch(
            Rectangle(
                (x, 0), 2 * cell_width, 1,
                facecolor=COLOR_PAPER_GRAY_DARK,
                edgecolor=COLOR_PAPER_WHITE,
                linewidth=0.8,
            )
        )
        ax.text(
            x + cell_width,
            0.5,
            ALGORITHM_LABELS[algorithm],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        for polarity_index, symbol in enumerate(("+", "−")):
            sub_x = x + polarity_index * cell_width
            ax.add_patch(
                Rectangle(
                    (sub_x, 1), cell_width, 1,
                    facecolor=COLOR_PAPER_GRAY_LIGHT,
                    edgecolor=COLOR_PAPER_WHITE,
                    linewidth=0.8,
                )
            )
            ax.text(sub_x + 0.5, 1.5, symbol, ha="center", va="center", fontsize=10)

    for row_index, code in enumerate(codes):
        y = header_rows + row_index
        label_color = COLOR_PAPER_GRAY_VERY_LIGHT if row_index % 2 == 0 else COLOR_PAPER_GRAY_LIGHT
        ax.add_patch(
            Rectangle(
                (0, y), code_width, 1,
                facecolor=label_color,
                edgecolor=COLOR_PAPER_WHITE,
                linewidth=0.65,
            )
        )
        ax.text(0.10, y + 0.5, labels[code], ha="left", va="center", fontsize=8)

        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for polarity_index, positive in enumerate((True, False)):
                x = code_width + (2 * algorithm_index + polarity_index) * cell_width
                cell = cells.get((code, algorithm, positive))
                runtime = (
                    float(cell["mean_seconds"])
                    if cell is not None and cell["mean_seconds"].strip()
                    else None
                )
                color = (
                    RUNTIME_CMAP(norm(runtime))
                    if runtime is not None and runtime > 0
                    else COLOR_PAPER_GRAY_VERY_LIGHT
                )
                ax.add_patch(
                    Rectangle(
                        (x, y), cell_width, 1,
                        facecolor=color,
                        edgecolor=COLOR_PAPER_WHITE,
                        linewidth=0.65,
                    )
                )
                text = cell["primary_decider"] if cell is not None else "−"
                if not text:
                    text = "-"
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold" if text != "-" else "normal",
                    color=_text_color(color),
                )

    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.set_aspect("auto")
    ax.axis("off")

    figure.suptitle(
        "Hybrid Runtime and Main Deciding Stage on Structured Codes",
        x=0.415,
        y=0.975,
        fontsize=12,
    )

    colorbar_ax = figure.add_axes([0.84, 0.70, 0.022, 0.18])
    bar = figure.colorbar(scalar_mappable(RUNTIME_CMAP, norm), cax=colorbar_ax)
    decimal_ticks(bar)
    mark_timeout(bar, timeout)
    bar.set_label("Mean runtime [s]", labelpad=8)

    legend_ax = figure.add_axes([0.82, 0.12, 0.17, 0.50])
    legend_ax.axis("off")
    legend_ax.text(0, 1.0, "Decision stage", ha="left", va="top", fontsize=9, fontweight="bold")
    legend_lines = [f"{tag:<4} {label}" for tag, label in COMPONENT_LEGEND]
    legend_lines.extend(("", "-    unavailable", "+    equivalent", "-    inequivalent"))
    legend_ax.text(
        0,
        0.95,
        "\n".join(legend_lines),
        ha="left",
        va="top",
        fontsize=7.5,
        linespacing=1.45,
        family="monospace",
        color="#202020",
    )

    ax.add_patch(
        Rectangle(
            (0, 0), total_width, total_height,
            facecolor="none",
            edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK,
            linewidth=0.7,
        )
    )
    return save_png(figure, OUTPUT)


if __name__ == "__main__":
    render()
