"""Render A7 as two decision-count tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from paper.experiments.extract_a7 import EXPERIMENT1, EXPERIMENT2
from paper.visualizations.common import (
    COLOR_PAPER_BLUE,
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_LIGHT,
    COLOR_PAPER_GRAY_MEDIUM,
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_LIGHT_BLUE,
    COLOR_PAPER_WHITE,
    RESULTS_DIR,
    load_rows,
    save_png,
    use_style,
)

INPUT = RESULTS_DIR / "a7" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a7" / "a7.png"
CONDITIONS = ("A", "B1", "B2", "C")

REQUIRED = (
    "experiment",
    "condition",
    "n",
    "r",
    "median_base_decisions",
    "median_invalid_mapping_decisions",
)


def _format_decisions(value: str | int | float | None) -> str:
    if value is None or str(value).strip() == "":
        return "—"
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}"


def _cell(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    color: str = "#202020",
    weight: str = "normal",
    fontsize: float = 8.5,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=COLOR_PAPER_WHITE,
            linewidth=1.0,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontweight=weight,
        fontsize=fontsize,
    )


def _finish_table(ax: Any, total_height: float) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            total_height,
            facecolor="none",
            edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK,
            linewidth=0.8,
        )
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(total_height, 0)
    ax.axis("off")


def _render_experiment1(ax: Any, rows: list[dict[str, str]]) -> None:
    cells = {
        (int(row["n"]), row["condition"]): row
        for row in rows
        if row["experiment"] == EXPERIMENT1
    }
    ns = sorted({n for n, _ in cells})
    if not ns:
        raise ValueError("A7 extraction contains no Experiment-1 rows")

    widths = (0.10, 0.10, 0.12, 0.28, 0.40)
    positions = [0.0]
    for width in widths:
        positions.append(positions[-1] + width)
    header_height = 0.9
    row_height = 0.72
    total_height = header_height + len(ns) * len(CONDITIONS) * row_height
    headers = ("n", "r", "Case", "Base decisions", "Invalid-mapping decisions")
    for index, header in enumerate(headers):
        _cell(
            ax,
            positions[index],
            0,
            widths[index],
            header_height,
            header,
            facecolor=COLOR_PAPER_GRAY_DARK,
            weight="bold",
            fontsize=8.2,
        )

    y = header_height
    for group, n in enumerate(ns):
        group_rows = [cells.get((n, condition)) for condition in CONDITIONS]
        if any(row is None for row in group_rows):
            raise ValueError(f"A7 Experiment 1 is incomplete for n={n}")
        base_color = (
            COLOR_PAPER_GRAY_VERY_LIGHT
            if group % 2 == 0
            else COLOR_PAPER_GRAY_LIGHT
        )
        group_height = len(CONDITIONS) * row_height
        first_row = group_rows[0]
        assert first_row is not None
        _cell(
            ax,
            positions[0],
            y,
            widths[0],
            group_height,
            str(n),
            facecolor=base_color,
        )
        _cell(
            ax,
            positions[1],
            y,
            widths[1],
            group_height,
            first_row["r"],
            facecolor=base_color,
        )
        for index, row in enumerate(group_rows):
            assert row is not None
            row_y = y + index * row_height
            row_color = COLOR_PAPER_LIGHT_BLUE if row["condition"] == "C" else base_color
            text_color = COLOR_PAPER_BLUE if row["condition"] == "C" else "#202020"
            weight = "bold" if row["condition"] == "C" else "normal"
            values = (
                row["condition"],
                _format_decisions(row["median_base_decisions"]),
                _format_decisions(row["median_invalid_mapping_decisions"]),
            )
            for column, value in enumerate(values, start=2):
                _cell(
                    ax,
                    positions[column],
                    row_y,
                    widths[column],
                    row_height,
                    value,
                    facecolor=row_color,
                    color=text_color,
                    weight=weight,
                )
        y += group_height
    _finish_table(ax, total_height)
    ax.set_title(
        "Experiment 1 — Survival of invalid permutation choices",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )


def _render_experiment2(ax: Any, rows: list[dict[str, str]]) -> None:
    cells = {
        (int(row["n"]), row["condition"]): row
        for row in rows
        if row["experiment"] == EXPERIMENT2
    }
    ns = sorted({n for n, _ in cells})
    if not ns:
        raise ValueError("A7 extraction contains no Experiment-2 rows")

    widths = (0.18, 0.18, 0.32, 0.32)
    positions = [0.0]
    for width in widths:
        positions.append(positions[-1] + width)
    header_height = 0.65
    row_height = 0.72
    total_height = 2 * header_height + len(ns) * row_height
    _cell(
        ax,
        positions[0],
        0,
        widths[0],
        2 * header_height,
        "n",
        facecolor=COLOR_PAPER_GRAY_DARK,
        weight="bold",
    )
    _cell(
        ax,
        positions[1],
        0,
        widths[1],
        2 * header_height,
        "r",
        facecolor=COLOR_PAPER_GRAY_DARK,
        weight="bold",
    )
    _cell(
        ax,
        positions[2],
        0,
        widths[2] + widths[3],
        header_height,
        "Median decisions",
        facecolor=COLOR_PAPER_GRAY_DARK,
        weight="bold",
    )
    for column, label in ((2, "Clean"), (3, "Mixed")):
        _cell(
            ax,
            positions[column],
            header_height,
            widths[column],
            header_height,
            label,
            facecolor=COLOR_PAPER_GRAY_MEDIUM,
            weight="bold",
        )

    y = 2 * header_height
    for index, n in enumerate(ns):
        clean = cells.get((n, "clean"))
        mixed = cells.get((n, "mixed"))
        if clean is None or mixed is None:
            raise ValueError(f"A7 Experiment 2 is incomplete for n={n}")
        color = (
            COLOR_PAPER_GRAY_VERY_LIGHT
            if index % 2 == 0
            else COLOR_PAPER_GRAY_LIGHT
        )
        values = (
            str(n),
            clean["r"],
            _format_decisions(clean["median_base_decisions"]),
            _format_decisions(mixed["median_base_decisions"]),
        )
        for column, value in enumerate(values):
            _cell(
                ax,
                positions[column],
                y,
                widths[column],
                row_height,
                value,
                facecolor=color,
            )
        y += row_height
    _finish_table(ax, total_height)
    ax.set_title(
        "Experiment 2 — Full row mixing of CSS tableaus",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows = load_rows(input_file, REQUIRED)
    use_style()
    figure = plt.figure(figsize=(12.4, 7.0))
    experiment1_ax = figure.add_axes([0.045, 0.31, 0.61, 0.56])
    experiment2_ax = figure.add_axes([0.70, 0.48, 0.27, 0.32])
    _render_experiment1(experiment1_ax, rows)
    _render_experiment2(experiment2_ax, rows)

    figure.suptitle(
        "Uncoupled and Independent CSS Structure INHERENTLY increases SAT Search",
        x=0.5,
        y=0.97,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.70,
        0.455,
        "A   General code, unrestricted R\n"
        "B1  Two actual row blocks, unrestricted R\n"
        "B2  Same B1 pair, exposed R1/R2; both blocks see X and Z\n"
        "C   Balanced CSS, independent Rx/Rz; each block sees only X or Z",
        ha="left",
        va="top",
        fontsize=6.4,
        linespacing=1.45,
        family="monospace",
    )
    return save_png(figure, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    render(arguments.input_file, arguments.output)
