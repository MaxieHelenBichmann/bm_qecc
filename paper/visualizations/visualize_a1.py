"""Render A1 as family-split rejection maps and an overall-rate table."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from paper.visualizations.common import (
    COLOR_PAPER_BLUE,
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_LIGHT,
    COLOR_PAPER_GRAY_VERY_DARK,
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GREEN_RAMP,
    COLOR_PAPER_ORANGE_RAMP,
    COLOR_PAPER_PINK_RAMP,
    COLOR_PAPER_WHITE,
    RESULTS_DIR,
    WIDE_TEXT_SCALE,
    half_cell_key,
    load_rows,
    outline_partition,
    parameter_axis,
    partition_cell,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a1" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a1" / "a1.png"
OVERALL_OUTPUT = RESULTS_DIR / "a1" / "a1_overall.png"

CMAPS = {
    "linear_dependency": LinearSegmentedColormap.from_list("linear_dependency", COLOR_PAPER_GREEN_RAMP),
    "signatures": LinearSegmentedColormap.from_list("signatures", COLOR_PAPER_PINK_RAMP),
    "local_invariant": LinearSegmentedColormap.from_list("local_invariant", COLOR_PAPER_ORANGE_RAMP),
}
LABELS = {"linear_dependency": "Linear column dependencies", "signatures": "Signatures", "local_invariant": "Local invariant"}


def _aggregate(rows: Sequence[dict[str, str]], problems: set[str]) -> dict[tuple[int, int, str], tuple[int, int]]:
    values: dict[tuple[int, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["problem"] not in problems or row["invariant"] == "combined":
            continue
        key = (int(row["n"]), int(row["r"]), row["invariant"])
        values[key][0] += int(row["num_rejected"])
        values[key][1] += int(row["num_valid"])
    return {key: (value[0], value[1]) for key, value in values.items()}


def _draw(
    ax,
    families: Sequence[dict[tuple[int, int, str], tuple[int, int]]],
    invariant: str,
) -> None:
    """Draw one invariant, splitting cells consistently by code family."""
    for index, values in enumerate(families):
        for (n, r, name), (rejected, valid) in values.items():
            if name != invariant or valid <= 0:
                continue
            partition_cell(
                ax, n, r, index, len(families), CMAPS[invariant](rejected / valid)
            )
            if rejected == 0:
                outline_partition(
                    ax, n, r, index, len(families), COLOR_PAPER_GRAY_VERY_DARK
                )


def _overall(values: dict[tuple[int, int, str], tuple[int, int]], invariant: str) -> float | None:
    selected = [value for (*_, name), value in values.items() if name == invariant]
    rejected = sum(value[0] for value in selected)
    valid = sum(value[1] for value in selected)
    return 100 * rejected / valid if valid else None


def _label(invariant: str, values: dict[tuple[int, int, str], tuple[int, int]]) -> str:
    overall = _overall(values, invariant)
    if overall is None:
        return f"{LABELS[invariant]} (not yet collected)"
    return f"{LABELS[invariant]} ({overall:.1f}% overall)"


def _family_label(
    name: str,
    values: dict[tuple[int, int, str], tuple[int, int]],
) -> str:
    linear = _format_rate(_overall(values, "linear_dependency"))
    signatures = _format_rate(_overall(values, "signatures"))
    return f"{name}: {linear} linear, {signatures} signatures"


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _render_overall_table(pm_stb, pm_css, lc, output: Path) -> Path:
    """Render overall rejection rates, keeping the two permutation families separate."""
    values = (
        _overall(pm_stb, "linear_dependency"),
        _overall(pm_css, "linear_dependency"),
        _overall(pm_stb, "signatures"),
        _overall(pm_css, "signatures"),
        _overall(lc, "local_invariant"),
    )
    subheaders = (
        "General stabilizer",
        "CSS",
        "General stabilizer",
        "CSS",
        "General stabilizer (LC)",
    )
    widths = (0.2, 0.2, 0.2, 0.2, 0.2)
    body_height = 0.67

    figure, ax = plt.subplots(figsize=(7.2, 1.55))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.77)
    ax.axis("off")
    table = ax.table(
        cellText=[[_format_rate(value) for value in values]],
        colLabels=subheaders,
        colWidths=widths,
        cellLoc="center",
        colLoc="center",
        bbox=(0, 0, 1, body_height),
    )
    table.auto_set_font_size(False)
    for column in range(5):
        header = table[(0, column)]
        header.set_facecolor(COLOR_PAPER_GRAY_LIGHT)
        header.set_edgecolor(COLOR_PAPER_WHITE)
        header.get_text().set_color("#202020")
        header.get_text().set_fontsize(7.5)
        header.get_text().set_fontweight("bold")

    for column in range(5):
        cell = table[(1, column)]
        cell.set_facecolor(COLOR_PAPER_GRAY_VERY_LIGHT)
        cell.set_edgecolor(COLOR_PAPER_WHITE)
        cell.get_text().set_fontsize(10)
        if cell.get_text().get_text() != "—":
            cell.get_text().set_color(COLOR_PAPER_BLUE)
            cell.get_text().set_fontweight("bold")

    major_headers = (
        (0.0, 0.4, LABELS["linear_dependency"]),
        (0.4, 0.4, LABELS["signatures"]),
        (0.8, 0.2, LABELS["local_invariant"]),
    )
    for x, width, label in major_headers:
        ax.add_patch(
            Rectangle(
                (x, body_height),
                width,
                1 - body_height,
                transform=ax.transAxes,
                facecolor=COLOR_PAPER_GRAY_DARK,
                edgecolor=COLOR_PAPER_WHITE,
                linewidth=1.0,
            )
        )
        ax.text(
            x + width / 2,
            body_height + (1 - body_height) / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#202020",
            fontsize=8,
            fontweight="bold",
        )
    figure.suptitle("Overall rejection rates", fontsize=11, fontweight="bold", y=0.96)
    return save_png(figure, output)


def render(input_file: Path = INPUT, output: Path = OUTPUT, overall_output: Path | None = None) -> Path:
    rows = load_rows(input_file, ("problem", "n", "r", "invariant", "num_valid", "num_rejected"))
    pm_stb = _aggregate(rows, {"pm_stb"})
    pm_css = _aggregate(rows, {"pm_css"})
    lc = _aggregate(rows, {"lc_stb"})
    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.22, top=0.80, wspace=0.18)
    parameter_axis(axes[0], LABELS["linear_dependency"])
    parameter_axis(axes[1], LABELS["signatures"])
    parameter_axis(axes[2], "Local Clifford Equivalence")
    _draw(axes[0], (pm_stb, pm_css), "linear_dependency")
    _draw(axes[1], (pm_stb, pm_css), "signatures")
    _draw(axes[2], (lc,), "local_invariant")

    handles: list[Patch | Line2D] = [
        half_cell_key("left", COLOR_PAPER_GRAY_DARK, _family_label("Stabilizer", pm_stb)),
        half_cell_key("right", COLOR_PAPER_GRAY_DARK, _family_label("CSS", pm_css)),
        Patch(facecolor=CMAPS["local_invariant"](0.72), label=_label("local_invariant", lc)),
        Patch(
            facecolor=CMAPS["linear_dependency"](0),
            edgecolor=COLOR_PAPER_GRAY_VERY_DARK,
            label="0% rejected (measured)",
        ),
        Patch(facecolor=COLOR_PAPER_GRAY_VERY_LIGHT, edgecolor="none", label="not measured"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)
    figure.suptitle(
        "Invariants' Rejection Patterns and Rates of Inequivalent Codes",
        fontsize=12 * WIDE_TEXT_SCALE,
        y=0.96,
    )

    bar = figure.colorbar(scalar_mappable("Greys", Normalize(0, 1)), ax=axes, fraction=0.025, pad=0.02)
    bar.set_label(
        "Deeper color means\n"
        "more rejected instances"
    )
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    main_output = save_png(figure, output)
    if overall_output is None:
        overall_output = (
            OVERALL_OUTPUT
            if output == OUTPUT
            else output.with_name(f"{output.stem}_overall{output.suffix}")
        )
    _render_overall_table(pm_stb, pm_css, lc, overall_output)
    return main_output


if __name__ == "__main__":
    render()
