"""Render A1 as two full-grid rejection maps."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper.visualizations.common import (
    COLOR_PAPER_BLUE,
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GREEN_RAMP,
    COLOR_PAPER_ORANGE_RAMP,
    COLOR_PAPER_PINK_RAMP,
    COLOR_PAPER_WHITE,
    RESULTS_DIR,
    half_cell_key,
    load_rows,
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


def _draw(ax, values: dict[tuple[int, int, str], tuple[int, int]], invariants: Sequence[str]) -> None:
    grouped: dict[tuple[int, int], dict[str, tuple[int, int]]] = defaultdict(dict)
    for (n, r, invariant), value in values.items():
        grouped[(n, r)][invariant] = value
    for (n, r), cell in grouped.items():
        visible = [name for name in invariants if cell.get(name, (0, 0))[0] > 0]
        for index, name in enumerate(visible):
            rejected, valid = cell[name]
            fraction = rejected / valid if valid else 0
            partition_cell(ax, n, r, index, len(visible), CMAPS[name](fraction))


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


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _render_overall_table(pm_stb, pm_css, lc, output: Path) -> Path:
    """Render overall rejection rates, keeping the two permutation families separate."""
    rows = (
        (
            "General stabilizer codes",
            _overall(pm_stb, "linear_dependency"),
            _overall(pm_stb, "signatures"),
            None,
        ),
        (
            "CSS codes",
            _overall(pm_css, "linear_dependency"),
            _overall(pm_css, "signatures"),
            None,
        ),
        (
            "General stabilizer codes (LC)",
            None,
            None,
            _overall(lc, "local_invariant"),
        ),
    )
    figure, ax = plt.subplots(figsize=(7.2, 1.75))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.79)
    ax.axis("off")
    table = ax.table(
        cellText=[[label, *(_format_rate(value) for value in values)] for label, *values in rows],
        colLabels=("Code family", LABELS["linear_dependency"], LABELS["signatures"], LABELS["local_invariant"]),
        cellLoc="center",
        colLoc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    for column in range(4):
        header = table[(0, column)]
        header.set_facecolor(COLOR_PAPER_GRAY_DARK)
        header.set_edgecolor(COLOR_PAPER_WHITE)
        header.get_text().set_color("#202020")
        header.get_text().set_fontsize(8)
        header.get_text().set_fontweight("bold")

    for row in range(1, len(rows) + 1):
        for column in range(4):
            cell = table[(row, column)]
            cell.set_facecolor(COLOR_PAPER_GRAY_VERY_LIGHT)
            cell.set_edgecolor(COLOR_PAPER_WHITE)
            cell.get_text().set_fontsize(8.5 if column == 0 else 10)
            if column > 0 and cell.get_text().get_text() != "—":
                cell.get_text().set_color(COLOR_PAPER_BLUE)
                cell.get_text().set_fontweight("bold")
    figure.suptitle("Overall rejection rates", fontsize=11, fontweight="bold", y=0.96)
    return save_png(figure, output)


def render(input_file: Path = INPUT, output: Path = OUTPUT, overall_output: Path | None = None) -> Path:
    rows = load_rows(input_file, ("problem", "n", "r", "invariant", "num_valid", "num_rejected"))
    pm_stb = _aggregate(rows, {"pm_stb"})
    pm_css = _aggregate(rows, {"pm_css"})
    pm = _aggregate(rows, {"pm_stb", "pm_css"})
    lc = _aggregate(rows, {"lc_stb"})
    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.84, wspace=0.18)
    parameter_axis(axes[0], "Permutation Equivalence")
    parameter_axis(axes[1], "Local Clifford Equivalence")
    _draw(axes[0], pm, ("linear_dependency", "signatures"))
    _draw(axes[1], lc, ("local_invariant",))

    # The permutation panel splits every cell between its two invariants, so
    # those keys are drawn as the matching half-cells; the local invariant owns
    # its cell outright and keeps a full square.
    handles: list[Patch | Line2D] = [
        half_cell_key("left", CMAPS["linear_dependency"](0.72), _label("linear_dependency", pm)),
        half_cell_key("right", CMAPS["signatures"](0.72), _label("signatures", pm)),
        Patch(facecolor=CMAPS["local_invariant"](0.72), label=_label("local_invariant", lc)),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)
    figure.suptitle("Invariants' Rejection Patterns and Rates of Inequivalent Codes", fontsize=12)

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
