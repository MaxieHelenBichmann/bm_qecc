"""Render A1 as two full-grid rejection maps.

Run with no arguments::

    python3 -m paper.visualizations.plot_invariant_rejection

The permutation panel combines the PM-STB and PM-CSS populations. A point is
split into invariant-colored sectors; an absent sector means that invariant
rejected none of the measured negatives. Color depth is the rejection fraction.
The LC panel shows the local invariant. Exactly one PNG is written next to the
input CSV.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

from paper.visualizations.common import RESULTS_DIR, load_plot_rows, mark_synthetic, parameter_axis, partition_cell, save_png, scalar_mappable, use_style

INPUT = RESULTS_DIR / "invariant_rejection" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "invariant_rejection" / "invariant_rejection.png"

CMAPS = {
    "linear_dependency": LinearSegmentedColormap.from_list("linear_dependency", ["#F2FAE9", "#98D256", "#57891F"]),
    "signatures": LinearSegmentedColormap.from_list("signatures", ["#FBF3FF", "#C79AE8", "#7F3FB8"]),
    "local_invariant": LinearSegmentedColormap.from_list("local_invariant", ["#FFF4DF", "#FFA404", "#B85B00"]),
}
LABELS = {"linear_dependency": "linear dependencies", "signatures": "signatures", "local_invariant": "local invariant"}


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


def _overall(values: dict[tuple[int, int, str], tuple[int, int]], invariant: str) -> float:
    selected = [value for (*_, name), value in values.items() if name == invariant]
    rejected = sum(value[0] for value in selected)
    valid = sum(value[1] for value in selected)
    return 100 * rejected / valid if valid else float("nan")


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows, synthetic = load_plot_rows(input_file, ("problem", "n", "r", "invariant", "num_valid", "num_rejected"))
    pm = _aggregate(rows, {"pm_stb", "pm_css"})
    lc = _aggregate(rows, {"lc_stb"})
    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.84, wspace=0.18)
    parameter_axis(axes[0], "Permutation equivalence")
    parameter_axis(axes[1], "Local-Clifford equivalence")
    _draw(axes[0], pm, ("linear_dependency", "signatures"))
    _draw(axes[1], lc, ("local_invariant",))

    handles = []
    for name, values in (("linear_dependency", pm), ("signatures", pm), ("local_invariant", lc)):
        handles.append(Patch(facecolor=CMAPS[name](0.72), label=f"{LABELS[name]} ({_overall(values, name):.1f}% overall)"))
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    figure.suptitle("Invariant rejection of certified inequivalent pairs", fontsize=12)
    # A small common depth scale is sufficient because every invariant uses the
    # same 0..100% normalization, even though their hue families differ.
    bar = figure.colorbar(scalar_mappable("Greys", Normalize(0, 1)), ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("color depth = rejected fraction")
    bar.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    mark_synthetic(figure, synthetic)
    return save_png(figure, output)


if __name__ == "__main__":
    render()
