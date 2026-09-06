"""Render A8 with explicit coverage, certification, and failure annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from paper.experiments.extract_a8 import ALGORITHMS, CODE_ORDER, POPULATIONS
from paper.visualizations.common import (
    COLOR_PAPER_GRAY_DARK, COLOR_PAPER_GRAY_LIGHT, COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK, COLOR_PAPER_WHITE, RESULTS_DIR, RUNTIME_CMAP,
    TIMEOUT_SECONDS, decimal_ticks, load_rows, mark_timeout, runtime_norm, save_png,
    scalar_mappable, use_style,
)

INPUT = RESULTS_DIR / "a8" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a8" / "a8.png"

ALGORITHM_LABELS = {
    "pm_stb_hybrid": "PM-STB", "pm_css_hybrid": "PM-CSS", "lc_stb_hybrid": "LC-STB",
}
POPULATION_LABELS = {
    "positive_control": "control",
    "certified_negative": "negative",
}
COMPONENT_LEGEND = (
    ("CI", "cheap invariants"), ("EI", "expensive invariants"),
    ("S", "signatures"), ("BF", "brute force"), ("MI", "matroid isomorphism"),
    ("GI", "graph isomorphism"), ("SAT", "SAT solver"), ("LSE", "graph-state LSE"),
)
REQUIRED = (
    "algorithm", "code", "code_label", "population", "applicable",
    "restricted_mean_seconds", "primary_decider", "num_requested",
    "num_certified_equivalent", "num_unresolved_labels", "num_generation_failures",
    "num_certification_failures", "num_timeouts", "num_memory_limited", "num_errors",
    "num_incorrect", "num_blocked", "execution_timeout_seconds",
)


def _boolean(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _integer(row: dict[str, str], field: str) -> int:
    return int(row.get(field, "") or 0)


def _text_color(color: Any) -> str:
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return COLOR_PAPER_WHITE if luminance < 0.46 else "#202020"


def _cell_note(row: dict[str, str]) -> str:
    notes: list[str] = []
    equivalent_proposals = _integer(row, "num_certified_equivalent")
    unresolved = _integer(row, "num_unresolved_labels")
    timeouts = _integer(row, "num_timeouts")
    failures = sum(_integer(row, field) for field in (
        "num_generation_failures", "num_certification_failures", "num_memory_limited",
        "num_errors", "num_incorrect", "num_blocked",
    ))
    if equivalent_proposals:
        notes.append(f"e{equivalent_proposals}")
    if unresolved:
        notes.append(f"u{unresolved}")
    if timeouts:
        notes.append(f"t{timeouts}")
    if failures:
        notes.append(f"f{failures}")
    requested = _integer(row, "num_requested")
    if requested:
        notes.append(f"/{requested}")
    return " ".join(notes)


def render(input_file: Path = INPUT, output_file: Path = OUTPUT) -> Path:
    rows = [row for row in load_rows(input_file, REQUIRED) if row["algorithm"] in ALGORITHMS]
    if not rows:
        raise ValueError(f"{input_file} contains no A8 hybrid rows")
    cells = {(row["code"], row["algorithm"], row["population"]): row for row in rows}
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(row["code"], row["code_label"])
    codes = sorted(labels, key=lambda code: (CODE_ORDER.get(code, len(CODE_ORDER)), code))
    runtimes = [float(row["restricted_mean_seconds"]) for row in rows
                if row["restricted_mean_seconds"].strip() and float(row["restricted_mean_seconds"]) > 0]
    timeout = max((float(row["execution_timeout_seconds"]) for row in rows
                   if row["execution_timeout_seconds"].strip()), default=TIMEOUT_SECONDS)
    norm = runtime_norm(runtimes, timeout=timeout)

    use_style()
    figure_height = max(6.4, 0.39 * len(codes) + 1.9)
    figure = plt.figure(figsize=(11.4, figure_height))
    ax = figure.add_axes([0.035, 0.055, 0.78, 0.88])
    code_width, cell_width, header_rows = 3.0, 1.08, 2
    total_width = code_width + len(ALGORITHMS) * len(POPULATIONS) * cell_width
    total_height = header_rows + len(codes)

    ax.add_patch(Rectangle((0, 0), code_width, header_rows,
                           facecolor=COLOR_PAPER_GRAY_DARK, edgecolor=COLOR_PAPER_WHITE,
                           linewidth=0.8))
    ax.text(0.10, 1.0, "Code", ha="left", va="center", fontsize=9, fontweight="bold")
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        x = code_width + algorithm_index * len(POPULATIONS) * cell_width
        ax.add_patch(Rectangle((x, 0), len(POPULATIONS) * cell_width, 1,
                               facecolor=COLOR_PAPER_GRAY_DARK, edgecolor=COLOR_PAPER_WHITE,
                               linewidth=0.8))
        ax.text(x + cell_width, 0.5, ALGORITHM_LABELS[algorithm], ha="center",
                va="center", fontsize=8.5, fontweight="bold")
        for pop_index, population in enumerate(POPULATIONS):
            px = x + pop_index * cell_width
            ax.add_patch(Rectangle((px, 1), cell_width, 1,
                                   facecolor=COLOR_PAPER_GRAY_LIGHT,
                                   edgecolor=COLOR_PAPER_WHITE, linewidth=0.8))
            ax.text(px + cell_width / 2, 1.5, POPULATION_LABELS[population],
                    ha="center", va="center", fontsize=7.2)

    for row_index, code in enumerate(codes):
        y = header_rows + row_index
        ax.add_patch(Rectangle((0, y), code_width, 1,
                               facecolor=COLOR_PAPER_GRAY_VERY_LIGHT,
                               edgecolor=COLOR_PAPER_WHITE, linewidth=0.65))
        ax.text(0.10, y + 0.5, labels[code], ha="left", va="center", fontsize=7.7)
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for pop_index, population in enumerate(POPULATIONS):
                x = code_width + (algorithm_index * len(POPULATIONS) + pop_index) * cell_width
                cell = cells.get((code, algorithm, population))
                applicable = cell is not None and _boolean(cell["applicable"])
                runtime = (float(cell["restricted_mean_seconds"])
                           if cell is not None and cell["restricted_mean_seconds"].strip() else None)
                color = (RUNTIME_CMAP(norm(runtime)) if runtime is not None and runtime > 0
                         else COLOR_PAPER_GRAY_VERY_LIGHT)
                hatch = None
                if cell is not None and applicable and any(_integer(cell, field) for field in (
                    "num_unresolved_labels", "num_generation_failures",
                    "num_certification_failures", "num_timeouts", "num_memory_limited",
                    "num_errors", "num_incorrect", "num_blocked",
                )):
                    hatch = "///"
                ax.add_patch(Rectangle((x, y), cell_width, 1, facecolor=color,
                                       edgecolor=COLOR_PAPER_WHITE, linewidth=0.65,
                                       hatch=hatch))
                if cell is None:
                    text, note = "missing", ""
                elif not applicable:
                    text, note = "N/A", ""
                else:
                    text = cell["primary_decider"] or "—"
                    note = _cell_note(cell)
                text_color = _text_color(color)
                ax.text(x + cell_width / 2, y + (0.40 if note else 0.5), text,
                        ha="center", va="center", fontsize=7.1, fontweight="bold",
                        color=text_color)
                if note:
                    ax.text(x + cell_width / 2, y + 0.72, note, ha="center", va="center",
                            fontsize=5.3, color=text_color)

    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")
    figure.suptitle("A8 Hybrid Runtime, Decision Stage, and Coverage", x=0.43,
                    y=0.98, fontsize=12)

    colorbar_ax = figure.add_axes([0.855, 0.72, 0.021, 0.16])
    bar = figure.colorbar(scalar_mappable(RUNTIME_CMAP, norm), cax=colorbar_ax)
    decimal_ticks(bar)
    mark_timeout(bar, timeout)
    bar.set_label("Restricted mean runtime [s]", labelpad=8)

    legend_ax = figure.add_axes([0.835, 0.10, 0.16, 0.55])
    legend_ax.axis("off")
    legend_ax.text(0, 1.0, "Decision stage", ha="left", va="top", fontsize=9,
                   fontweight="bold")
    lines = [f"{tag:<4} {label}" for tag, label in COMPONENT_LEGEND]
    lines.extend(("", "e#   equivalent proposals skipped", "u#   unresolved negative labels", "t#   execution timeouts",
                  "f#   other failures/blocked", "/#   requested instances",
                  "///  cell has unresolved/failure", "N/A  PM-CSS on non-CSS code"))
    legend_ax.text(0, 0.95, "\n".join(lines), ha="left", va="top", fontsize=6.7,
                   linespacing=1.35, family="monospace", color="#202020")
    ax.add_patch(Rectangle((0, 0), total_width, total_height, facecolor="none",
                           edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK, linewidth=0.7))
    return save_png(figure, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=INPUT)
    parser.add_argument("--output-file", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    render(arguments.input_file, arguments.output_file)
