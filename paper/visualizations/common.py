"""Shared visual language for the fixed paper figures."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from benchmarks.thesis.thesis_prototypes import measurement_dimensions

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "paper" / "results"
DIMENSIONS = tuple(measurement_dimensions())

EMPTY = "#FCFCFC"
GRID = "#B9B9B9"
TEXT = "#202020"
MEMORY_BLUE = "#2E61F1"
ERROR_PURPLE = "#7F5DC9"

RUNTIME_CMAP = LinearSegmentedColormap.from_list(
    "runtime", ["#FDFAC8", "#FFD9B2", "#F39385", "#E9341A", "#A72512"]
)
SIGNATURE_CMAP = LinearSegmentedColormap.from_list(
    "signature_space", ["#FDFAC8", "#FFA404", "#E9341A", "#A72512"]
)


def use_style() -> None:
    """Apply compact, paper-friendly plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "font.family": "sans-serif",
        }
    )


def load_rows(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    """Read a collector CSV and reject stale schemas with a useful message."""
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run its paper experiment extractor first")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(required) - fields
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"{path} uses an obsolete schema (missing {names}); "
                "rerun the corresponding experiment extractor"
            )
        return list(reader)


def load_plot_rows(path: Path, required: Sequence[str]) -> tuple[list[dict[str, str]], bool]:
    """Prefer real collector data, falling back explicitly to temporary fixtures."""
    try:
        return load_rows(path, required), False
    except (FileNotFoundError, ValueError) as primary_error:
        synthetic = path.with_name("synthetic_by_cell.csv")
        if not synthetic.is_file():
            raise primary_error
        print(f"warning: {primary_error}; using {synthetic}", flush=True)
        return load_rows(synthetic, required), True


def mark_synthetic(figure, synthetic: bool) -> None:
    """Make it impossible to mistake a temporary fixture render for paper data."""
    if synthetic:
        figure.text(
            0.99,
            0.01,
            "SYNTHETIC PLACEHOLDER DATA",
            ha="right",
            va="bottom",
            color="#9A1B1B",
            fontsize=7,
            fontweight="bold",
        )


def boolean(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value)
    return result if result == result else None


def integer(value: str | None) -> int:
    return int(value or 0)


def aggregate_cells(
    rows: Sequence[Mapping[str, str]], value_field: str, count_field: str
) -> dict[tuple[int, int], dict[str, float | int]]:
    """Combine positive/negative rows into success-count-weighted cell means."""
    grouped: dict[tuple[int, int], dict[str, float | int]] = {}
    for row in rows:
        key = (int(row["n"]), int(row["r"]))
        cell = grouped.setdefault(
            key,
            {
                "weighted_value": 0.0,
                "num_successful": 0,
                "num_timeouts": 0,
                "num_memory_limited": 0,
                "num_errors": 0,
            },
        )
        count = int(row[count_field])
        value = number(row[value_field])
        if value is not None:
            cell["weighted_value"] += value * count
            cell["num_successful"] += count
        for field in ("num_timeouts", "num_memory_limited", "num_errors"):
            cell[field] += int(row.get(field, 0) or 0)
    for cell in grouped.values():
        count = int(cell["num_successful"])
        cell["mean_value"] = (
            float(cell["weighted_value"]) / count if count else float("nan")
        )
    return grouped


def parameter_axis(ax, title: str) -> None:
    """Draw the closed integer ``(n, r)`` triangle as edge-to-edge squares."""
    for n in range(3, 48):
        for r in range(1, n + 1):
            ax.add_patch(
                Rectangle(
                    (n - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=EMPTY,
                    edgecolor="#F7F7F7",
                    linewidth=0.12,
                    zorder=0,
                )
            )
    ax.plot([2.5, 47.5], [2.5, 47.5], color=GRID, linewidth=0.55, zorder=3)
    ax.set_xlim(2.5, 47.5)
    ax.set_ylim(0.5, 47.5)
    ax.set_aspect("equal", adjustable="box")
    ticks = [3, 5, 10, 15, 20, 25, 30, 35, 40, 47]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(False)
    ax.set_xlabel("physical qubits $n$")
    ax.set_ylabel("stabilizer rank $r=n-k$")
    ax.set_title(title, pad=6)
    ax.spines[["top", "right"]].set_visible(False)


def partition_cell(
    ax,
    n: int,
    r: int,
    index: int,
    count: int,
    color: Any,
) -> None:
    """Paint one vertical partition of a square parameter cell."""
    width = 1 / count
    ax.add_patch(
        Rectangle(
            (n - 0.5 + index * width, r - 0.5),
            width,
            1,
            facecolor=color,
            edgecolor="#FFFFFF",
            linewidth=0.16,
            zorder=2,
        )
    )


def split_cell(ax, n: int, r: int, left: Any | None, right: Any | None) -> None:
    """Paint equivalent (left) and inequivalent (right) square half-cells."""
    if left is not None:
        partition_cell(ax, n, r, 0, 2, left)
    if right is not None:
        partition_cell(ax, n, r, 1, 2, right)


def failure_marks(ax, n: int, r: int, row: Mapping[str, Any], side: int = 0) -> None:
    """Overlay explicit memory, timeout, and other-error glyphs."""
    x = n + side * 0.17
    if integer(row.get("num_memory_limited")):
        ax.scatter(x, r, marker="x", s=17, linewidths=0.8, color=MEMORY_BLUE, zorder=5)
    if integer(row.get("num_errors")):
        ax.scatter(x, r, marker="*", s=14, linewidths=0.3, color=ERROR_PURPLE, zorder=5)


def runtime_norm(values: Iterable[float]) -> LogNorm:
    positive = [value for value in values if value > 0]
    if not positive:
        return LogNorm(vmin=1e-3, vmax=1.0)
    low, high = min(positive), max(positive)
    if low == high:
        low, high = low / 2, high * 2
    return LogNorm(vmin=low, vmax=high)


def save_png(figure, path: Path) -> Path:
    """Write exactly one cropped PNG and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path}", flush=True)
    return path


def polarity_legend() -> list[Line2D]:
    return [
        Line2D([], [], marker="s", fillstyle="left", markerfacecolor="#555555", markerfacecoloralt="#FFFFFF", markeredgecolor="#777777", linestyle="none", markersize=7, label="left half: equivalent"),
        Line2D([], [], marker="s", fillstyle="right", markerfacecolor="#555555", markerfacecoloralt="#FFFFFF", markeredgecolor="#777777", linestyle="none", markersize=7, label="right half: inequivalent"),
    ]


def failure_legend() -> list[Line2D]:
    return [
        Line2D([], [], marker="x", linestyle="none", color=MEMORY_BLUE, label="memory limit"),
        Line2D([], [], marker="*", linestyle="none", color=ERROR_PURPLE, label="error / wrong result"),
    ]


def scalar_mappable(cmap, norm: Normalize):
    from matplotlib.cm import ScalarMappable

    result = ScalarMappable(norm=norm, cmap=cmap)
    result.set_array([])
    return result
