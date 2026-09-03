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
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter

from benchmarks.thesis.thesis_prototypes import measurement_dimensions

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "paper" / "results"
DIMENSIONS = tuple(measurement_dimensions())

# Thesis palette, mirrored from settings/commands.tex so figures and text match.
COLOR_PAPER_ZX_GREEN = "#74D374"
COLOR_PAPER_ZX_ORANGE = "#FFA404"
COLOR_PAPER_ZX_GREEN_MUTED = "#CCFFBF"
COLOR_PAPER_ZX_ORANGE_MUTED = "#FFD9B2"
COLOR_PAPER_ZX_GREEN_DARK = "#199900"
COLOR_PAPER_ZX_ORANGE_DARK = "#FF7C00"
COLOR_PAPER_ZX_BLUE = "#2E6DF1"

COLOR_PAPER_LIGHT_RED = "#F39385"
COLOR_PAPER_RED = "#E9341A"
COLOR_PAPER_DARK_RED = "#A72512"

COLOR_PAPER_LIGHT_BLUE = "#D5DEF8"
COLOR_PAPER_MEDIUM_BLUE = "#94ABEA"
COLOR_PAPER_BLUE = "#2E61F1"
COLOR_PAPER_DARK_BLUE = "#1F45B2"

COLOR_PAPER_LIGHT_GREEN = "#C3DCA7"
COLOR_PAPER_GREEN_DEEP = "#6DA226"

COLOR_PAPER_GREEN = "#98D256"
COLOR_PAPER_DARK_GREEN = "#57891F"

COLOR_PAPER_YELLOW_LIGHT = "#FDFAC8"
COLOR_PAPER_YELLOW_DARK = "#FFF56D"

COLOR_PAPER_SALMON_LIGHT = "#FFDEDF"
COLOR_PAPER_SALMON = "#FFC9CA"
COLOR_PAPER_SALMON_DARK = "#FFA7A9"

COLOR_PAPER_LIGHT_LILA = "#CAB6E6"
COLOR_PAPER_LILA = "#A283CD"
COLOR_PAPER_DARK_LILA = "#7F5DC9"
COLOR_PAPER_CYAN_STRONG = "#0D9299"

COLOR_PAPER_LIGHT_CYAN = "#81C9CE"
COLOR_PAPER_CYAN = "#3B969C"
COLOR_PAPER_DARK_CYAN = "#1F6367"

COLOR_PAPER_LIGHT_PINK = "#FEBAFF"
COLOR_PAPER_PINK = "#F794F9"
COLOR_PAPER_DARK_PINK = "#EC71EE"

COLOR_PAPER_GRAY_VERY_LIGHT = "#F9F9F9"
COLOR_PAPER_GRAY_LIGHT = "#F2F2F2"
COLOR_PAPER_GRAY_MEDIUM = "#E8E8E8"
COLOR_PAPER_GRAY_DARK = "#DFDFDF"
COLOR_PAPER_GRAY_VERY_DARK = "#CCCCCC"
COLOR_PAPER_GRAY_VERY_VERY_DARK = "#808080"

COLOR_PAPER_GREEN_RAMP = ("#EFF8E5", "#D5EDBB", "#ACDB76", "#7AB931", "#476B1F")
COLOR_PAPER_PINK_RAMP = ("#FFDDFE", "#FF9FFF", "#D158D4", "#8F0493", "#450048")
COLOR_PAPER_ORANGE_RAMP = ("#FFF3DE", "#FFE0A8", "#FFC052", "#EB9600", "#865703")
COLOR_PAPER_CYAN_RAMP = ("#E7F5F6", "#C0E5E7", "#81CBCF", "#40A4AA", "#275F62")

COLOR_PAPER_WHITE = "#FFFFFF"

WIDE_TEXT_SCALE = 1.3

TIMEOUT_SECONDS = 5_400.0

EMPTY = COLOR_PAPER_GRAY_VERY_LIGHT
MEMORY_BLUE = COLOR_PAPER_BLUE
ERROR_PURPLE = COLOR_PAPER_DARK_LILA

RUNTIME_CMAP = LinearSegmentedColormap.from_list(
    "runtime",
    [
        COLOR_PAPER_YELLOW_LIGHT,
        COLOR_PAPER_ZX_ORANGE_MUTED,
        COLOR_PAPER_LIGHT_RED,
        COLOR_PAPER_RED,
        COLOR_PAPER_DARK_RED,
    ],
)

# A2 increases from no refinement to complete pairwise refinement, so stronger
# results receive the darker end of the sequential palette.
SIGNATURE_CMAP = LinearSegmentedColormap.from_list(
    "signature_space", COLOR_PAPER_CYAN_RAMP
)

# Diverging ratio scale: white at parity, blue where the invariant is cheaper
# than the backend it screens for, red where it costs more. The arms are
# mirrored in log space, so equal colour intensity means an equal factor in
# either direction.
RELATIVE_DECADES = 3
RELATIVE_BLUE_ARM = (
    COLOR_PAPER_WHITE,
    COLOR_PAPER_LIGHT_BLUE,
    COLOR_PAPER_MEDIUM_BLUE,
    COLOR_PAPER_BLUE,
    COLOR_PAPER_DARK_BLUE,
)
RELATIVE_RED_ARM = (
    COLOR_PAPER_WHITE,
    COLOR_PAPER_SALMON,
    COLOR_PAPER_LIGHT_RED,
    COLOR_PAPER_RED,
    COLOR_PAPER_DARK_RED,
)
# Ratios cluster within one decade of parity, so the arms are warped to reach a
# readable tint after a factor of about two and spend their last colours on the
# three-decade tails. Both arms carry the same warp, so the scale stays
# symmetric and the colorbar shows the compression it applies.
RELATIVE_GAMMA = 1.8


def _relative_stops() -> list[tuple[float, str]]:
    stops: list[tuple[float, str]] = []
    for arm, direction in ((RELATIVE_BLUE_ARM, -1), (RELATIVE_RED_ARM, 1)):
        steps = len(arm) - 1
        for index, color in enumerate(arm):
            distance = (index / steps) ** RELATIVE_GAMMA
            stops.append((0.5 + direction * 0.5 * distance, color))
    return sorted(stops)


RELATIVE_CMAP = LinearSegmentedColormap.from_list("relative_runtime", _relative_stops())


def relative_norm(decades: int = RELATIVE_DECADES) -> LogNorm:
    """Symmetric log scale around 1, clipping the tails onto the end colours.

    ``LogNorm`` puts 1 exactly at the midpoint of a symmetric decade range, so
    the diverging colormap's white lands on parity without a second norm class.
    """
    return LogNorm(vmin=10.0**-decades, vmax=10.0**decades, clip=True)


def ratio_ticks(bar, decades: int = RELATIVE_DECADES) -> None:
    """Label a ratio colorbar as factors, reading outwards from parity."""
    ticks = [10.0**power for power in range(-decades, decades + 1)]
    labels = [
        f"1/{10**-power:g}" if power < 0 else f"{10**power:g}"
        for power in range(-decades, decades + 1)
    ]
    bar.set_ticks(ticks, labels=labels)


def outline_partition(ax, n: int, r: int, index: int, count: int, color: Any) -> None:
    """Trace one partition of a parameter cell without repainting its fill."""
    width = 1 / count
    ax.add_patch(
        Rectangle(
            (n - 0.5 + index * width, r - 0.5),
            width,
            1,
            facecolor="none",
            edgecolor=color,
            linewidth=0.5,
            zorder=4,
        )
    )


def use_style(scale: float = 1.0) -> None:
    """Apply compact, paper-friendly plotting defaults.

    ``scale`` enlarges every text size together. The three-panel figures are the
    widest, so the page shrinks them furthest and their type has to start larger
    to land at the same printed size as the two-panel ones.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8 * scale,
            "axes.titlesize": 10 * scale,
            "axes.labelsize": 9 * scale,
            "xtick.labelsize": 7 * scale,
            "ytick.labelsize": 7 * scale,
            "legend.fontsize": 7 * scale,
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


def parameter_axis(
    ax,
    title: str,
    *,
    nmax: int = 47,
    empty_color: Any = EMPTY,
) -> None:
    """Draw the closed integer ``(n, r)`` triangle as edge-to-edge squares."""
    for n in range(3, nmax + 1):
        for r in range(1, n + 1):
            ax.add_patch(
                Rectangle(
                    (n - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=empty_color,
                    edgecolor=COLOR_PAPER_WHITE,
                    linewidth=0.12,
                    zorder=0,
                )
            )
    ax.set_xlim(2.5, nmax + 0.5)
    ax.set_ylim(0.5, nmax + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ticks = [tick for tick in (3, 5, 10, 15, 20, 25, 30, 35, 40, 47) if tick <= nmax]
    if nmax not in ticks:
        ticks.append(nmax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(False)
    ax.set_xlabel("Physical qubits $n$")
    ax.set_ylabel("Stabilizer rank $r=n-k$")
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
            edgecolor=COLOR_PAPER_WHITE,
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
        ax.scatter(x, r, marker="x", s=34, linewidths=1.1, color=MEMORY_BLUE, zorder=5)
    if integer(row.get("num_errors")):
        ax.scatter(x, r, marker="*", s=14, linewidths=0.3, color=ERROR_PURPLE, zorder=5)


def runtime_norm(values: Iterable[float], *, timeout: float = TIMEOUT_SECONDS) -> LogNorm:
    """Scale runtimes logarithmically, always reaching the collector timeout.

    Measured means stay below the cap, so without stretching the top the timeout
    would fall off the colorbar and ``mark_timeout`` would have nothing to draw.
    """
    positive = [value for value in values if value > 0]
    if not positive:
        return LogNorm(vmin=1e-3, vmax=timeout)
    low, high = min(positive), max(max(positive), timeout)
    if low == high:
        low, high = low / 2, high * 2
    return LogNorm(vmin=low, vmax=high)


def decimal_ticks(bar) -> None:
    """Label a logarithmic colorbar with plain decimals rather than powers of ten."""
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def mark_timeout(bar, timeout: float = TIMEOUT_SECONDS) -> None:
    """Draw the collector timeout onto a runtime colorbar.

    Cells at the cap are means over runs that hit it, so the reader needs to see
    where the measurable range ends rather than inferring it from the top tick.
    """
    if not bar.norm.vmin <= timeout <= bar.norm.vmax:
        return
    if timeout < bar.norm.vmax:
        # Only worth a rule when it falls inside the bar; at the top it would
        # sit under the frame and read as a border rather than a threshold.
        bar.ax.axhline(
            timeout,
            color=COLOR_PAPER_GRAY_VERY_VERY_DARK,
            linewidth=0.9,
            zorder=5,
        )
    bar.ax.yaxis.set_minor_locator(FixedLocator([timeout]))
    bar.ax.yaxis.set_minor_formatter(FixedFormatter([f"{timeout:g}"]))
    bar.ax.tick_params(axis="y", which="minor", length=3)


def save_png(figure, path: Path) -> Path:
    """Write exactly one cropped PNG and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=COLOR_PAPER_WHITE)
    plt.close(figure)
    print(f"wrote {path}", flush=True)
    return path


def half_cell_key(
    fill: str,
    color: Any,
    label: str,
    *,
    size: float = 7,
) -> Line2D:
    """Legend key shaped like one half of a split parameter cell.

    ``fill`` is ``"left"`` or ``"right"``. Filling only the half the entry
    describes lets the key say which side of the cell it means, instead of
    leaving the reader to infer it from the order of a row of flat patches.
    """
    return Line2D(
        [], [], marker="s", fillstyle=fill, linestyle="none", markersize=size,
        markerfacecolor=color,
        markerfacecoloralt=COLOR_PAPER_WHITE,
        markeredgecolor=color,
        label=label,
    )


def polarity_legend() -> list[Line2D]:
    return [
        half_cell_key("left", COLOR_PAPER_GRAY_VERY_VERY_DARK, "left half: equivalent"),
        half_cell_key("right", COLOR_PAPER_GRAY_VERY_VERY_DARK, "right half: inequivalent"),
    ]


def failure_legend() -> list[Line2D]:
    return [
        Line2D([], [], marker="x", linestyle="none", color=MEMORY_BLUE, label="OOM"),
        Line2D([], [], marker="*", linestyle="none", color=ERROR_PURPLE, label="error / wrong result"),
    ]


def scalar_mappable(cmap, norm: Normalize):
    from matplotlib.cm import ScalarMappable

    result = ScalarMappable(norm=norm, cmap=cmap)
    result.set_array([])
    return result
