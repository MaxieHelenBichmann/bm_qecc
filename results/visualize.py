"""Create matplotlib plots from benchmark statistics CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln

Axis = Literal["n", "k", "r", "d", "s"]
StatFileKind = Literal["randomized", "named", "mixed"]

AXIS_LABELS: dict[Axis, str] = {
    "n": "n [physical qubits]",
    "k": "k [logical qubits]",
    "r": "r = n - k [number of stabilizer generators]",
    "d": "d [density]",
    "s": "s [symmetry]",
}

def expected_functions(algorithm: str) -> Callable:
    """Return the expected theoretical function for a given algorithm."""
    # TODO
    def _pm_css_bruteforce(n, a, b):
        return a * factorial(n) + b
    
    def _pm_css_classical(n, a, b):
        return a * factorial(n) + b
    
    def _pm_css_graph_iso(n, a, b):
        return a * factorial(n) + b
    
    def _pm_css_matroid(n, a, b):
        return a * factorial(n) + b
    
    def _pm_css_sat(n, a, b):
        return n
    
    def _pm_stb_aut(n, a, b):
        return n
    
    def _pm_stb_bruteforce(n, a, b):
        return n
    
    def _pm_stb_classical(n, a, b):
        return n
    
    def _pm_stb_graph_iso(n, a, b):
        return n
    
    def _pm_stb_sat(n, a, b):
        return n
    
    def _lc_equ_bruteforce(n, a, b):
        return n
    
    def _lc_equ_graph_state(n, a, b):
        return n
    
    def _lc_equ_graph_state_small_k(n, a, b):
        return n
    
    def _lc_equ_graph_iso(n, a, b):
        return n
    
    def _lc_equ_sat(n, a, b):
        return n
    
    def _lc_css_bruteforce(n, a, b):
        return n
    
    def _lc_css_kls(n, a, b):
        return n
    
    def _lc_css_cliff_orbit(n, a, b):
        return n
    
    def _lc_css_lc_orbit(n, a, b):
        return n
    
    def _lc_css_sat(n, a, b):
        return n
    
    if algorithm == "pm_css_bruteforce":
        return _pm_css_bruteforce
    if algorithm == "pm_css_classical":
        return _pm_css_classical
    if algorithm == "pm_css_graph_iso":
        return _pm_css_graph_iso
    if algorithm == "pm_css_matroid":
        return _pm_css_matroid
    if algorithm == "pm_css_sat":
        return _pm_css_sat
    if algorithm == "pm_stb_aut":
        return _pm_stb_aut
    if algorithm == "pm_stb_bruteforce":
        return _pm_stb_bruteforce
    if algorithm == "pm_stb_classical":
        return _pm_stb_classical
    if algorithm == "pm_stb_graph_iso":
        return _pm_stb_graph_iso
    if algorithm == "pm_stb_sat":
        return _pm_stb_sat
    if algorithm == "lc_equ_graph_state":
        return _lc_equ_graph_state
    if algorithm == "lc_equ_graph_state_small_k":
        return _lc_equ_graph_state_small_k
    if algorithm == "lc_equ_bruteforce":
        return _lc_equ_bruteforce
    if algorithm == "lc_equ_graph_iso":
        return _lc_equ_graph_iso
    if algorithm == "lc_equ_sat":
        return _lc_equ_sat
    if algorithm == "lc_css_bruteforce":
        return _lc_css_bruteforce
    if algorithm == "lc_css_kls":
        return _lc_css_kls
    if algorithm == "lc_css_cliff_orbit":
        return _lc_css_cliff_orbit
    if algorithm == "lc_css_lc_orbit":
        return _lc_css_lc_orbit
    if algorithm == "lc_css_sat":
        return _lc_css_sat
    
def boundary_functions(algorithm: str) -> Callable | None:
    def _permutation(n, a, b):
        return a * np.exp(gammaln(np.asarray(n, dtype=float) + 1.0)) + b
    
    def _lc(n, a, b):
        return a * np.power(6.0, np.asarray(n, dtype=float)) + b
    
    if algorithm.startswith("pm_"):
        return _permutation
    
    if algorithm.startswith("lc_"):
        return _lc

    return None


@dataclass(frozen=True)
class PlotColors:
    """Colors for the visual elements of one plotted series."""

    line: str
    point: str
    error: str
    maximum: str


COLOR_FAMILIES_POS: tuple[PlotColors, ...] = (
    PlotColors(
        line="cornflowerblue",
        point="royalblue",
        error="lightsteelblue",
        maximum="lightskyblue",
    ),
    PlotColors(
        line="forestgreen",
        point="darkgreen",
        error="palegreen",
        maximum="mediumseagreen",
    ),
    PlotColors(
        line="mediumpurple",
        point="indigo",
        error="thistle",
        maximum="plum",
    ),
    PlotColors(
        line="lightseagreen",
        point="darkcyan",
        error="paleturquoise",
        maximum="mediumturquoise",
    ),
)
COLOR_FAMILIES_NEG: tuple[PlotColors, ...] = (
    PlotColors(
        line="orange",
        point="darkorange",
        error="moccasin",
        maximum="peru",
    ),
    PlotColors(
        line="indianred",
        point="darkred",
        error="mistyrose",
        maximum="salmon",
    ),
    PlotColors(
        line="peru",
        point="saddlebrown",
        error="tan",
        maximum="burlywood",
    ),
    PlotColors(
        line="hotpink",
        point="mediumvioletred",
        error="pink",
        maximum="palevioletred",
    ),
)

@dataclass(frozen=True)
class StatRow:
    """One row emitted by benchmarks.run.write_stats."""

    algorithm: str
    name: str | None
    n: int
    k: int
    positive: bool
    density: float | None
    symmetry: float | None
    mean_seconds: float
    stddev_seconds: float
    maximum_seconds: float | None

    @property
    def r(self) -> int:
        return self.n - self.k

    def axis_value(self, axis: Axis) -> float:
        if axis == "n":
            return self.n
        if axis == "k":
            return self.k
        if axis == "r":
            return self.r
        if axis == "d":
            if self.density is None:
                raise ValueError("Cannot use density as x-axis for rows without density.")
            return self.density
        if self.symmetry is None:
            raise ValueError("Cannot use symmetry as x-axis for rows without symmetry.")
        return self.symmetry


@dataclass(frozen=True)
class InvariantRow:
    """One row emitted by benchmarks.run.write_bms for invariant benchmarks."""

    algorithm: str
    case: str
    n: int
    k: int
    seconds: float
    result: bool | None
    expected: bool | None
    success: bool
    error: str

    @property
    def r(self) -> int:
        return self.n - self.k

    def axis_value(self, axis: Axis) -> float:
        if axis == "n":
            return self.n
        if axis == "k":
            return self.k
        if axis == "r":
            return self.r
        raise ValueError(f"Invariant plots only support n, k, or r as x-axis, got {axis!r}.")


@dataclass(frozen=True)
class PlotSeries:
    """A single line/scatter series in the diagram."""

    label: str
    rows: tuple[StatRow, ...]


@dataclass(frozen=True)
class PointLabel:
    """A direct label for one named benchmark point."""

    text: str
    x: float
    y: float
    color: str


def jittered_axis_values(rows: Sequence[StatRow], axis: Axis) -> list[float]:
    """Spread rows with identical x-values just enough to keep points visible."""
    axis_values = [row.axis_value(axis) for row in rows]
    unique_values = sorted(set(axis_values))
    if len(unique_values) > 1:
        min_gap = min(
            right - left
            for left, right in zip(unique_values, unique_values[1:])
            if right > left
        )
        jitter_step = min_gap * 0.12
    else:
        jitter_step = max(abs(unique_values[0]) * 0.02, 0.1)

    duplicates: dict[float, list[int]] = {}
    for index, value in enumerate(axis_values):
        duplicates.setdefault(value, []).append(index)

    jittered_values = list(axis_values)
    for value, indices in duplicates.items():
        if len(indices) == 1:
            continue
        midpoint = (len(indices) - 1) / 2
        for duplicate_index, row_index in enumerate(indices):
            jittered_values[row_index] = value + (duplicate_index - midpoint) * jitter_step
    return jittered_values


def parse_optional_float(value: str | None) -> float | None:
    """Parse an optional CSV float field."""
    if value is None or value == "" or value.strip().lower() == "none":
        return None
    return float(value)


def parse_bool(value: str) -> bool:
    """Parse boolean values written by csv.DictWriter."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected boolean field, got {value!r}.")


def parse_optional_bool(value: str | None) -> bool | None:
    """Parse an optional CSV boolean field."""
    if value is None or value == "" or value.strip().lower() == "none":
        return None
    return parse_bool(value)


def read_stats_csv(path: Path) -> list[StatRow]:
    """Read a statistics CSV written by benchmarks.run.write_stats.

    The first row contains only the benchmark seed. The actual header starts on
    the second line.
    """
    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            return []

        sample = file.readline()
        if not sample:
            return []

        header = sample.strip().split(",")
        if header and header[0] == "algorithm":
            file.seek(0)
            next(file)
        else:
            file.seek(0)

        reader = csv.DictReader(file)
        return [
            StatRow(
                algorithm=row["algorithm"],
                name=row["name"] or None,
                n=int(row["n"]),
                k=int(row["k"]),
                positive=parse_bool(row["positive"]),
                density=parse_optional_float(row.get("density")),
                symmetry=parse_optional_float(row.get("symmetry")),
                mean_seconds=float(row["mean_seconds"]),
                stddev_seconds=float(row["stddev_seconds"]),
                maximum_seconds=parse_optional_float(row.get("maximum_seconds")),
            )
            for row in reader
        ]


def read_invariant_csv(path: Path) -> list[InvariantRow]:
    """Read an invariant benchmark CSV written by benchmarks.run.write_bms."""
    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            return []

        sample = file.readline()
        if not sample:
            return []

        header = sample.strip().split(",")
        if header and header[0] == "algorithm":
            file.seek(0)
            next(file)
        else:
            file.seek(0)

        reader = csv.DictReader(file)
        return [
            InvariantRow(
                algorithm=row["algorithm"],
                case=row["case"],
                n=int(row["n"]),
                k=int(row["k"]),
                seconds=float(row["seconds"]),
                result=parse_optional_bool(row.get("result")),
                expected=parse_optional_bool(row.get("expected")),
                success=parse_bool(row["success"]),
                error=row.get("error", ""),
            )
            for row in reader
        ]


def stat_file_kind(rows: Sequence[StatRow]) -> StatFileKind:
    """Classify statistics rows by whether they come from generated or named cases."""
    has_named_rows = any(row.name is not None for row in rows)
    has_randomized_rows = any(row.name is None for row in rows)
    if has_named_rows and has_randomized_rows:
        return "mixed"
    if has_named_rows:
        return "named"
    return "randomized"


def matches_filter(row: StatRow, args: argparse.Namespace) -> bool:
    """Return whether a statistics row should be included."""
    if args.x == "d" and row.density is None:
        return False
    if args.x == "s" and row.symmetry is None:
        return False
    if args.algorithm and row.algorithm not in args.algorithm:
        return False
    if args.name is not None and row.name != args.name:
        return False
    if args.positive != "all" and row.positive != (args.positive == "true"):
        return False
    if row.name is not None:
        return True
    if args.n is not None and row.n != args.n:
        return False
    if args.k is not None and row.k != args.k:
        return False
    if args.r is not None and row.r != args.r:
        return False
    if args.density is not None and row.density != args.density:
        return False
    if args.symmetry is not None and row.symmetry != args.symmetry:
        return False
    return True


def series_label(row: StatRow, include_positive: bool, include_algorithm: bool) -> str:
    """Build a compact legend label for a row group."""
    if include_positive:
        sign = "positive" if row.positive else "negative"
        if not include_algorithm:
            return sign
        return f"{row.algorithm} ({sign})"
    if not include_algorithm:
        return "_nolegend_"
    return row.algorithm


def build_series(
    rows: Iterable[StatRow],
    axis: Axis,
    include_positive: bool,
    include_algorithm: bool,
) -> list[PlotSeries]:
    """Group rows into plot series."""
    grouped: dict[str, list[StatRow]] = {}
    for row in rows:
        grouped.setdefault(series_label(row, include_positive, include_algorithm), []).append(row)

    return [
        PlotSeries(
            label=label,
            rows=tuple(sorted(group, key=lambda row: row.axis_value(axis))),
        )
        for label, group in sorted(grouped.items())
    ]


def configure_axis_constraints(args: argparse.Namespace, parser: argparse.ArgumentParser, kind: StatFileKind) -> None:
    """Validate fixed dimension filters for the selected x-axis."""
    if kind == "named":
        if args.x in {"d", "s"}:
            parser.error(f"--x {args.x} cannot be used for named benchmark cases because they have no generated-code metadata.")
        if args.theory:
            parser.error("--theory cannot be used for named benchmark cases because those plots are scatter-only.")
        return

    if kind == "mixed":
        parser.error("Cannot infer one plotting mode from a CSV containing both named and randomized rows.")

    if args.x == "n" and args.k is None:
        parser.error("--x n requires --k so only one code rate family is plotted.")
    if args.x == "k" and args.n is None:
        parser.error("--x k requires --n so only one block length is plotted.")
    if args.x == "r" and (args.n is None) == (args.k is None):
        parser.error("--x r requires exactly one of --n or --k.")
    if args.x in {"d", "s"} and (args.n is None or args.k is None):
        parser.error(f"--x {args.x} requires both --n and --k.")


def fixed_parameter_title(args: argparse.Namespace, rows: Sequence[StatRow]) -> str:
    """Build a title line that describes fixed dimension parameters."""
    algorithms = sorted({row.algorithm for row in rows})
    single_algorithm = algorithms[0] if len(algorithms) == 1 else None
    title = args.title or "Benchmark runtimes"
    if single_algorithm is not None:
        title = f"{title}: {single_algorithm}"

    if any(row.name is not None for row in rows):
        return title

    fixed_parts = []
    if args.n is not None and args.x != "n":
        fixed_parts.append(f"n = {args.n}")
    if args.k is not None and args.x != "k":
        fixed_parts.append(f"k = {args.k}")
    if args.r is not None and args.x != "r":
        fixed_parts.append(f"r = {args.r}")
    if args.density is not None and args.x != "d":
        fixed_parts.append(f"d = {args.density:g}")
    if args.symmetry is not None and args.x != "s":
        fixed_parts.append(f"s = {args.symmetry:g}")

    context = ", ".join(fixed_parts)
    title_parts = [title]
    if context:
        title_parts.append(context)
    return "\n".join(title_parts)


def label_position(
    label: PointLabel,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cluster_index: int,
) -> tuple[tuple[int, int], str, str]:
    """Choose a readable offset for a label near its data point."""
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    x_fraction = 0.5 if x_span == 0 else (label.x - xlim[0]) / x_span
    y_fraction = 0.5 if y_span == 0 else (label.y - ylim[0]) / y_span

    place_left = x_fraction > 0.68
    place_below = y_fraction > 0.72
    horizontal_offsets = (-5, -16) if place_left else (5, 16)
    vertical_offsets = (-5, -14, -23) if place_below else (5, 14, 23)

    x_offset = horizontal_offsets[(cluster_index // len(vertical_offsets)) % len(horizontal_offsets)]
    y_offset = vertical_offsets[cluster_index % len(vertical_offsets)]
    horizontal_alignment = "right" if place_left else "left"
    vertical_alignment = "top" if place_below else "bottom"
    return (x_offset, y_offset), horizontal_alignment, vertical_alignment


def clustered_point_labels(labels: Sequence[PointLabel], xlim: tuple[float, float]) -> list[list[PointLabel]]:
    """Group labels whose x-positions are close enough that text may overlap."""
    if not labels:
        return []

    x_span = xlim[1] - xlim[0]
    threshold = max(x_span * 0.08, 1e-12)
    clusters: list[list[PointLabel]] = []
    for label in sorted(labels, key=lambda point_label: (point_label.x, point_label.y)):
        if not clusters or abs(label.x - clusters[-1][-1].x) > threshold:
            clusters.append([label])
        else:
            clusters[-1].append(label)
    return clusters


def initial_fit_guess(
    boundary_function: Callable,
    x,
    y,
) -> tuple[float, float]:
    """Estimate stable starting parameters for a two-parameter boundary fit."""
    import numpy as np

    baseline = float(np.min(y))
    boundary_at_x = boundary_function(x, 1.0, 0.0)
    boundary_span = float(np.max(boundary_at_x) - np.min(boundary_at_x))
    y_span = float(np.max(y) - np.min(y))
    scale = y_span / boundary_span if boundary_span > 0 else 1.0
    return scale, baseline


def plot_boundary_fits(series: Sequence[PlotSeries], axis: Axis, ax) -> None:
    """Draw one faint fitted theoretical boundary curve per algorithm."""
    if axis not in {"n", "k", "r"}:
        return

    rows_by_algorithm: dict[str, list[StatRow]] = {}
    colors_by_algorithm: dict[str, PlotColors] = {}
    for color_index, item in enumerate(series):
        if not item.rows:
            continue
        algorithm = item.rows[0].algorithm
        rows_by_algorithm.setdefault(algorithm, []).extend(item.rows)
        colors_by_algorithm.setdefault(algorithm, COLOR_FAMILIES_POS[color_index % len(COLOR_FAMILIES_POS)])

    for algorithm, rows in rows_by_algorithm.items():
        boundary_function = boundary_functions(algorithm)
        if boundary_function is None or len(rows) < 2:
            continue

        x = np.asarray([row.axis_value(axis) for row in rows], dtype=float)
        y = np.asarray([row.mean_seconds for row in rows], dtype=float)
        if len(set(x)) < 2:
            continue

        try:
            params, _ = curve_fit(
                boundary_function,
                x,
                y,
                p0=initial_fit_guess(boundary_function, x, y),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            continue

        colors = colors_by_algorithm[algorithm]
        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_fit = np.maximum(boundary_function(x_fit, *params), 0)
        ax.plot(
            x_fit,
            y_fit,
            color=colors.line,
            alpha=0.2,
            linewidth=1.1,
            linestyle="--",
            label="_nolegend_",
            zorder=1,
        )


def draw_minute_guides(ax) -> None:
    """Draw minute and hour runtime guides without changing the data-driven axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    first_minute = 60
    last_minute = int(ylim[1] // 60) * 60

    for seconds in range(first_minute, last_minute + 1, 60):
        if seconds <= ylim[0]:
            continue
        is_hour = seconds % 3600 == 0
        ax.axhline(
            seconds,
            color="red",
            linestyle=":",
            linewidth=1.4 if is_hour else 0.5,
            alpha=0.7 if is_hour else 0.4,
            label="_nolegend_",
            zorder=0,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def configure_xaxis_ticks(axis: Axis, ax) -> None:
    """Keep discrete dimension axes labeled as integers."""
    if axis not in {"n", "k"}:
        return

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))


def invariant_algorithm_base(algorithm: str) -> str:
    """Return the base invariant name without a trailing subset-size suffix."""
    return re.sub(r"_s\d+$", "", algorithm)


def invariant_algorithm_variant(algorithm: str) -> int | None:
    """Return the subset-size suffix of an invariant algorithm, if present."""
    match = re.search(r"_s(\d+)$", algorithm)
    if match is None:
        return None
    return int(match.group(1))


def invariant_marker(algorithm: str) -> str:
    """Return a marker shape that distinguishes full and subset-sized variants."""
    variant = invariant_algorithm_variant(algorithm)
    if variant is None:
        return "o"
    markers = ("s", "^", "D", "P", "v", "X")
    return markers[(variant - 1) % len(markers)]


def invariant_alpha(algorithm: str) -> float:
    """Use transparency to keep stacked invariant dots readable."""
    return 0.5 if invariant_algorithm_variant(algorithm) is None else 0.62


def invariant_tone(base_color, algorithm: str):
    """Return a related but distinct tone for subset-sized invariant variants."""
    variant = invariant_algorithm_variant(algorithm)
    if variant is None:
        return base_color

    import colorsys

    red, green, blue, alpha = base_color
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    direction = -1 if variant % 2 == 0 else 1
    amount = min(0.1 + 0.06 * ((variant - 1) // 2), 0.28)
    lightness = max(0.18, min(0.82, lightness + direction * amount))
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (red, green, blue, alpha)


def plot_invariant_rows(
    rows: Sequence[InvariantRow],
    axis: Axis,
    output: Path | None,
    title: str | None,
) -> None:
    """Render invariant benchmark rows as a scatter plot."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit("matplotlib is required. Install it with `python3 -m pip install -r requirements.txt`.") from exc
        raise

    ordered_rows = tuple(sorted(rows, key=lambda row: (row.axis_value(axis), row.algorithm, row.case)))
    x_values = [row.axis_value(axis) for row in ordered_rows]
    algorithms = sorted({row.algorithm for row in ordered_rows})
    algorithm_bases = sorted({invariant_algorithm_base(algorithm) for algorithm in algorithms})

    cmap = plt.get_cmap("tab10" if len(algorithm_bases) <= 10 else "tab20")
    base_colors = {base: cmap(index % cmap.N) for index, base in enumerate(algorithm_bases)}
    colors = {
        algorithm: invariant_tone(base_colors[invariant_algorithm_base(algorithm)], algorithm)
        for algorithm in algorithms
    }

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for algorithm in algorithms:
        points = [
            (x, row.seconds, row.success)
            for x, row in zip(x_values, ordered_rows)
            if row.algorithm == algorithm
        ]
        if not points:
            continue
        x, y, success = zip(*points)
        ax.scatter(
            x,
            y,
            s=42,
            color=colors[algorithm],
            alpha=invariant_alpha(algorithm),
            marker=invariant_marker(algorithm),
            edgecolors="none",
            label=algorithm,
            zorder=3,
        )
        failed_points = [(px, py) for px, py, ok in points if not ok]
        if failed_points:
            failed_x, failed_y = zip(*failed_points)
            ax.scatter(
                failed_x,
                failed_y,
                s=55,
                color=colors[algorithm],
                marker="x",
                linewidths=1.3,
                label="_nolegend_",
                zorder=4,
            )

    ax.set_xlabel(AXIS_LABELS[axis])
    ax.set_ylabel("runtime [s]")
    ax.margins(x=0.12, y=0.18)
    ax.set_ylim(bottom=0)
    configure_xaxis_ticks(axis, ax)
    draw_minute_guides(ax)
    ax.grid(True, which="major", alpha=0.15)
    ax.legend()
    ax.set_title(title or "Invariant benchmark runtimes")

    if output is None:
        plt.show()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved diagram to {output}.")


def plot_series(
    series: Sequence[PlotSeries],
    axis: Axis,
    output: Path | None,
    title: str | None,
    show_theory: bool,
) -> None:
    """Render the selected series with mean/stddev and maximum markers."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit("matplotlib is required. Install it with `python3 -m pip install -r requirements.txt`.") from exc
        raise

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    point_labels: list[PointLabel] = []
    if show_theory:
        plot_boundary_fits(series, axis, ax)

    for color_index, item in enumerate(series):
        if item.rows and item.rows[0].positive:
            colors = COLOR_FAMILIES_POS[color_index % len(COLOR_FAMILIES_POS)]
        else:
            colors = COLOR_FAMILIES_NEG[color_index % len(COLOR_FAMILIES_NEG)]
            
        has_named_rows = any(row.name is not None for row in item.rows)
        x = jittered_axis_values(item.rows, axis) if has_named_rows else [row.axis_value(axis) for row in item.rows]
        mean = [row.mean_seconds for row in item.rows]
        lower_error = [min(row.stddev_seconds, row.mean_seconds) for row in item.rows]
        upper_error = [row.stddev_seconds for row in item.rows]
        maximum_points = [
            (label_x, row.maximum_seconds)
            for row, label_x in zip(item.rows, x)
            if row.maximum_seconds is not None
        ]

        ax.errorbar(
            x,
            mean,
            yerr=[lower_error, upper_error],
            marker="o",
            capsize=3,
            color=colors.line,
            ecolor=colors.error,
            markerfacecolor=colors.point,
            markeredgecolor=colors.point,
            linestyle="none" if has_named_rows else "-",
            linewidth=1.6,
            markersize=4.5,
            label=item.label,
            zorder=3,
        )
        if maximum_points:
            maximum_x, maximum = zip(*maximum_points)
            ax.scatter(maximum_x, maximum, s=18, alpha=0.5, color=colors.maximum, marker="x", zorder=3)
        if has_named_rows:
            for row, label_x in zip(item.rows, x):
                if row.name is None:
                    continue
                point_labels.append(
                    PointLabel(
                        text=row.name,
                        x=label_x,
                        y=row.mean_seconds,
                        color=colors.point,
                    )
                )

    ax.set_xlabel(AXIS_LABELS[axis])
    ax.set_ylabel("runtime [s]")
    ax.margins(x=0.12, y=0.18)
    ax.set_ylim(bottom=0)
    configure_xaxis_ticks(axis, ax)
    draw_minute_guides(ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for cluster in clustered_point_labels(point_labels, xlim):
        for cluster_index, label in enumerate(sorted(cluster, key=lambda point_label: point_label.y, reverse=True)):
            offset, horizontal_alignment, vertical_alignment = label_position(label, xlim, ylim, cluster_index)
            ax.annotate(
                label.text,
                (label.x, label.y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color=label.color,
                ha=horizontal_alignment,
                va=vertical_alignment,
                annotation_clip=False,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.78},
                arrowprops={"arrowstyle": "-", "color": label.color, "alpha": 0.45, "lw": 0.6},
            )
    ax.grid(True, which="major", alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    if title:
        ax.set_title(title)

    if output is None:
        plt.show()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved diagram to {output}.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Statistics CSV from benchmarks/run.py --stats, or invariant CSV from --inv.")
    parser.add_argument("--x", choices=("n", "k", "r", "d", "s"), required=True, help="Parameter used for the x-axis.")
    parser.add_argument("--output", type=Path, help="Where to save the diagram. Shows an interactive window if omitted.")
    parser.add_argument("--title", help="Optional diagram title.")
    parser.add_argument("--theory", action="store_true", help="Draw a faint fitted theoretical boundary function.")
    parser.add_argument("--inv", action="store_true", help="Read invariant benchmark CSVs written by benchmarks/run.py --inv.")

    parser.add_argument("--algorithm", action="append", help="Algorithm to include. Can be passed multiple times.")
    parser.add_argument("--name", help="Case name to include.")
    parser.add_argument("--positive", choices=("true", "false", "all"), default="all", help="Filter positive/negative cases.")

    parser.add_argument("--n", type=int, help="Fix block length n.")
    parser.add_argument("--k", type=int, help="Fix logical dimension k.")
    parser.add_argument("--r", type=int, help="Fix redundancy r = n - k.")
    parser.add_argument("--density", type=float, help="Fix generated-code density.")
    parser.add_argument("--symmetry", type=float, help="Fix generated-code symmetry.")

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the visualization CLI."""
    args = parse_args(argv)
    parser = build_parser()

    if args.inv:
        if args.x not in {"n", "k", "r"}:
            parser.error("--inv only supports --x n, --x k, or --x r.")
        if args.theory:
            parser.error("--theory cannot be used with --inv because invariant plots are scatter-only.")
        if args.name is not None:
            parser.error("--name cannot be used with --inv; invariant plots include all cases.")
        if args.positive != "all":
            parser.error("--positive cannot be used with --inv; invariant plots include all cases.")
        if any(value is not None for value in (args.n, args.k, args.r, args.density, args.symmetry)):
            parser.error("--inv does not support fixed parameter filters; choose only the x-axis.")

        all_rows = read_invariant_csv(args.csv)
        if not all_rows:
            raise SystemExit("No rows found in the invariant CSV.")

        rows = [
            row
            for row in all_rows
            if not args.algorithm or row.algorithm in args.algorithm
        ]
        if not rows:
            raise SystemExit("No invariant rows matched the selected filters.")

        plot_invariant_rows(
            rows,
            axis=args.x,
            output=args.output,
            title=args.title,
        )
        return 0

    all_rows = read_stats_csv(args.csv)
    if not all_rows:
        raise SystemExit("No rows found in the statistics CSV.")

    configure_axis_constraints(args, parser, stat_file_kind(all_rows))

    rows = [row for row in all_rows if matches_filter(row, args)]
    if not rows:
        raise SystemExit("No rows matched the selected filters.")

    series = build_series(
        rows,
        axis=args.x,
        include_positive=args.positive == "all",
        include_algorithm=len({row.algorithm for row in rows}) > 1,
    )
    plot_series(
        series,
        axis=args.x,
        output=args.output,
        title=fixed_parameter_title(args, rows),
        show_theory=args.theory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
