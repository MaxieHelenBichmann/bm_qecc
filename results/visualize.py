"""Create matplotlib plots from benchmark statistics CSV files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln

Axis = Literal["n", "k", "r", "d", "s"]

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
    
    def _lc_css_orbit(n, a, b):
        return n
    
    def _lc_css_orbit_small_k(n, a, b):
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
    if algorithm == "lc_css_orbit":
        return _lc_css_orbit
    if algorithm == "lc_css_orbit_small_k":
        return _lc_css_orbit_small_k
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


COLOR_FAMILIES: tuple[PlotColors, ...] = (
    PlotColors(
        line="cornflowerblue",
        point="royalblue",
        error="lightsteelblue",
        maximum="lightskyblue",
    ),
    PlotColors(
        line="orange",
        point="darkorange",
        error="moccasin",
        maximum="peru",
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
        line="indianred",
        point="darkred",
        error="mistyrose",
        maximum="salmon",
    ),
    PlotColors(
        line="lightseagreen",
        point="darkcyan",
        error="paleturquoise",
        maximum="mediumturquoise",
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
    maximum_seconds: float

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
    if value is None or value == "":
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
                maximum_seconds=float(row["maximum_seconds"]),
            )
            for row in reader
        ]


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


def series_label(row: StatRow, include_positive: bool) -> str:
    """Build a compact legend label for a row group."""
    if include_positive:
        sign = "pos" if row.positive else "neg"
        return f"{row.algorithm} ({sign})"
    return row.algorithm


def build_series(rows: Iterable[StatRow], axis: Axis, include_positive: bool) -> list[PlotSeries]:
    """Group rows into plot series."""
    grouped: dict[str, list[StatRow]] = {}
    for row in rows:
        grouped.setdefault(series_label(row, include_positive), []).append(row)

    return [
        PlotSeries(
            label=label,
            rows=tuple(sorted(group, key=lambda row: row.axis_value(axis))),
        )
        for label, group in sorted(grouped.items())
    ]


def configure_axis_constraints(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate fixed dimension filters for the selected x-axis."""
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
    if any(row.name is not None for row in rows):
        return args.title or "Benchmark runtimes"

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
    if args.title and context:
        return f"{args.title}\n{context}"
    if args.title:
        return args.title
    if context:
        return f"Benchmark runtimes\n{context}"
    return "Benchmark runtimes"


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
        colors_by_algorithm.setdefault(algorithm, COLOR_FAMILIES[color_index % len(COLOR_FAMILIES)])

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
        colors = COLOR_FAMILIES[color_index % len(COLOR_FAMILIES)]
        has_named_rows = any(row.name is not None for row in item.rows)
        x = jittered_axis_values(item.rows, axis) if has_named_rows else [row.axis_value(axis) for row in item.rows]
        mean = [row.mean_seconds for row in item.rows]
        lower_error = [min(row.stddev_seconds, row.mean_seconds) for row in item.rows]
        upper_error = [row.stddev_seconds for row in item.rows]
        maximum = [row.maximum_seconds for row in item.rows]

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
            label="_nolegend_" if has_named_rows else item.label,
            zorder=3,
        )
        ax.scatter(x, maximum, s=18, alpha=0.5, color=colors.maximum, marker="x", zorder=3)
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
    ax.grid(True, which="major", alpha=0.25)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Statistics CSV from benchmarks/run.py --stats.")
    parser.add_argument("--x", choices=("n", "k", "r", "d", "s"), required=True, help="Parameter used for the x-axis.")
    parser.add_argument("--output", type=Path, help="Where to save the diagram. Shows an interactive window if omitted.")
    parser.add_argument("--title", help="Optional diagram title.")
    parser.add_argument("--theory", action="store_true", help="Draw a faint fitted theoretical boundary function.")

    parser.add_argument("--algorithm", action="append", help="Algorithm to include. Can be passed multiple times.")
    parser.add_argument("--name", help="Case name to include.")
    parser.add_argument("--positive", choices=("true", "false", "all"), default="all", help="Filter positive/negative cases.")

    parser.add_argument("--n", type=int, help="Fix block length n.")
    parser.add_argument("--k", type=int, help="Fix logical dimension k.")
    parser.add_argument("--r", type=int, help="Fix redundancy r = n - k.")
    parser.add_argument("--density", type=float, help="Fix generated-code density.")
    parser.add_argument("--symmetry", type=float, help="Fix generated-code symmetry.")

    args = parser.parse_args(argv)
    configure_axis_constraints(args, parser)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the visualization CLI."""
    args = parse_args(argv)

    rows = [row for row in read_stats_csv(args.csv) if matches_filter(row, args)]
    if not rows:
        raise SystemExit("No rows matched the selected filters.")

    series = build_series(rows, axis=args.x, include_positive=args.positive == "all")
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
