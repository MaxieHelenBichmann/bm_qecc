"""Plot randomized benchmark statistics as continuous runtime curves."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from visual_utils import (
    AXIS_LABELS,
    COLOR_FAMILIES_NEG,
    COLOR_FAMILIES_POS,
    PlotColors,
    Axis,
    StatRow,
    add_common_stat_filters,
    configure_xaxis_ticks,
    draw_minute_guides,
    load_matplotlib_pyplot,
    positive_filter,
    read_stats_csv,
    stat_file_kind,
)


@dataclass(frozen=True)
class PlotSeries:
    """A single randomized-code line series."""

    label: str
    rows: tuple[StatRow, ...]


def boundary_functions(algorithm: str) -> Callable | None:
    """Return a coarse theoretical boundary function for known algorithm families."""
    def _permutation(n, a, b):
        import numpy as np
        from scipy.special import gammaln

        return a * np.exp(gammaln(np.asarray(n, dtype=float) + 1.0)) + b

    def _lc(n, a, b):
        import numpy as np

        return a * np.power(6.0, np.asarray(n, dtype=float)) + b

    if algorithm.startswith("pm_"):
        return _permutation
    if algorithm.startswith("lc_"):
        return _lc
    return None


def initial_fit_guess(boundary_function: Callable, x, y) -> tuple[float, float]:
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
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "scipy"}:
            msg = f"{exc.name} is required for --theory. Install dependencies with `python3 -m pip install -r requirements.txt`."
            raise SystemExit(msg) from exc
        raise

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


def matches_filter(row: StatRow, args: argparse.Namespace) -> bool:
    """Return whether a randomized statistics row should be included."""
    if row.name is not None:
        return False
    if args.x == "d" and row.density is None:
        return False
    if args.x == "s" and row.symmetry is None:
        return False
    if args.algorithm and row.algorithm not in args.algorithm:
        return False
    selected_positive = positive_filter(args.positive)
    if selected_positive is not None and row.positive != selected_positive:
        return False
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
    """Build a compact legend label for a randomized row group."""
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
        PlotSeries(label=label, rows=tuple(sorted(group, key=lambda row: row.axis_value(axis))))
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
    algorithms = sorted({row.algorithm for row in rows})
    single_algorithm = algorithms[0] if len(algorithms) == 1 else None
    title = args.title or "Randomized benchmark runtimes"
    if single_algorithm is not None:
        title = f"{title}: {single_algorithm}"

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


def plot_random_series(
    series: Sequence[PlotSeries],
    axis: Axis,
    output: Path | None,
    title: str | None,
    show_theory: bool,
) -> None:
    """Render randomized-code curves with mean/stddev and maximum markers."""
    plt = load_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    if show_theory:
        plot_boundary_fits(series, axis, ax)

    for color_index, item in enumerate(series):
        if item.rows and item.rows[0].positive:
            colors = COLOR_FAMILIES_POS[color_index % len(COLOR_FAMILIES_POS)]
        else:
            colors = COLOR_FAMILIES_NEG[color_index % len(COLOR_FAMILIES_NEG)]

        x = [row.axis_value(axis) for row in item.rows]
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
            linestyle="-",
            linewidth=1.6,
            markersize=4.5,
            label=item.label,
            zorder=3,
        )
        if maximum_points:
            maximum_x, maximum = zip(*maximum_points)
            ax.scatter(maximum_x, maximum, s=18, alpha=0.5, color=colors.maximum, marker="x", zorder=3)

    ax.set_xlabel(AXIS_LABELS[axis])
    ax.set_ylabel("runtime [s]")
    ax.margins(x=0.12, y=0.18)
    ax.set_ylim(bottom=0)
    configure_xaxis_ticks(axis, ax)
    draw_minute_guides(ax)
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
    parser.add_argument("csv", type=Path, help="Statistics CSV from benchmarks/run.py --stats.")
    parser.add_argument("--x", choices=("n", "k", "r", "d", "s"), required=True, help="Parameter used for the x-axis.")
    parser.add_argument("--output", type=Path, help="Where to save the diagram. Shows an interactive window if omitted.")
    parser.add_argument("--title", help="Optional diagram title.")
    parser.add_argument("--theory", action="store_true", help="Draw a faint fitted theoretical boundary function.")
    add_common_stat_filters(parser)
    parser.add_argument("--n", type=int, help="Fix block length n.")
    parser.add_argument("--k", type=int, help="Fix logical dimension k.")
    parser.add_argument("--r", type=int, help="Fix redundancy r = n - k.")
    parser.add_argument("--density", type=float, help="Fix generated-code density.")
    parser.add_argument("--symmetry", type=float, help="Fix generated-code symmetry.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the randomized-code visualization CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_axis_constraints(args, parser)

    all_rows = read_stats_csv(args.csv)
    if not all_rows:
        raise SystemExit("No rows found in the statistics CSV.")
    if stat_file_kind(all_rows) == "named":
        raise SystemExit("This CSV contains only named rows. Use results/visualize_named.py instead.")

    rows = [row for row in all_rows if matches_filter(row, args)]
    if not rows:
        raise SystemExit("No randomized rows matched the selected filters.")

    series = build_series(
        rows,
        axis=args.x,
        include_positive=args.positive == "all",
        include_algorithm=len({row.algorithm for row in rows}) > 1,
    )
    plot_random_series(
        series,
        axis=args.x,
        output=args.output,
        title=fixed_parameter_title(args, rows),
        show_theory=args.theory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
