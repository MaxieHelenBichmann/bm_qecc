"""Plot seeded invariant benchmark statistics as runtime curves."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from visual_utils import (
    AXIS_LABELS,
    MEMORY_LIMIT_COLOR,
    Axis,
    StatCsv,
    StatRow,
    configure_xaxis_ticks,
    draw_runtime_guides,
    metadata_title_suffix,
    positive_filter,
    read_stats_csv_with_metadata,
)

GRID_MAX_SECONDS = 5 * 60


def invariant_algorithm_base(algorithm: str) -> str:
    """Return the base invariant name without a trailing subset-size suffix."""
    return re.sub(r"_s\d+$", "", algorithm)


def invariant_algorithm_variant(algorithm: str) -> int | None:
    """Return the subset-size suffix of an invariant algorithm, if present."""
    match = re.search(r"_s(\d+)$", algorithm)
    return None if match is None else int(match.group(1))


def invariant_marker(algorithm: str) -> str:
    """Return a marker shape that distinguishes full and subset-sized variants."""
    variant = invariant_algorithm_variant(algorithm)
    if variant is None:
        return "o"
    markers = ("s", "^", "D", "P", "v", "X")
    return markers[(variant - 1) % len(markers)]


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
    return red, green, blue, alpha


def axis_limit(value: str) -> tuple[float, float]:
    """Parse a min,max x-axis data limit."""
    raw = value.strip().removeprefix("(").removesuffix(")")
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("expected MIN,MAX or (MIN,MAX)")
    try:
        lower, upper = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("x-axis limits must be numeric") from exc
    if lower > upper:
        raise argparse.ArgumentTypeError("minimum x-axis limit must be <= maximum")
    return lower, upper


def matches_filter(row: StatRow, args: argparse.Namespace) -> bool:
    """Return whether an invariant statistics row should be plotted."""
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
    if args.xlim is not None:
        lower, upper = args.xlim
        if not lower <= row.axis_value(args.x) <= upper:
            return False
    return True


def uses_dimension_case_axis(args: argparse.Namespace) -> bool:
    """Return whether n/r needs a categorical axis to distinguish dimensions."""
    if args.x == "n":
        return args.k is None and args.r is None
    if args.x == "r":
        return args.n is None and args.k is None
    return False


def dimension_case_key(row: StatRow, axis: Axis) -> tuple[int, int]:
    """Return a stable composite dimension key."""
    if axis == "n":
        return row.n, row.r
    if axis == "r":
        return row.r, row.n
    raise ValueError("Composite case axes only support n or r.")


def dimension_case_positions(rows: Sequence[StatRow], axis: Axis) -> dict[tuple[int, int], int]:
    """Map every distinct dimension case to a categorical x-position."""
    keys = sorted({dimension_case_key(row, axis) for row in rows})
    return {key: index for index, key in enumerate(keys, start=1)}


def plot_title(args: argparse.Namespace, rows: Sequence[StatRow], stat_csv: StatCsv) -> str:
    """Build an informative title from family, filters, and run metadata."""
    families = {"PM" if row.algorithm.startswith("pm_") else "LC" for row in rows}
    family = next(iter(families)) if len(families) == 1 else None
    title = f"{family} invariant benchmarks" if family else "Invariant benchmarks"

    algorithms = sorted({row.algorithm for row in rows})
    if len(algorithms) == 1:
        title += f": {algorithms[0]}"

    fixed = []
    if args.n is not None and args.x != "n":
        fixed.append(f"n = {args.n}")
    if args.k is not None and args.x != "k":
        fixed.append(f"k = {args.k}")
    if args.r is not None and args.x != "r":
        fixed.append(f"r = {args.r}")
    if args.positive != "all":
        fixed.append("positive cases" if args.positive == "true" else "negative cases")

    lines = [title]
    if fixed:
        lines.append(", ".join(fixed))
    metadata = metadata_title_suffix(stat_csv.metadata)
    if metadata:
        lines.append(metadata)
    return "\n".join(lines)


def plot_invariant_rows(
    rows: Sequence[StatRow],
    axis: Axis,
    output: Path | None,
    title: str,
    timeout_seconds: float | None,
    case_positions: dict[tuple[int, int], int] | None,
) -> None:
    """Render invariant statistics with uncertainty and failure annotations."""
    algorithms = sorted({row.algorithm for row in rows})
    bases = sorted({invariant_algorithm_base(algorithm) for algorithm in algorithms})
    cmap = plt.get_cmap("tab10" if len(bases) <= 10 else "tab20")
    base_colors = {base: cmap(index % cmap.N) for index, base in enumerate(bases)}
    colors = {
        algorithm: invariant_tone(base_colors[invariant_algorithm_base(algorithm)], algorithm)
        for algorithm in algorithms
    }

    grouped: dict[tuple[str, bool], list[StatRow]] = {}
    for row in rows:
        grouped.setdefault((row.algorithm, row.positive), []).append(row)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    has_memory = False
    has_memory_only = False
    has_maximum = False
    for (algorithm, positive), group in sorted(grouped.items()):
        sort_key = (lambda row: dimension_case_key(row, axis)) if case_positions else (lambda row: row.axis_value(axis))
        group = sorted(group, key=sort_key)
        if case_positions:
            x = [case_positions[dimension_case_key(row, axis)] for row in group]
        else:
            x = [row.axis_value(axis) for row in group]

        finite = [(label_x, row) for label_x, row in zip(x, group) if math.isfinite(row.mean_seconds)]
        label = f"{algorithm} ({'positive' if positive else 'negative'})"
        color = colors[algorithm]
        if finite:
            finite_x = [label_x for label_x, _ in finite]
            mean = [row.mean_seconds for _, row in finite]
            lower = [min(row.stddev_seconds, row.mean_seconds) for _, row in finite]
            upper = [row.stddev_seconds for _, row in finite]
            ax.errorbar(
                finite_x,
                mean,
                yerr=[lower, upper],
                marker=invariant_marker(algorithm),
                markerfacecolor=color if positive else "white",
                markeredgecolor=color,
                color=color,
                ecolor=(*color[:3], 0.35),
                capsize=3,
                linewidth=1.4,
                markersize=5,
                linestyle="-" if positive else "--",
                label=label,
                zorder=3,
            )

            maximum = [(label_x, row.maximum_seconds) for label_x, row in finite if row.maximum_seconds is not None]
            if maximum:
                maximum_x, maximum_y = zip(*maximum)
                ax.scatter(maximum_x, maximum_y, marker="x", s=20, color=color, alpha=0.55, zorder=3)
                has_maximum = True

            memory = [(label_x, row.mean_seconds) for label_x, row in finite if row.num_memory_limited > 0]
            if memory:
                memory_x, memory_y = zip(*memory)
                ax.scatter(
                    memory_x,
                    memory_y,
                    s=115,
                    facecolors="none",
                    edgecolors=MEMORY_LIMIT_COLOR,
                    linewidths=1.8,
                    marker="o",
                    zorder=4,
                )
                has_memory = True

        memory_only = [
            label_x for label_x, row in zip(x, group)
            if row.num_memory_limited > 0 and not math.isfinite(row.mean_seconds)
        ]
        for label_x in memory_only:
            ax.axvline(label_x, color=MEMORY_LIMIT_COLOR, linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)
            has_memory_only = True

    if case_positions:
        ordered = sorted(case_positions.items(), key=lambda item: item[1])
        labels = []
        previous: int | None = None
        for (first, _), _ in ordered:
            labels.append(str(first) if first != previous else "")
            previous = first
        ax.set_xticks([position for _, position in ordered])
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(f"case ({axis}, {'r' if axis == 'n' else 'n'}), sorted by {axis}")
    else:
        ax.set_xlabel(AXIS_LABELS[axis])
        configure_xaxis_ticks(axis, ax)

    ax.set_ylabel("runtime [s]")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08, y=0.16)
    draw_runtime_guides(ax, timeout_seconds)
    ax.grid(which="major", alpha=0.15) if ax.get_ylim()[1] <= GRID_MAX_SECONDS else ax.grid(False)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if has_maximum:
        handles.append(Line2D([], [], marker="x", color="gray", linestyle="none", label="maximum runtime"))
        labels.append("maximum runtime")
    if has_memory:
        handles.append(Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=MEMORY_LIMIT_COLOR,
                              linestyle="none", markersize=8, label="memory limited"))
        labels.append("memory limited")
    if has_memory_only:
        handles.append(Line2D([], [], color=MEMORY_LIMIT_COLOR, linestyle="--", linewidth=1.8,
                              label="memory limited (no runtime)"))
        labels.append("memory limited (no runtime)")
    if timeout_seconds is not None and ax.get_ylim()[0] <= timeout_seconds <= ax.get_ylim()[1]:
        handles.append(Line2D([], [], color="#e4572e", linestyle="-", linewidth=1.4,
                              label=f"timeout ({timeout_seconds:g} s)"))
        labels.append(f"timeout ({timeout_seconds:g} s)")
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved diagram to {output}.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Invariant statistics CSV from benchmarks/run.py --inv.")
    parser.add_argument("--x", choices=("n", "k", "r"), required=True, help="Parameter used for the x-axis.")
    parser.add_argument("--xlim", type=axis_limit, help="Only include x-axis values in MIN,MAX.")
    parser.add_argument("--algorithm", action="append", help="Invariant to include. Can be passed multiple times.")
    parser.add_argument("--positive", choices=("true", "false", "all"), default="all")
    parser.add_argument("--n", type=int, help="Fix block length n.")
    parser.add_argument("--k", type=int, help="Fix logical dimension k.")
    parser.add_argument("--r", type=int, help="Fix redundancy r = n - k.")
    parser.add_argument("--output", type=Path, help="Save the diagram instead of showing it interactively.")
    parser.add_argument("--title", help="Override the automatically generated title.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the invariant visualization CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.x == "k" and args.n is None:
        parser.error("--x k requires --n so only one block length is plotted.")
    if args.x == "r" and args.n is not None and args.k is not None:
        parser.error("--x r accepts at most one of --n or --k.")

    stat_csv = read_stats_csv_with_metadata(args.csv)
    if not stat_csv.rows:
        raise SystemExit("No rows found in the invariant statistics CSV.")
    rows = [row for row in stat_csv.rows if matches_filter(row, args)]
    if not rows:
        raise SystemExit("No invariant rows matched the selected filters.")

    positions = dimension_case_positions(rows, args.x) if uses_dimension_case_axis(args) else None
    title = args.title or plot_title(args, rows, stat_csv)
    plot_invariant_rows(rows, args.x, args.output, title, stat_csv.metadata.timeout_seconds, positions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
