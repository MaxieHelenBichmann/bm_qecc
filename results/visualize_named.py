"""Plot named benchmark cases as labeled runtime scatter plots."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from visual_utils import (
    AXIS_LABELS,
    COLOR_FAMILIES_NEG,
    COLOR_FAMILIES_POS,
    Axis,
    StatRow,
    add_common_stat_filters,
    configure_xaxis_ticks,
    draw_minute_guides,
    load_matplotlib_pyplot,
    positive_filter,
    read_stats_csv,
)


@dataclass(frozen=True)
class PlotSeries:
    """A single named-case scatter series."""

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
        min_gap = min(right - left for left, right in zip(unique_values, unique_values[1:]) if right > left)
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


def matches_filter(row: StatRow, args: argparse.Namespace) -> bool:
    """Return whether a named statistics row should be included."""
    if row.name is None:
        return False
    if args.algorithm and row.algorithm not in args.algorithm:
        return False
    if args.name is not None and row.name != args.name:
        return False
    selected_positive = positive_filter(args.positive)
    if selected_positive is not None and row.positive != selected_positive:
        return False
    return True


def series_label(row: StatRow, include_positive: bool, include_algorithm: bool) -> str:
    """Build a compact legend label for a named row group."""
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


def plot_named_series(series: Sequence[PlotSeries], axis: Axis, output: Path | None, title: str | None) -> None:
    """Render named cases with mean/stddev, maximum markers, and direct labels."""
    plt = load_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    point_labels: list[PointLabel] = []
    for color_index, item in enumerate(series):
        if item.rows and item.rows[0].positive:
            colors = COLOR_FAMILIES_POS[color_index % len(COLOR_FAMILIES_POS)]
        else:
            colors = COLOR_FAMILIES_NEG[color_index % len(COLOR_FAMILIES_NEG)]

        x = jittered_axis_values(item.rows, axis)
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
            linestyle="none",
            linewidth=1.6,
            markersize=4.5,
            label=item.label,
            zorder=3,
        )
        if maximum_points:
            maximum_x, maximum = zip(*maximum_points)
            ax.scatter(maximum_x, maximum, s=18, alpha=0.5, color=colors.maximum, marker="x", zorder=3)
        for row, label_x in zip(item.rows, x):
            if row.name is None:
                continue
            point_labels.append(PointLabel(text=row.name, x=label_x, y=row.mean_seconds, color=colors.point))

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
    parser.add_argument("csv", type=Path, help="Statistics CSV from benchmarks/run.py --stats.")
    parser.add_argument("--x", choices=("n", "k", "r"), required=True, help="Parameter used for the x-axis.")
    parser.add_argument("--output", type=Path, help="Where to save the diagram. Shows an interactive window if omitted.")
    parser.add_argument("--title", help="Optional diagram title.")
    parser.add_argument("--name", help="Case name to include.")
    add_common_stat_filters(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named-code visualization CLI."""
    args = build_parser().parse_args(argv)
    all_rows = read_stats_csv(args.csv)
    if not all_rows:
        raise SystemExit("No rows found in the statistics CSV.")

    rows = [row for row in all_rows if matches_filter(row, args)]
    if not rows:
        raise SystemExit("No named rows matched the selected filters.")

    series = build_series(
        rows,
        axis=args.x,
        include_positive=args.positive == "all",
        include_algorithm=len({row.algorithm for row in rows}) > 1,
    )
    plot_named_series(series, axis=args.x, output=args.output, title=args.title or "Named benchmark runtimes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
