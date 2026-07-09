"""Plot randomized benchmark statistics as n-by-k/n-by-r runtime heatmaps."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt

from visual_utils import (
    StatMetadata,
    StatRow,
    add_common_stat_filters,
    add_x_range_args,
    metadata_title_suffix,
    positive_filter,
    read_stats_csv_with_metadata,
    stat_file_kind,
    validate_x_range_args,
    x_in_range,
)

VerticalAxis = Literal["k", "r"]
AggregateMode = Literal["mean", "min", "max"]


def common_metadata(metadata: Sequence[StatMetadata]) -> StatMetadata:
    """Keep only run metadata values shared by every input CSV."""
    def shared(attribute: str):
        values = {getattr(item, attribute) for item in metadata}
        return next(iter(values)) if len(values) == 1 else None

    return StatMetadata(
        seed=shared("seed"),
        timeout_seconds=shared("timeout_seconds"),
        memory_limit_bytes=shared("memory_limit_bytes"),
    )


def vertical_value(row: StatRow, axis: VerticalAxis) -> int:
    """Return the selected vertical code dimension."""
    if axis == "k":
        return row.k
    return row.r


def in_optional_range(value: int, minimum: int | None, maximum: int | None) -> bool:
    """Return whether an integer lies within an optional inclusive range."""
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def matches_filter(row: StatRow, args: argparse.Namespace) -> bool:
    """Return whether a randomized statistics row should be included."""
    if row.name is not None:
        return False
    if args.algorithm and row.algorithm not in args.algorithm:
        return False
    selected_positive = positive_filter(args.positive)
    if selected_positive is not None and row.positive != selected_positive:
        return False
    if args.density is not None and row.density != args.density:
        return False
    if args.symmetry is not None and row.symmetry != args.symmetry:
        return False
    if not x_in_range(float(row.n), args.xmin, args.xmax):
        return False
    return in_optional_range(vertical_value(row, args.y), args.ymin, args.ymax)


def aggregate_values(values: Sequence[float], mode: AggregateMode) -> float:
    """Aggregate one or more runtime values for the same heatmap cell."""
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    if mode == "min":
        return min(finite_values)
    if mode == "max":
        return max(finite_values)
    return mean(finite_values)


def build_heatmap_grid(
    rows: Iterable[StatRow],
    y_axis: VerticalAxis,
    aggregate: AggregateMode,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Build a dense matrix whose cells contain aggregated mean runtime."""
    cell_values: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        cell_values.setdefault((row.n, vertical_value(row, y_axis)), []).append(row.mean_seconds)

    x_values = sorted({n for n, _ in cell_values})
    y_values = sorted({y for _, y in cell_values})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}

    grid = [[float("nan") for _ in x_values] for _ in y_values]
    for (n, y), values in cell_values.items():
        grid[y_index[y]][x_index[n]] = aggregate_values(values, aggregate)
    return x_values, y_values, grid


def heatmap_title(args: argparse.Namespace, rows: Sequence[StatRow], metadata_suffix: str = "") -> str:
    """Build a compact title for the heatmap."""
    algorithms = sorted({row.algorithm for row in rows})
    title = "Random codes benchmark heatmap"
    if len(algorithms) == 1:
        title = f"{title}: {algorithms[0]}"

    context_parts = [f"value = {args.aggregate} mean runtime per cell"]
    if args.positive != "all":
        context_parts.append(f"positive = {args.positive}")
    if args.density is not None:
        context_parts.append(f"d = {args.density:g}")
    if args.symmetry is not None:
        context_parts.append(f"s = {args.symmetry:g}")
    if metadata_suffix:
        context_parts.append(metadata_suffix)
    return "\n".join([title, ", ".join(context_parts)])


def plot_heatmap(
    x_values: Sequence[int],
    y_values: Sequence[int],
    grid: Sequence[Sequence[float]],
    y_axis: VerticalAxis,
    output: Path | None,
    title: str,
    log_scale: bool,
) -> None:
    """Render a mean-runtime heatmap."""
    try:
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            msg = "numpy is required for heatmaps. Install dependencies with `python3 -m pip install -r requirements.txt`."
            raise SystemExit(msg) from exc
        raise

    values = np.ma.masked_invalid(np.asarray(grid, dtype=float))
    if values.count() == 0:
        raise SystemExit("No finite runtime values matched the selected filters.")

    norm = None
    if log_scale:
        positive_values = values[values > 0]
        if positive_values.count() == 0:
            raise SystemExit("--log requires at least one positive finite runtime value.")
        values = np.ma.masked_where(values <= 0, values)
        norm = LogNorm(vmin=float(positive_values.min()), vmax=float(positive_values.max()))

    fig_width = max(7.0, min(14.0, 0.45 * len(x_values) + 3.0))
    fig_height = max(5.0, min(12.0, 0.45 * len(y_values) + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(value) for value in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(value) for value in y_values])
    ax.set_xlabel("n [physical qubits]")
    ax.set_ylabel("k [logical qubits]" if y_axis == "k" else "r = n - k [number of stabilizer generators]")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mean runtime [s]")

    ax.set_xticks([tick - 0.5 for tick in range(1, len(x_values))], minor=True)
    ax.set_yticks([tick - 0.5 for tick in range(1, len(y_values))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    if output is None:
        plt.show()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved heatmap to {output}.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="One or more statistics CSVs from benchmarks/run.py --stats.",
    )
    parser.add_argument(
        "--y",
        choices=("k", "r"),
        default="k",
        help="Vertical code dimension. The horizontal dimension is always n.",
    )
    add_x_range_args(parser)
    parser.add_argument("--ymin", type=int, help="Only include vertical values >= MIN.")
    parser.add_argument("--ymax", type=int, help="Only include vertical values <= MAX.")
    parser.add_argument(
        "--aggregate",
        choices=("mean", "min", "max"),
        default="mean",
        help="How to combine multiple rows that land in the same (n, y) cell.",
    )
    parser.add_argument("--density", type=float, help="Fix generated-code density.")
    parser.add_argument("--symmetry", type=float, help="Fix generated-code symmetry.")
    parser.add_argument("--log", action="store_true", help="Use logarithmic color scaling.")
    parser.add_argument("--output", type=Path, help="Where to save the diagram. Shows an interactive window if omitted.")
    add_common_stat_filters(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the randomized-code heatmap visualization CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_x_range_args(parser, args.xmin, args.xmax, tuple_expected=False)
    if args.ymin is not None and args.ymax is not None and args.ymin > args.ymax:
        parser.error("--ymin must be <= --ymax")

    stat_csvs = [read_stats_csv_with_metadata(path) for path in args.csv]
    empty_paths = [path for path, stat_csv in zip(args.csv, stat_csvs) if not stat_csv.rows]
    if empty_paths:
        raise SystemExit(f"No rows found in: {', '.join(map(str, empty_paths))}")
    named_paths = [
        path
        for path, stat_csv in zip(args.csv, stat_csvs)
        if stat_file_kind(stat_csv.rows) == "named"
    ]
    if named_paths:
        raise SystemExit(
            f"These CSVs contain only named rows: {', '.join(map(str, named_paths))}. "
            "Heatmaps currently use randomized benchmark rows."
        )

    all_rows = [row for stat_csv in stat_csvs for row in stat_csv.rows]
    rows = [row for row in all_rows if matches_filter(row, args)]
    if not rows:
        raise SystemExit("No randomized rows matched the selected filters.")

    metadata = common_metadata([stat_csv.metadata for stat_csv in stat_csvs])
    x_values, y_values, grid = build_heatmap_grid(rows, args.y, args.aggregate)
    plot_heatmap(
        x_values=x_values,
        y_values=y_values,
        grid=grid,
        y_axis=args.y,
        output=args.output,
        title=heatmap_title(args, rows, metadata_title_suffix(metadata)),
        log_scale=args.log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
