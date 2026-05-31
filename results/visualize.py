"""Create matplotlib plots from benchmark statistics CSV files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Axis = Literal["n", "k", "r", "d", "s"]

AXIS_LABELS: dict[Axis, str] = {
    "n": "n [physical qubits]",
    "k": "k [logical qubits]",
    "r": "r = n - k [number of stabilizer generators]",
    "d": "d [density]",
    "s": "s [symmetry]",
}


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


def fixed_parameter_title(args: argparse.Namespace) -> str:
    """Build a title line that describes fixed dimension parameters."""
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


def plot_series(series: Sequence[PlotSeries], axis: Axis, output: Path | None, title: str | None) -> None:
    """Render the selected series with mean/stddev and maximum markers."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit("matplotlib is required. Install it with `python3 -m pip install -r requirements.txt`.") from exc
        raise

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

    for color_index, item in enumerate(series):
        colors = COLOR_FAMILIES[color_index % len(COLOR_FAMILIES)]
        x = [row.axis_value(axis) for row in item.rows]
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
            linewidth=1.6,
            markersize=4.5,
            label=item.label,
        )
        ax.scatter(x, maximum, s=18, alpha=0.5, color=colors.maximum, marker="x")

    ax.set_xlabel(AXIS_LABELS[axis])
    ax.set_ylabel("runtime [s]")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend()
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
    plot_series(series, axis=args.x, output=args.output, title=fixed_parameter_title(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
