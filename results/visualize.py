"""Create matplotlib plots from benchmark statistics CSV files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Axis = Literal["n", "k", "r"]


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

    def axis_value(self, axis: Axis) -> int:
        if axis == "n":
            return self.n
        if axis == "k":
            return self.k
        return self.r


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


def plot_series(series: Sequence[PlotSeries], axis: Axis, output: Path | None, title: str | None) -> None:
    """Render the selected series with mean/stddev and maximum markers."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit("matplotlib is required. Install it with `python3 -m pip install -r requirements.txt`.") from exc
        raise

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

    for item in series:
        x = [row.axis_value(axis) for row in item.rows]
        mean = [row.mean_seconds for row in item.rows]
        stddev = [row.stddev_seconds for row in item.rows]
        maximum = [row.maximum_seconds for row in item.rows]

        line = ax.errorbar(
            x,
            mean,
            yerr=stddev,
            marker="o",
            capsize=3,
            linewidth=1.6,
            markersize=4.5,
            label=item.label,
        )
        color = line.lines[0].get_color()
        ax.scatter(x, maximum, s=18, alpha=0.35, color=color, marker="x")

    ax.set_xlabel(axis if axis != "r" else "r = n - k")
    ax.set_ylabel("runtime [s]")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend()
    if title:
        ax.set_title(title)

    if output is None:
        plt.show()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Statistics CSV from benchmarks/run.py --stats.")
    parser.add_argument("--x", choices=("n", "k", "r"), required=True, help="Dimension used for the x-axis.")
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
    plot_series(series, axis=args.x, output=args.output, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
