"""Shared helpers for benchmark visualization scripts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Axis = Literal["n", "k", "r", "d", "s"]
StatFileKind = Literal["randomized", "named", "mixed"]

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


COLOR_FAMILIES_POS: tuple[PlotColors, ...] = (
    PlotColors("cornflowerblue", "royalblue", "lightsteelblue", "lightskyblue"),
    PlotColors("forestgreen", "darkgreen", "palegreen", "mediumseagreen"),
    PlotColors("mediumpurple", "indigo", "thistle", "plum"),
    PlotColors("lightseagreen", "darkcyan", "paleturquoise", "mediumturquoise"),
)
COLOR_FAMILIES_NEG: tuple[PlotColors, ...] = (
    PlotColors("orange", "darkorange", "moccasin", "peru"),
    PlotColors("indianred", "darkred", "mistyrose", "salmon"),
    PlotColors("peru", "saddlebrown", "tan", "burlywood"),
    PlotColors("hotpink", "mediumvioletred", "pink", "palevioletred"),
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
    """Read a statistics CSV written by benchmarks.run.write_stats."""
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


def configure_xaxis_ticks(axis: Axis, ax) -> None:
    """Keep discrete dimension axes labeled as integers."""
    if axis not in {"n", "k", "r"}:
        return

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))


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


def positive_filter(value: str) -> bool | None:
    """Return a bool for positive filters, or None for all cases."""
    if value == "all":
        return None
    return value == "true"


def add_common_stat_filters(parser: argparse.ArgumentParser) -> None:
    """Add filters shared by named and randomized statistics plots."""
    parser.add_argument("--algorithm", action="append", help="Algorithm to include. Can be passed multiple times.")
    parser.add_argument("--positive", choices=("true", "false", "all"), default="all", help="Filter positive/negative cases.")


def load_matplotlib_pyplot():
    """Import matplotlib lazily and give a helpful install hint."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            msg = "matplotlib is required. Install it with `python3 -m pip install -r requirements.txt`."
            raise SystemExit(msg) from exc
        raise
    return plt
