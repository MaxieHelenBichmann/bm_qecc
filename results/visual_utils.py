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
class StatMetadata:
    """Metadata stored in the first row of benchmark statistics CSVs."""

    seed: int | None
    timeout_seconds: float | None
    memory_limit_bytes: int | None


@dataclass(frozen=True)
class StatCsv:
    """Parsed statistics CSV contents."""

    metadata: StatMetadata
    rows: list["StatRow"]


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
    num_cases: int
    num_successful: int
    num_timeouts: int
    num_memory_limited: int

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


def parse_optional_int(value: str | None) -> int | None:
    """Parse an optional CSV integer field."""
    if value is None or value == "" or value.strip().lower() == "none":
        return None
    return int(value)


def parse_stat_metadata(line: str, header: Sequence[str]) -> StatMetadata:
    """Parse optional metadata from the first statistics CSV row."""
    first_row = [part.strip() for part in line.strip().split(",")]
    if (first_row and first_row[0] == "algorithm") or (header and header[0] != "algorithm"):
        return StatMetadata(seed=None, timeout_seconds=None, memory_limit_bytes=None)

    return StatMetadata(
        seed=parse_optional_int(first_row[0]) if len(first_row) >= 1 else None,
        timeout_seconds=parse_optional_float(first_row[1]) if len(first_row) >= 2 else None,
        memory_limit_bytes=parse_optional_int(first_row[2]) if len(first_row) >= 3 else None,
    )


def read_stats_csv_with_metadata(path: Path) -> StatCsv:
    """Read a statistics CSV written by benchmarks.run.write_stats."""
    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            return StatCsv(StatMetadata(None, None, None), [])

        sample = file.readline()
        if not sample:
            return StatCsv(parse_stat_metadata(first_line, []), [])

        header = sample.strip().split(",")
        metadata = parse_stat_metadata(first_line, header)
        first_row = first_line.strip().split(",")
        if first_row and first_row[0] == "algorithm":
            file.seek(0)
        else:
            file.seek(0)
            next(file)

        reader = csv.DictReader(file)
        return StatCsv(
            metadata=metadata,
            rows=[
                StatRow(
                    algorithm=row["algorithm"],
                    name=row["name"] or None,
                    n=int(row["n"]),
                    k=int(row["k"]),
                    positive=parse_bool(row["positive"]),
                    density=parse_optional_float(row.get("density")),
                    symmetry=parse_optional_float(row.get("symmetry")),
                    mean_seconds=parse_optional_float(row.get("mean_seconds")) or float("nan"),
                    stddev_seconds=parse_optional_float(row.get("stddev_seconds")) or 0.0,
                    maximum_seconds=parse_optional_float(row.get("maximum_seconds")),
                    num_cases=int(row.get("num_cases") or row.get("num_times") or 0),
                    num_successful=int(row.get("num_successful") or row.get("num_times") or 0),
                    num_timeouts=int(row.get("num_timeouts") or 0),
                    num_memory_limited=int(row.get("num_memory_limited") or 0),
                )
                for row in reader
            ],
        )


def read_stats_csv(path: Path) -> list[StatRow]:
    """Read statistics rows from a CSV written by benchmarks.run.write_stats."""
    return read_stats_csv_with_metadata(path).rows


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


def draw_runtime_guides(ax, timeout_seconds: float | None = None) -> None:
    """Draw runtime guide lines without changing the data-driven axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    first_minute = 60
    guide_ceiling = ylim[1] if timeout_seconds is None else min(ylim[1], timeout_seconds)
    last_minute = int(guide_ceiling // 60) * 60

    for seconds in range(first_minute, last_minute + 1, 60):
        if seconds <= ylim[0] or (timeout_seconds is not None and seconds >= timeout_seconds):
            continue
        is_hour = seconds % 3600 == 0
        ax.axhline(
            seconds,
            color="#7c746e" if is_hour else "#9a9a9a",
            linestyle=":",
            linewidth=1.1 if is_hour else 0.45,
            alpha=0.38 if is_hour else 0.24,
            label="_nolegend_",
            zorder=0,
        )

    if timeout_seconds is not None and ylim[0] <= timeout_seconds <= ylim[1]:
        ax.axhline(
            timeout_seconds,
            color="#e4572e",
            linestyle="-",
            linewidth=1.4,
            alpha=0.85,
            label="_nolegend_",
            zorder=1,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def draw_minute_guides(ax) -> None:
    """Draw minute and hour runtime guides without changing the data-driven axes."""
    draw_runtime_guides(ax)


def format_memory_limit(memory_limit_bytes: int | None) -> str | None:
    """Format a byte count in GiB for plot titles."""
    if memory_limit_bytes is None:
        return None

    gib_value = float(memory_limit_bytes) / (1024.0 ** 3)
    return f"{gib_value:.3g} GiB"


def metadata_title_suffix(metadata: StatMetadata) -> str:
    """Build a compact title suffix for benchmark run metadata."""
    parts: list[str] = []
    if metadata.seed is not None:
        parts.append(f"seed = {metadata.seed}")
    memory_limit = format_memory_limit(metadata.memory_limit_bytes)
    if memory_limit is not None:
        parts.append(f"max memory = {memory_limit}")
    return ", ".join(parts)


def positive_filter(value: str) -> bool | None:
    """Return a bool for positive filters, or None for all cases."""
    if value == "all":
        return None
    return value == "true"


def add_common_stat_filters(parser: argparse.ArgumentParser) -> None:
    """Add filters shared by named and randomized statistics plots."""
    parser.add_argument("--algorithm", action="append", help="Algorithm to include. Can be passed multiple times.")
    parser.add_argument("--positive", choices=("true", "false", "all"), default="all", help="Filter positive/negative cases.")
