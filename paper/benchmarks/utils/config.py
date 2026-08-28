"""Shared constants and tiny CSV helpers for the paper collectors."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.experiments.run import RunResult
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.thesis.thesis_prototypes import measurement_dimensions

ROOT = Path(__file__).resolve().parents[3]
COLLECTED_DATA_DIR = ROOT / "paper" / "data" / "collected"
ALGORITHM_DATA_DIR = COLLECTED_DATA_DIR / "algorithms"

MASTER_SEED = 42
NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
DIMENSIONS = tuple(measurement_dimensions())
TIMEOUT_SECONDS = 5_400.0
CERTIFICATION_TIMEOUT_SECONDS = 600.0
MEMORY_LIMIT_BYTES = 13 * 1024**3


def execution_status(result: RunResult) -> str:
    """Describe execution independently of whether a Boolean result is true."""
    if result.timed_out:
        return "timeout"
    if result.memory_exceeded:
        return "memory_limited"
    if result.error is not None:
        return "error"
    return "success"


def csv_key(*values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def completed_csv_keys(
    path: Path,
    key_fields: Sequence[str],
) -> set[tuple[str, ...]]:
    """Return result keys already present in an append-only collector CSV."""
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(key_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} has an incompatible header; missing {sorted(missing)}"
            )
        return {
            tuple(row[field] for field in key_fields)
            for row in reader
            if all(row.get(field) is not None for field in key_fields)
        }


def append_csv_row(
    path: Path,
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    """Persist one result immediately, adding the header to a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
