"""Shared CSV operations for the six paper experiment extractors."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COLLECTED_DATA_DIR = ROOT / "paper" / "data" / "collected"
ALGORITHM_DATA_DIR = COLLECTED_DATA_DIR / "algorithms"
RESULTS_DIR = ROOT / "paper" / "results"


def read_csv(path: Path, required: Sequence[str] = ()) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing collected data: {path}; run its paper collector first "
            "(see the collector table in paper/README.md)"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str | bool | None) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def as_int(value: str | int | None) -> int:
    return int(value or 0)


def as_float(value: str | float | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def problem_for_algorithm(algorithm: str) -> str:
    for problem in ("pm_stb", "pm_css", "lc_stb"):
        if algorithm.startswith(f"{problem}_"):
            return problem
    raise ValueError(f"cannot infer problem family from algorithm {algorithm!r}")


STAT_REQUIRED = (
    "algorithm", "n", "k", "positive", "seed", "nr_seeds", "mean_seconds",
    "stddev_seconds", "maximum_seconds", "num_cases", "num_successful",
    "num_unexpected", "num_timeouts", "num_memory_limited", "num_errors",
    "num_generation_errors",
)


def read_statistics(path: Path) -> list[dict[str, str]]:
    """Read append-only statistics, keeping the latest duplicate invocation."""
    rows = read_csv(path, STAT_REQUIRED)
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in ("algorithm", "n", "k", "positive", "seed")
        )
        previous = unique.get(key)
        if previous is not None and previous["nr_seeds"] != row["nr_seeds"]:
            print(
                "warning: superseding statistics row in "
                f"{path} for (algorithm={row['algorithm']}, n={row['n']}, "
                f"k={row['k']}, positive={row['positive']}, seed={row['seed']}): "
                f"nr_seeds changed from {previous['nr_seeds']} to {row['nr_seeds']}",
                file=sys.stderr,
                flush=True,
            )
        unique[key] = row
    return list(unique.values())


def _pooled(values: Sequence[tuple[int, float, float]]) -> tuple[float | None, float | None]:
    total = sum(count for count, _, _ in values)
    if total == 0:
        return None, None
    average = sum(count * mean for count, mean, _ in values) / total
    if total == 1:
        return average, 0.0
    squared = sum(
        max(0, count - 1) * deviation**2 + count * (mean - average) ** 2
        for count, mean, deviation in values
    )
    return average, math.sqrt(squared / (total - 1))


def combine_statistic_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Pool positive/negative standard-statistics rows for one parameter cell."""
    if not rows:
        raise ValueError("cannot combine an empty statistics group")
    distributions: list[tuple[int, float, float]] = []
    maxima: list[float] = []
    for row in rows:
        # statistics.py includes successful, unexpected, and timed-out calls in
        # runtime aggregates, while excluding execution and memory failures.
        observed = as_int(row["num_cases"]) - as_int(row["num_memory_limited"]) - as_int(row["num_errors"])
        average = as_float(row["mean_seconds"])
        deviation = as_float(row["stddev_seconds"])
        maximum = as_float(row["maximum_seconds"])
        if observed and average is not None:
            distributions.append((observed, average, deviation or 0.0))
        if maximum is not None:
            maxima.append(maximum)
    mean_seconds, stddev_seconds = _pooled(distributions)
    sample = rows[0]
    result: dict[str, Any] = {
        "algorithm": sample["algorithm"],
        "problem": problem_for_algorithm(sample["algorithm"]),
        "n": as_int(sample["n"]),
        "k": as_int(sample["k"]),
        "r": as_int(sample["n"]) - as_int(sample["k"]),
        "num_requested": sum(as_int(row["nr_seeds"]) for row in rows),
        "num_cases": sum(as_int(row["num_cases"]) for row in rows),
        "num_observed": sum(count for count, _, _ in distributions),
        "num_successful": sum(as_int(row["num_successful"]) for row in rows),
        "num_unexpected": sum(as_int(row["num_unexpected"]) for row in rows),
        "num_timeouts": sum(as_int(row["num_timeouts"]) for row in rows),
        "num_memory_limited": sum(as_int(row["num_memory_limited"]) for row in rows),
        "num_errors": sum(as_int(row["num_errors"]) for row in rows),
        "num_generation_errors": sum(as_int(row["num_generation_errors"]) for row in rows),
        "mean_seconds": mean_seconds,
        "stddev_seconds": stddev_seconds,
        "maximum_seconds": max(maxima) if maxima else None,
        "has_positive": any(as_bool(row["positive"]) for row in rows),
        "has_negative": any(not as_bool(row["positive"]) for row in rows),
    }
    result["complete"] = (
        result["has_positive"]
        and result["has_negative"]
        and result["num_successful"] == result["num_requested"]
        and not any(
            result[field]
            for field in (
                "num_unexpected", "num_timeouts", "num_memory_limited",
                "num_errors", "num_generation_errors",
            )
        )
    )
    return result


def aggregate_statistics(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], as_int(row["n"]), as_int(row["k"]))].append(row)
    return [combine_statistic_rows(group) for _, group in sorted(grouped.items())]


def load_algorithm(algorithm: str, directory: Path = ALGORITHM_DATA_DIR) -> list[dict[str, str]]:
    return read_statistics(directory / f"{algorithm}.csv")


def load_all_algorithms(directory: Path = ALGORITHM_DATA_DIR) -> list[dict[str, str]]:
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no complete algorithm CSVs found in {directory}")
    return [row for path in paths for row in read_statistics(path)]
