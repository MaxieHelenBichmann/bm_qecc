"""Reusable statistical benchmark orchestration.

The public :func:`run_statistics` function owns only repeated seeded case
generation, aggregation, and append-only CSV output. Benchmark suites decide
which algorithms, generators, parameter ranges, and resource limits to use.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from .run import RunResult, run


@dataclass(frozen=True)
class BenchmarkCase:
    """One generated input instance and its exact expected result."""

    inputs: tuple[Any, ...]
    expected: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Statistic:
    """Aggregate outcome of one algorithm/generator combination."""

    algorithm: str
    generator: str
    master_seed: int
    seeds: tuple[int, ...]
    runtimes: tuple[float, ...]
    mean_seconds: float
    stddev_seconds: float
    maximum_seconds: float
    num_requested: int
    num_cases: int
    num_successful: int
    num_unexpected: int
    num_timeouts: int
    num_memory_limited: int
    num_errors: int
    num_generation_errors: int
    metadata: Mapping[str, Any]


CSV_FIELDS = (
    "algorithm",
    "generator",
    "name",
    "n",
    "k",
    "positive",
    "density",
    "symmetry",
    "seed",
    "nr_seeds",
    "timeout_seconds",
    "memory_limit_bytes",
    "mean_seconds",
    "stddev_seconds",
    "maximum_seconds",
    "num_cases",
    "num_successful",
    "num_unexpected",
    "num_timeouts",
    "num_memory_limited",
    "num_errors",
    "num_generation_errors",
)


def deterministic_seeds(
    seed: int, nr_seeds: int, *, upper_bound: int = 2**32
) -> tuple[int, ...]:
    """Derive distinct seeds reproducibly from NumPy's seeded generator."""
    if nr_seeds <= 0:
        raise ValueError("nr_seeds must be greater than zero")
    if upper_bound <= 0 or nr_seeds > upper_bound:
        raise ValueError("nr_seeds cannot exceed the configured seed space")

    rng = np.random.default_rng(seed)
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < nr_seeds:
        candidate = int(rng.integers(0, upper_bound))
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return tuple(seeds)


def _callable_name(function: Callable[..., Any]) -> str:
    return getattr(function, "__name__", type(function).__name__)


def _case_metadata(
    generator: Callable[[int], BenchmarkCase], case: BenchmarkCase | None
) -> dict[str, Any]:
    metadata = dict(getattr(generator, "metadata", {}))
    if case is not None:
        metadata.update(case.metadata)
    return metadata


def _statistic(
    *,
    algorithm: Callable[..., Any],
    generator: Callable[[int], BenchmarkCase],
    master_seed: int,
    seeds: tuple[int, ...],
    results: Sequence[RunResult],
    generation_errors: int,
    metadata: Mapping[str, Any],
) -> Statistic:
    # A timeout contributes its capped runtime. Other failures do not describe
    # the algorithm's runtime and are therefore excluded from time statistics.
    runtimes = tuple(
        result.runtime
        for result in results
        if result.error is None and not result.memory_exceeded
    )
    mean_seconds = mean(runtimes) if runtimes else math.nan
    stddev_seconds = stdev(runtimes) if len(runtimes) > 1 else 0.0

    return Statistic(
        algorithm=str(metadata.get("algorithm", _callable_name(algorithm))),
        generator=str(metadata.get("generator", _callable_name(generator))),
        master_seed=master_seed,
        seeds=seeds,
        runtimes=runtimes,
        mean_seconds=mean_seconds,
        stddev_seconds=stddev_seconds,
        maximum_seconds=max(runtimes) if runtimes else math.nan,
        num_requested=len(seeds),
        num_cases=len(results),
        num_successful=sum(result.successful for result in results),
        num_unexpected=sum(
            result.error is None
            and not result.timed_out
            and not result.memory_exceeded
            and not result.result_is_expected
            for result in results
        ),
        num_timeouts=sum(result.timed_out for result in results),
        num_memory_limited=sum(result.memory_exceeded for result in results),
        num_errors=sum(result.error is not None for result in results),
        num_generation_errors=generation_errors,
        metadata=dict(metadata),
    )


def append_statistic(
    statistic: Statistic,
    output_file: str | Path,
    *,
    timeout: float | None,
    max_memory_bytes: int | None,
) -> None:
    """Append one statistic, writing the CSV header when the file is empty."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    metadata = statistic.metadata

    def number(value: float) -> str:
        return "" if math.isnan(value) else f"{value:.9f}"

    row = {
        "algorithm": statistic.algorithm,
        "generator": statistic.generator,
        "name": metadata.get("name", ""),
        "n": metadata.get("n", ""),
        "k": metadata.get("k", ""),
        "positive": metadata.get("positive", ""),
        "density": metadata.get("density", ""),
        "symmetry": metadata.get("symmetry", ""),
        "seed": statistic.master_seed,
        "nr_seeds": statistic.num_requested,
        "timeout_seconds": "" if timeout is None else timeout,
        "memory_limit_bytes": "" if max_memory_bytes is None else max_memory_bytes,
        "mean_seconds": number(statistic.mean_seconds),
        "stddev_seconds": number(statistic.stddev_seconds),
        "maximum_seconds": number(statistic.maximum_seconds),
        "num_cases": statistic.num_cases,
        "num_successful": statistic.num_successful,
        "num_unexpected": statistic.num_unexpected,
        "num_timeouts": statistic.num_timeouts,
        "num_memory_limited": statistic.num_memory_limited,
        "num_errors": statistic.num_errors,
        "num_generation_errors": statistic.num_generation_errors,
    }
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_statistics(
    algorithm: Callable[..., Any],
    generator: Callable[[int], BenchmarkCase],
    seed: int,
    nr_seeds: int,
    output_file: str | Path,
    *,
    timeout: float | None = None,
    max_memory_bytes: int | None = None,
    verbose: bool = False,
) -> Statistic:
    """Run deterministic seeded instances and append their statistics to CSV.

    ``generator`` is called once for each derived seed and must return a
    :class:`BenchmarkCase`. Input-generation failures are counted separately;
    failures in the benchmarked algorithm are represented by :class:`RunResult`.
    """
    seed_upper_bound = int(getattr(generator, "seed_upper_bound", 2**32))
    seeds = deterministic_seeds(seed, nr_seeds, upper_bound=seed_upper_bound)
    results: list[RunResult] = []
    generation_errors = 0
    representative_case: BenchmarkCase | None = None

    for case_seed in seeds:
        try:
            case = generator(case_seed)
            if not isinstance(case, BenchmarkCase):
                raise TypeError("generator must return BenchmarkCase")
        except Exception as exc:  # noqa: BLE001 - generation errors are statistics
            generation_errors += 1
            if verbose:
                print(f"        seed={case_seed}: generation error: {type(exc).__name__}: {exc}")
            continue

        representative_case = representative_case or case
        if verbose:
            print(f"        seed={case_seed}")
        results.append(
            run(
                algorithm,
                case.inputs,
                case.expected,
                timeout=timeout,
                max_memory_bytes=max_memory_bytes,
            )
        )

    metadata = _case_metadata(generator, representative_case)
    statistic = _statistic(
        algorithm=algorithm,
        generator=generator,
        master_seed=seed,
        seeds=seeds,
        results=results,
        generation_errors=generation_errors,
        metadata=metadata,
    )
    append_statistic(
        statistic,
        output_file,
        timeout=timeout,
        max_memory_bytes=max_memory_bytes,
    )
    return statistic


__all__ = [
    "BenchmarkCase",
    "CSV_FIELDS",
    "Statistic",
    "append_statistic",
    "deterministic_seeds",
    "run_statistics",
]
