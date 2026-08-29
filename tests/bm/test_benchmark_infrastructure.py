"""Focused tests for the layered benchmark infrastructure."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from benchmarks.experiments import run as run_module
from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import (
    BenchmarkCase,
    deterministic_seeds,
    run_statistics,
)


def _identity(value: int) -> int:
    return value


def _sleep(seconds: float) -> bool:
    time.sleep(seconds)
    return True


def _raise_memory_error() -> None:
    raise MemoryError


class _RecordingQueue:
    def __init__(self) -> None:
        self.item: object | None = None

    def put(self, item: object) -> None:
        self.item = item


def test_run_reports_expected_and_unexpected_results() -> None:
    expected = run(_identity, (3,), 3)
    unexpected = run(_identity, (3,), 4)

    assert expected.successful
    assert expected.result_is_expected
    assert not unexpected.successful
    assert not unexpected.result_is_expected
    assert unexpected.error is None


def test_run_reports_timeout_separately() -> None:
    result = run(_sleep, (0.2,), True, timeout=0.01)

    assert result.timed_out
    assert not result.memory_exceeded
    assert result.error is None


def test_run_reports_memory_error_as_memory_limit_failure() -> None:
    result = run(_raise_memory_error, (), None, max_memory_bytes=2**40)

    assert result.memory_exceeded
    assert not result.timed_out
    assert result.error is None


def test_run_reports_other_execution_error() -> None:
    result = run(lambda: 1 / 0, (), None)

    assert result.error == "ZeroDivisionError: division by zero"
    assert not result.timed_out
    assert not result.memory_exceeded


def test_worker_reports_call_runtime_without_supervisor_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _RecordingQueue()
    times = iter((10.0, 10.25))
    monkeypatch.setattr(run_module.os, "setsid", lambda: None)
    monkeypatch.setattr(run_module, "perf_counter", lambda: next(times))

    run_module._worker(_identity, (3,), queue, None)

    assert queue.item == ("result", 3, 0.25)


def test_statistics_uses_distinct_deterministic_seeds_and_appends_header_once(
    tmp_path: Path,
) -> None:
    observed_seeds: list[int] = []

    def generator(seed: int) -> BenchmarkCase:
        observed_seeds.append(seed)
        return BenchmarkCase(
            (_identity(seed),),
            seed,
            {"algorithm": "identity", "n": 1, "k": 0, "positive": True},
        )

    output = tmp_path / "statistics.csv"
    first = run_statistics(_identity, generator, 17, 4, output)
    second = run_statistics(_identity, generator, 17, 4, output)

    expected_seeds = deterministic_seeds(17, 4)
    assert first.seeds == second.seeds == expected_seeds
    assert len(set(expected_seeds)) == 4
    assert observed_seeds == list(expected_seeds) * 2

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0][0] == "algorithm"
    assert sum(row and row[0] == "algorithm" for row in rows) == 1
    assert len(rows) == 3
