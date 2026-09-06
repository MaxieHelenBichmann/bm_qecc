from __future__ import annotations

from pathlib import Path

import pytest

from paper.experiments.common import (
    STAT_REQUIRED,
    aggregate_statistics,
    combine_statistic_rows,
    problem_for_algorithm,
    read_statistics,
    write_csv,
)
from paper.benchmarks import collect_algorithm


def _row(*, positive: bool, nr_seeds: int, mean: float = 1.0) -> dict[str, object]:
    return {
        "algorithm": "pm_stb_sat",
        "n": 5,
        "k": 2,
        "positive": positive,
        "seed": 42,
        "nr_seeds": nr_seeds,
        "mean_seconds": mean,
        "stddev_seconds": 0.0,
        "maximum_seconds": mean,
        "num_cases": nr_seeds,
        "num_successful": nr_seeds,
        "num_unexpected": 0,
        "num_timeouts": 0,
        "num_memory_limited": 0,
        "num_errors": 0,
        "num_generation_errors": 0,
    }


def test_read_statistics_keeps_latest_same_sized_invocation(tmp_path: Path) -> None:
    path = tmp_path / "statistics.csv"
    write_csv(path, [_row(positive=True, nr_seeds=3, mean=1.0), _row(positive=True, nr_seeds=3, mean=2.0)], STAT_REQUIRED)

    rows = read_statistics(path)

    assert len(rows) == 1
    assert rows[0]["mean_seconds"] == "2.0"


def test_changed_seed_count_supersedes_and_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "statistics.csv"
    rows = [
        _row(positive=True, nr_seeds=3),
        _row(positive=False, nr_seeds=3),
        _row(positive=True, nr_seeds=5),
        _row(positive=False, nr_seeds=5),
    ]
    write_csv(path, rows, STAT_REQUIRED)

    aggregated = aggregate_statistics(read_statistics(path))

    assert len(aggregated) == 1
    assert aggregated[0]["num_requested"] == 10
    assert aggregated[0]["num_cases"] == 10
    assert "nr_seeds changed from 3 to 5" in capsys.readouterr().err


def test_combine_statistic_rows_pools_positive_and_negative() -> None:
    combined = combine_statistic_rows(
        [
            {key: str(value) for key, value in _row(positive=True, nr_seeds=2, mean=1.0).items()},
            {key: str(value) for key, value in _row(positive=False, nr_seeds=2, mean=3.0).items()},
        ]
    )

    assert combined["num_requested"] == 4
    assert combined["mean_seconds"] == pytest.approx(2.0)
    assert combined["complete"] is True


def test_problem_for_algorithm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="cannot infer problem family"):
        problem_for_algorithm("unknown_backend")


def test_automorphism_collection_skips_before_writing_without_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(collect_algorithm.shutil, "which", lambda _: None)
    monkeypatch.setattr(collect_algorithm, "OUTPUT_DIRECTORY", tmp_path)

    collect_algorithm.collect(("pm_stb_aut",))

    assert not list(tmp_path.iterdir())
    assert "skipping pm_stb_aut" in capsys.readouterr().out
