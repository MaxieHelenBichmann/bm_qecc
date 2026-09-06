from __future__ import annotations

from pathlib import Path

import pytest

from paper.experiments.common import STAT_REQUIRED, write_csv
from paper.experiments.extract_a1 import extract as extract_a1
from paper.experiments.extract_a2 import pairwise_refinement
from paper.experiments.extract_a5 import extract as extract_a5, select_winners
from paper.experiments.extract_a6 import extract as extract_a6


def _statistics_row(
    algorithm: str,
    positive: bool,
    *,
    mean: float = 1.0,
    successful: int = 1,
    timeouts: int = 0,
    errors: int = 0,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "n": 5,
        "k": 2,
        "positive": positive,
        "seed": 42,
        "nr_seeds": 1,
        "mean_seconds": mean,
        "stddev_seconds": 0.0,
        "maximum_seconds": mean,
        "num_cases": 1,
        "num_successful": successful,
        "num_unexpected": 0,
        "num_timeouts": timeouts,
        "num_memory_limited": 0,
        "num_errors": errors,
        "num_generation_errors": 0,
    }


def _method(algorithm: str, mean: float, *, complete: bool, timeouts: int = 0, errors: int = 0) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "problem": "pm_stb",
        "n": 5,
        "k": 2,
        "mean_seconds": mean,
        "complete": complete,
        "has_positive": True,
        "has_negative": True,
        "num_requested": 2,
        "num_successful": 2 - timeouts - errors,
        "num_timeouts": timeouts,
        "num_unexpected": 0,
        "num_memory_limited": 0,
        "num_errors": errors,
        "num_generation_errors": 0,
    }


def test_a1_keeps_measured_zero_rejections(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    output = tmp_path / "a1"
    fields = ("problem", "instance_id", "n", "k", "r", "invariant", "rejected", "status")
    write_csv(
        source,
        [
            {
                "problem": "pm_stb",
                "instance_id": "case-1",
                "n": 3,
                "k": 0,
                "r": 3,
                "invariant": invariant,
                "rejected": False,
                "status": "success",
            }
            for invariant in ("linear_dependency", "signatures")
        ],
        fields,
    )

    cells = extract_a1(source, output)

    measured = [row for row in cells if row["invariant"] != "combined"]
    assert len(measured) == 2
    assert all(row["num_valid"] == 1 and row["num_rejected"] == 0 for row in measured)
    assert not any(row["n"] == 4 for row in cells)


def test_pairwise_refinement_boundaries_and_range_check() -> None:
    assert pairwise_refinement(1 / 5, 5) == pytest.approx(1.0)
    assert pairwise_refinement(1.0, 5) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="outside the theoretical"):
        pairwise_refinement(0.1, 5)


def test_a5_prefers_completed_and_excludes_errors() -> None:
    winners = select_winners(
        [
            _method("pm_stb_sat", 2.0, complete=True),
            _method("pm_stb_bruteforce", 1.0, complete=False, timeouts=1),
            _method("pm_stb_graph_iso", 0.5, complete=False, errors=1),
        ]
    )

    assert winners[0]["winner"] == "pm_stb_sat"
    assert winners[0]["selection"] == "completed"
    assert "pm_stb_graph_iso (errors)" in winners[0]["excluded_algorithms"]


def test_a5_missing_file_degrades_gracefully(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    algorithms = tmp_path / "algorithms"
    output = tmp_path / "a5"
    write_csv(
        algorithms / "pm_stb_sat.csv",
        [
            _statistics_row("pm_stb_sat", True),
            _statistics_row("pm_stb_sat", False),
        ],
        STAT_REQUIRED,
    )

    winners = extract_a5(
        algorithms,
        output,
        algorithm_names=("pm_stb_sat", "pm_stb_aut"),
    )

    assert winners[0]["winner"] == "pm_stb_sat"
    assert "pm_stb_aut (missing data)" in winners[0]["excluded_algorithms"]
    assert "excluding algorithms with missing collected data: pm_stb_aut" in capsys.readouterr().err
    assert (output / "by_cell.csv").stat().st_size > 0


def test_a6_improvement_is_positive_when_check_matrix_is_faster(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    for algorithm, mean in (("pm_stb_sat", 4.0), ("pm_css_sat", 1.0)):
        write_csv(
            algorithms / f"{algorithm}.csv",
            [_statistics_row(algorithm, True, mean=mean), _statistics_row(algorithm, False, mean=mean)],
            STAT_REQUIRED,
        )
    extra = tmp_path / "pm_stb_sat_on_css.csv"
    write_csv(
        extra,
        [
            _statistics_row("pm_stb_sat_on_css", True, mean=10.0),
            _statistics_row("pm_stb_sat_on_css", False, mean=10.0),
        ],
        STAT_REQUIRED,
    )

    rows = extract_a6(algorithms, extra, tmp_path / "a6.csv")
    comparison = next(row for row in rows if row["variant"] == "pm_stb_sat_on_css")

    assert comparison["hx_hz_log_scale_improvement_percentage"] > 0
