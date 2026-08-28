"""Focused tests for the fixed paper data-collection infrastructure."""

from __future__ import annotations

from pathlib import Path
import csv

import pytest

from benchmarks.experiments.run import RunResult
from paper.benchmarks import collect_invariant_rejections
from paper.benchmarks.collect_signature_space import signature_metric
from paper.benchmarks import collect_algorithm
from paper.benchmarks.utils.config import DIMENSIONS, SEEDS
from paper.benchmarks.utils.generation import certified_negative_pair
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode


def test_fixed_suite_matches_the_thesis_grid_and_seed_schedule() -> None:
    assert len(DIMENSIONS) == 185
    assert DIMENSIONS[:3] == ((3, 0), (3, 1), (3, 2))
    assert SEEDS == (89, 773, 654, 438, 433, 858, 85, 697, 201, 94)


def test_independent_stabilizer_negative_is_certified() -> None:
    left, right = certified_negative_pair("pm_stb", 3, 1, 89, max_attempts=5)
    assert are_peq_stab_sat(left, right) is False


def test_independent_css_negative_has_a_complete_certificate() -> None:
    left, right = certified_negative_pair("pm_css", 3, 1, 89, max_attempts=5)
    assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
    # Different check ranks are themselves a complete PM-CSS certificate; the
    # SAT encoding is used internally when the ranks match.
    rank_mismatch = (left.Hx.shape[0], left.Hz.shape[0]) != (
        right.Hx.shape[0],
        right.Hz.shape[0],
    )
    assert rank_mismatch or are_peq_css_sat(left, right) is False


def test_signature_metric_has_the_expected_extremes() -> None:
    assert signature_metric([7], 7) == pytest.approx(1.0)
    assert signature_metric([1] * 7, 7) == pytest.approx(1 / 7)


def test_only_active_public_collectors_remain() -> None:
    directory = Path(__file__).parents[1] / "benchmarks"
    names = sorted(path.name for path in directory.glob("collect_*.py"))
    assert names == [
        "collect_algorithm.py",
        "collect_css_sat_encoding.py",
        "collect_hybrids.py",
        "collect_invariant_rejections.py",
        "collect_invariant_timings.py",
        "collect_signature_space.py",
    ]


def test_algorithm_collector_cli_exposes_only_algorithm_selection() -> None:
    args = collect_algorithm.parse_args(["--algorithm", "pm_stb_sat"])
    assert vars(args) == {"algorithm": ["pm_stb_sat"]}


def test_algorithm_collector_chooses_output_file_from_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(collect_algorithm, "OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(collect_algorithm, "VERBOSE", False)
    monkeypatch.setattr(
        collect_algorithm,
        "run_suite",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    collect_algorithm.collect(["pm_stb_sat"])

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (["pm_stb_sat"],)
    assert kwargs["output_file"] == tmp_path / "pm_stb_sat.csv"
    assert (kwargs["nmin"], kwargs["nmax"]) == collect_algorithm.ALGORITHM_N_RANGES["pm_stb_sat"]


def test_raw_collector_persists_rows_and_resumes_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rejections.csv"
    monkeypatch.setattr(collect_invariant_rejections, "PROBLEMS", ("pm_stb",))
    monkeypatch.setattr(
        collect_invariant_rejections,
        "certified_negative_pair",
        lambda *args: (object(), object()),
    )
    monkeypatch.setattr(
        collect_invariant_rejections,
        "run",
        lambda *args, **kwargs: RunResult(
            runtime=0.1,
            result=False,
            expected=None,
            result_is_expected=False,
            timed_out=False,
            memory_exceeded=False,
            error=None,
        ),
    )

    first = collect_invariant_rejections.collect(
        dimensions=[(3, 1)], seeds=[89], output_file=output
    )
    second = collect_invariant_rejections.collect(
        dimensions=[(3, 1)], seeds=[89], output_file=output
    )

    with output.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(first) == 2
    assert second == []
    assert len(persisted) == 2
