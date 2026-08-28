"""Focused tests for the fixed paper data-collection infrastructure."""

from __future__ import annotations

from pathlib import Path
import csv

import pytest

from benchmarks.experiments.run import RunResult
from paper.benchmarks import collect_invariant_rejections as rejections
from paper.benchmarks import collect_signature_space as signatures
from paper.benchmarks.collect_signature_space import signature_metric
from paper.benchmarks import collect_algorithm
from paper.benchmarks.collect_invariant_rejections import (
    DIMENSIONS,
    SEEDS,
    certified_negative_pair,
)
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode


def test_fixed_suite_matches_the_thesis_grid_and_seed_schedule() -> None:
    assert len(DIMENSIONS) == 185
    assert DIMENSIONS[:3] == ((3, 0), (3, 1), (3, 2))
    assert SEEDS == (89, 773, 654, 438, 433, 858, 85, 697, 201, 94)


def test_clifford_perturbed_stabilizer_negative_is_certified() -> None:
    left, right = certified_negative_pair("pm_stb", 3, 1, 89, max_attempts=5)
    assert are_peq_stab_sat(left, right) is False


def test_independent_css_negative_has_a_complete_certificate() -> None:
    left, right = certified_negative_pair("pm_css", 3, 1, 89, max_attempts=5)
    assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
    assert (left.Hx.shape[0], left.Hz.shape[0]) == (
        right.Hx.shape[0],
        right.Hz.shape[0],
    )
    assert are_peq_css_sat(left, right) is False


def test_every_independent_css_candidate_has_matching_check_ranks() -> None:
    for seed in range(10):
        left, right = rejections._candidate_pair("pm_css", 7, 3, seed)
        assert (left.Hx.shape[0], left.Hz.shape[0]) == (
            right.Hx.shape[0],
            right.Hz.shape[0],
        )


def test_stabilizer_candidates_use_clifford_perturbations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (object(), object())
    calls = []
    monkeypatch.setattr(
        rejections.NonPEqCodePairGenerator,
        "stabilizer_codes_clifford_candidate",
        lambda *args, **kwargs: calls.append((args, kwargs)) or pair,
    )
    assert rejections._candidate_pair("pm_stb", 7, 3, 89) is pair
    assert rejections._candidate_pair("lc_stb", 7, 3, 89) is pair
    assert len(calls) == 2
    assert all(
        kwargs["gate_steps"] == rejections.STABILIZER_CLIFFORD_GATE_STEPS
        for _, kwargs in calls
    )


def test_css_certifier_selection_respects_backend_limits() -> None:
    assert rejections._css_certifier(47, 38) is are_peq_css_sat  # r = 9
    assert rejections._css_certifier(28, 18) is rejections.are_peq_css_matroid
    assert rejections._css_certifier(29, 19) is None


def test_large_high_rank_css_uses_certified_generator_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (object(), object())
    monkeypatch.setattr(
        rejections.NonPEqCodePairGenerator,
        "css_codes_cascaded",
        lambda *args: pair,
    )
    monkeypatch.setattr(
        rejections,
        "_certified_inequivalent",
        lambda *args: pytest.fail("large CSS fallback must not invoke a backend"),
    )

    assert rejections.certified_negative_pair(
        "pm_css", 29, 19, 89, max_attempts=1
    ) is pair


def test_large_pm_stb_signature_pair_matches_by_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (object(), object())
    monkeypatch.setattr(
        signatures.NonPEqCodePairGenerator,
        "stabilizer_codes_x_z_rank_projection",
        lambda *args: pair,
    )
    monkeypatch.setattr(
        signatures,
        "certified_negative_pair",
        lambda *args, **kwargs: pytest.fail(
            "large PM-STB signature generation must not rejection-sample signatures"
        ),
    )

    assert signatures.signature_pair("pm_stb", 21, 10, 89, False) is pair


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


def test_algorithm_collector_runs_every_configured_algorithm_without_selector() -> None:
    args = collect_algorithm.parse_args([])
    assert args.algorithm == list(collect_algorithm.ALGORITHM_N_RANGES)


def test_algorithm_collector_chooses_output_file_from_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(collect_algorithm, "OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(collect_algorithm, "VERBOSE", False)
    monkeypatch.setattr(collect_algorithm, "measurement_dimensions", lambda *args: [(3, 1)])
    monkeypatch.setattr(
        collect_algorithm,
        "run_statistics",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    collect_algorithm.collect(["pm_stb_sat"])

    assert len(calls) == 2
    args, kwargs = calls[0]
    assert args[0] is collect_algorithm.ALGORITHMS["pm_stb_sat"]
    assert args[1].n == 3 and args[1].k == 1 and args[1].positive is True
    assert args[2:4] == (collect_algorithm.MASTER_SEED, collect_algorithm.NUM_SEEDS)
    assert args[4] == tmp_path / "pm_stb_sat.csv"
    assert calls[1][0][1].positive is False


def test_raw_collector_persists_rows_and_resumes_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rejections.csv"
    monkeypatch.setattr(rejections, "PROBLEMS", ("pm_stb",))
    monkeypatch.setattr(
        rejections,
        "certified_negative_pair",
        lambda *args: (object(), object()),
    )
    monkeypatch.setattr(
        rejections,
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

    first = rejections.collect(
        dimensions=[(3, 1)], seeds=[89], output_file=output
    )
    second = rejections.collect(
        dimensions=[(3, 1)], seeds=[89], output_file=output
    )

    with output.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(first) == 2
    assert second == []
    assert len(persisted) == 2
