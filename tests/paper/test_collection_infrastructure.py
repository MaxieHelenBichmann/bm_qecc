"""Focused tests for the paper data-collection infrastructure."""

from __future__ import annotations

from pathlib import Path
import csv

import pytest

from benchmarks.experiments.run import RunResult
from paper.benchmarks import collect_a1 as rejections
from paper.benchmarks import collect_a3 as timings
from paper.benchmarks import collect_a2 as signatures
from paper.benchmarks.collect_a2 import signature_metric
from paper.benchmarks import collect_algorithm
from paper.benchmarks.collect_a1 import (
    DIMENSIONS,
    SEEDS,
    certified_negative_pair,
)
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


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


def test_invariant_timing_generator_certifies_locally_before_preparing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (
        StabilizerCode.get_trivial_code(3),
        StabilizerCode.get_trivial_code(3),
    )
    events: list[object] = []

    prepared = (object(), object())
    monkeypatch.setattr(
        timings,
        "certified_negative_pair",
        lambda *args: events.append(("certify", args)) or pair,
    )
    monkeypatch.setattr(
        timings,
        "_prepared",
        lambda *inputs: events.append(("prepare", inputs)) or prepared,
    )

    generated = timings.generate_pair("lc_stb", 3, 0, False, 89)
    matrices = timings._prepared("lc_stb", *generated)

    assert matrices is prepared
    assert [event[0] for event in events] == ["certify", "prepare"]


def test_runtime_negative_uses_local_a1_style_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (object(), object())
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        collect_algorithm,
        "certified_negative_pair",
        lambda *args: calls.append(args) or pair,
    )

    first = collect_algorithm.CertifiedRandomCaseGenerator(
        "pm_stb_sat", 7, 3, False
    )(89)
    second = collect_algorithm.CertifiedRandomCaseGenerator(
        "pm_stb_graph_iso", 7, 3, False
    )(89)

    assert first.inputs == second.inputs == pair
    assert calls == [("pm_stb", 7, 3, 89), ("pm_stb", 7, 3, 89)]


def test_negative_certification_failure_becomes_generation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        timings,
        "certified_negative_pair",
        lambda *args: calls.append(args)
        or (_ for _ in ()).throw(RuntimeError("certification timed out")),
    )

    monkeypatch.setattr(
        timings,
        "INVARIANTS",
        {"lc_stb": ("local_invariant",)},
    )
    monkeypatch.setattr(
        timings,
        "INVARIANT_N_RANGES",
        {("lc_stb", "local_invariant"): (3, 3)},
    )
    monkeypatch.setattr(timings, "VERBOSE", False)

    rows = timings.collect(
        dimensions=[(3, 0)],
        seeds=(89,),
        output_file=tmp_path / "timings.csv",
    )

    negative = next(row for row in rows if row["positive"] is False)
    assert negative["status"] == "generation_error"
    assert negative["runtime_seconds"] is None
    assert calls == [("lc_stb", 3, 0, 89)]


def test_prepared_matrix_arity_matches_each_invariant_family() -> None:
    pm_stb = timings._prepared(
        "pm_stb", *timings.generate_pair("pm_stb", 3, 1, True, 89)
    )
    pm_css = timings._prepared(
        "pm_css", *timings.generate_pair("pm_css", 3, 1, True, 89)
    )
    lc_stb = timings._prepared(
        "lc_stb", *timings.generate_pair("lc_stb", 3, 1, True, 89)
    )

    assert len(pm_stb) == 2
    assert len(pm_css) == 4
    assert len(lc_stb) == 2


def test_signature_collector_covers_both_families_and_full_grid() -> None:
    assert signatures.PROBLEMS == ("pm_stb", "pm_css")
    assert set(signatures.DIMENSIONS) == set(signatures.measurement_dimensions())
    ranks = [n - k for n, k in signatures.DIMENSIONS]
    assert ranks == sorted(ranks)


def test_signature_collector_generates_one_code_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "signatures.csv"
    code = object()
    generated = []
    assert not hasattr(signatures, "NonPEqCodePairGenerator")
    monkeypatch.setattr(
        signatures,
        "PROBLEMS",
        ("pm_css",),
    )
    monkeypatch.setattr(
        signatures,
        "generate_random_code",
        lambda *args: generated.append(args) or (code, 1),
    )
    monkeypatch.setattr(
        signatures,
        "run",
        lambda *args, **kwargs: RunResult(
            runtime=0.1,
            result=[2, 1],
            expected=None,
            result_is_expected=False,
            timed_out=False,
            memory_exceeded=False,
            error=None,
        ),
    )

    first = signatures.collect(dimensions=[(3, 1)], seeds=[89], output_file=output)
    second = signatures.collect(dimensions=[(3, 1)], seeds=[89], output_file=output)

    with output.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert generated == [("pm_css", 3, 1, 89)]
    assert len(first) == 1
    assert second == []
    assert len(persisted) == 1
    assert "positive" not in persisted[0]
    assert persisted[0]["x_rank"] == "1"


def test_signature_metric_has_the_expected_extremes() -> None:
    assert signature_metric([7], 7) == pytest.approx(1.0)
    assert signature_metric([1] * 7, 7) == pytest.approx(1 / 7)


def test_only_active_public_collectors_remain() -> None:
    directory = Path(__file__).resolve().parents[2] / "paper" / "benchmarks"
    names = sorted(path.name for path in directory.glob("collect_*.py"))
    assert names == [
        "collect_a1.py",
        "collect_a2.py",
        "collect_a3.py",
        "collect_a6.py",
        "collect_a7.py",
        "collect_a8.py",
        "collect_algorithm.py",
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
        dimensions=[(3, 1)],
        seeds=[89],
        output_file=output,
        problems=("pm_stb",),
    )
    second = rejections.collect(
        dimensions=[(3, 1)],
        seeds=[89],
        output_file=output,
        problems=("pm_stb",),
    )

    with output.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(first) == 2
    assert second == []
    assert len(persisted) == 2
