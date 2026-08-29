"""Dependency and aggregation tests for the paper experiment layer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmarks.experiments.statistics import CSV_FIELDS
from paper.experiments.a1_rejection_rates.extract import extract as extract_a1
from paper.experiments.a2_signature_space.extract import extract as extract_a2
from paper.experiments.a3_invariant_cost.extract import extract as extract_a3
from paper.experiments.a4_representation_cost.extract import extract as extract_a4
from paper.experiments.a5_winners.extract import extract as extract_a5
from paper.experiments.a6_sat_css.extract import extract as extract_a6


def _stat_row(
    algorithm: str,
    positive: bool,
    mean: float,
    *,
    n: int = 3,
    k: int = 1,
    successful: int = 10,
    timeouts: int = 0,
    generation_errors: int = 0,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "generator": "fixture",
        "name": "",
        "n": n,
        "k": k,
        "positive": positive,
        "density": "",
        "symmetry": "",
        "seed": 42,
        "nr_seeds": 10,
        "timeout_seconds": 5400,
        "memory_limit_bytes": 1,
        "mean_seconds": mean,
        "stddev_seconds": 0,
        "maximum_seconds": mean,
        "num_cases": 10,
        "num_successful": successful,
        "num_unexpected": 0,
        "num_timeouts": timeouts,
        "num_memory_limited": 0,
        "num_errors": 0,
        "num_generation_errors": generation_errors,
    }


def _write_statistics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _algorithm(path: Path, name: str, positive_mean: float, negative_mean: float, **kwargs) -> None:
    _write_statistics(
        path / f"{name}.csv",
        [
            _stat_row(name, True, positive_mean, **kwargs),
            _stat_row(name, False, negative_mean, **kwargs),
        ],
    )


def _write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _invariant_timing_rows(
    problem: str,
    invariant: str,
    runtime: float,
) -> list[dict[str, object]]:
    return [
        {
            "problem": problem,
            "invariant": invariant,
            "seed": seed,
            "n": 3,
            "k": 1,
            "positive": positive,
            "runtime_seconds": runtime,
            "status": "success",
        }
        for positive in (True, False)
        for seed in range(5)
    ]


def test_a1_computes_component_and_combined_rejections(tmp_path: Path) -> None:
    source = tmp_path / "rejections.csv"
    fields = ("problem", "instance_id", "n", "k", "r", "invariant", "rejected", "status")
    _write_rows(
        source,
        fields,
        [
            {"problem": "pm_stb", "instance_id": "one", "n": 3, "k": 1, "r": 2, "invariant": "linear_dependency", "rejected": False, "status": "success"},
            {"problem": "pm_stb", "instance_id": "one", "n": 3, "k": 1, "r": 2, "invariant": "signatures", "rejected": True, "status": "success"},
        ],
    )

    cells = extract_a1(source, tmp_path / "a1")

    combined = next(row for row in cells if row["invariant"] == "combined")
    assert combined["num_rejected"] == 1
    assert combined["rejection_percentage"] == 100


def test_a2_aggregates_random_code_instances(tmp_path: Path) -> None:
    source = tmp_path / "signatures.csv"
    fields = ("problem", "seed", "n", "k", "q_pairs", "status")
    _write_rows(
        source,
        fields,
        [
            {"problem": "pm_css", "seed": 89, "n": 3, "k": 1, "q_pairs": 1.0, "status": "success"},
            {"problem": "pm_css", "seed": 773, "n": 3, "k": 1, "q_pairs": 0.5, "status": "success"},
        ],
    )

    cells = extract_a2(source, tmp_path / "a2.csv")

    assert len(cells) == 1
    assert cells[0]["num_valid"] == 2
    # For n=3, q=1 maps to 1 and q=1/2 maps to 1/4 after removing
    # unavoidable self-pairs; the two independently seeded codes are averaged.
    assert cells[0]["mean_distinct_pair_fraction"] == pytest.approx(0.625)


def test_a4_reads_only_graph_algorithm_files_and_aggregates_polarities(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_graph_iso", 1.0, 3.0)
    _algorithm(algorithms, "lc_stb_graph_iso", 2.0, 4.0)
    _algorithm(algorithms, "pm_stb_sat", 0.01, 0.01)
    output = tmp_path / "a4.csv"

    rows = extract_a4(algorithms, output)

    assert {row["algorithm"] for row in rows} == {
        "pm_stb_graph_iso",
        "lc_stb_graph_iso",
    }
    pm = next(row for row in rows if row["algorithm"] == "pm_stb_graph_iso")
    assert pm["mean_total_seconds"] == pytest.approx(2.0)
    assert output.is_file()


def test_a5_selects_fastest_complete_algorithm_and_rejects_failed_one(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_graph_iso", 2.0, 2.0)
    _algorithm(algorithms, "pm_stb_sat", 1.0, 1.0)
    _algorithm(
        algorithms,
        "pm_stb_bruteforce",
        0.1,
        0.1,
        successful=9,
        timeouts=1,
    )

    winners = extract_a5(algorithms, tmp_path / "winners")

    assert winners[0]["winner"] == "pm_stb_sat"
    assert winners[0]["runner_up"] == "pm_stb_graph_iso"
    assert winners[0]["num_eligible_algorithms"] == 2


def test_a3_backend_choice_is_parameter_dependent(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_sat", 2.0, 2.0)
    _algorithm(algorithms, "pm_stb_graph_iso", 1.0, 1.0)
    invariant_file = tmp_path / "invariants.csv"
    invariant_rows = _invariant_timing_rows("pm_stb", "signatures", 0.2)
    _write_rows(
        invariant_file,
        tuple(invariant_rows[0]),
        invariant_rows,
    )

    rows = extract_a3(invariant_file, algorithms, tmp_path / "a3.csv")

    assert rows[0]["backend_algorithm"] == "pm_stb_graph_iso"
    assert rows[0]["relative_runtime"] == pytest.approx(0.2)


def test_a3_excludes_incomplete_invariant_cells(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "lc_stb_sat", 1.0, 1.0)
    invariant_file = tmp_path / "invariants.csv"
    invariant_rows = _invariant_timing_rows(
        "lc_stb", "local_invariant", 0.2
    )
    invariant_rows[-1]["status"] = "generation_error"
    invariant_rows[-1]["runtime_seconds"] = ""
    _write_rows(
        invariant_file,
        tuple(invariant_rows[0]),
        invariant_rows,
    )

    rows = extract_a3(invariant_file, algorithms, tmp_path / "a3.csv")

    assert rows == []


def test_a6_uses_two_complete_files_and_only_the_extra_css_file(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_sat", 1.0, 3.0)
    _algorithm(algorithms, "pm_css_sat", 2.0, 4.0)
    extra = tmp_path / "extra.csv"
    _write_statistics(
        extra,
        [
            _stat_row("pm_stb_sat_on_css", True, 3.0),
            _stat_row("pm_stb_sat_on_css", False, 5.0),
        ],
    )

    rows = extract_a6(algorithms, extra, tmp_path / "a6.csv")

    assert {row["variant"] for row in rows} == {
        "pm_stb_sat_on_stabilizer",
        "pm_css_sat_on_css",
        "pm_stb_sat_on_css",
    }
    assert all(row["num_requested"] == 20 for row in rows)
