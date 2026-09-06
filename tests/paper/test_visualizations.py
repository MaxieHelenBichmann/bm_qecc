"""Tests for the paper visualizations' fixed schemas."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from paper.visualizations.common import load_rows
from paper.visualizations.visualize_a1 import render as render_rejections
from paper.visualizations.visualize_a2 import render
from paper.visualizations.visualize_a3 import render as render_relative
from paper.visualizations.visualize_a5 import render as render_winners


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_stale_collector_schema_is_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    _write(path, ("problem", "median_q_pairs"), [{"problem": "pm_stb", "median_q_pairs": 1}])
    with pytest.raises(ValueError, match="obsolete schema"):
        load_rows(path, ("problem", "mean_pairwise_refinement"))


def test_rejection_plot_writes_main_and_overall_pngs(tmp_path: Path) -> None:
    path = tmp_path / "by_cell.csv"
    fields = ("problem", "invariant", "n", "k", "r", "num_valid", "num_rejected")
    rows = [
        {
            "problem": problem,
            "invariant": invariant,
            "n": 3,
            "k": 1,
            "r": 2,
            "num_valid": 20,
            "num_rejected": rejected,
        }
        for problem, invariant, rejected in (
            ("pm_stb", "linear_dependency", 10),
            ("pm_css", "linear_dependency", 15),
            ("pm_stb", "signatures", 12),
            ("pm_css", "signatures", 18),
            ("lc_stb", "local_invariant", 8),
        )
    ]
    _write(path, fields, rows)
    output = render_rejections(path, tmp_path / "rejections.png")
    assert output.is_file()
    assert sorted(file.name for file in tmp_path.glob("*.png")) == [
        "rejections.png",
        "rejections_overall.png",
    ]


def test_signature_plot_writes_one_png(tmp_path: Path) -> None:
    path = tmp_path / "by_cell.csv"
    fields = (
        "problem", "n", "k", "r", "num_valid",
        "num_censored", "mean_pairwise_refinement",
    )
    rows = [
        {
            "problem": problem, "n": 3, "k": 1, "r": 2,
            "num_valid": 20, "num_censored": 0,
            "mean_pairwise_refinement": value,
        }
        for problem in ("pm_stb", "pm_css")
        for value in (0.5,)
    ]
    _write(path, fields, rows)
    output = render(path, tmp_path / "signature_space.png")
    assert output.is_file()
    assert list(tmp_path.glob("*.png")) == [output]


def test_winner_plot_writes_one_png(tmp_path: Path) -> None:
    path = tmp_path / "by_cell.csv"
    fields = (
        "problem", "n", "k", "r", "winner", "mean_seconds", "runner_up",
        "runner_up_mean_seconds", "speed_ratio", "num_eligible_algorithms",
        "selection",
    )
    rows = [
        {
            "problem": problem,
            "n": 3,
            "k": 1,
            "r": 2,
            "winner": algorithm,
            "mean_seconds": 1.0,
            "runner_up": "",
            "runner_up_mean_seconds": "",
            "speed_ratio": "",
            "num_eligible_algorithms": 1,
            "selection": "completed",
        }
        for problem, algorithm in (
            ("pm_stb", "pm_stb_sat"),
            ("pm_css", "pm_css_matroid"),
            ("lc_stb", "lc_stb_lse"),
        )
    ]
    _write(path, fields, rows)
    output = render_winners(path, tmp_path / "winner_maps.png")
    assert output.is_file()
    assert list(tmp_path.glob("*.png")) == [output]


def test_relative_cost_plot_writes_one_png(tmp_path: Path) -> None:
    path = tmp_path / "by_cell.csv"
    fields = (
        "problem", "invariant", "n", "k", "r", "invariant_mean_seconds",
        "invariant_stddev_seconds", "backend_algorithm", "backend_mean_seconds",
        "backend_selection", "backend_num_timeouts", "relative_runtime",
        "num_invariant_requested", "num_invariant_successful",
    )
    rows = [
        {
            "problem": problem, "invariant": invariant, "n": 3, "k": 1, "r": 2,
            "invariant_mean_seconds": 0.5, "invariant_stddev_seconds": 0.0,
            "backend_algorithm": f"{problem}_sat", "backend_mean_seconds": 1.0,
            "backend_selection": selection, "backend_num_timeouts": 0,
            "relative_runtime": 0.5,
            "num_invariant_requested": 10, "num_invariant_successful": 10,
        }
        for invariant in ("linear_dependency", "signatures")
        for problem, selection in (
            ("pm_stb", "completed"),
            ("pm_css", "timeout_fallback"),
        )
    ]
    _write(path, fields, rows)
    output = render_relative(path, tmp_path / "relative_cost.png")
    assert output.is_file()
    assert list(tmp_path.glob("*.png")) == [output]
