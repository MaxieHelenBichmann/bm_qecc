from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper.experiments.common import write_csv
from paper.experiments.extract_a7 import EXPERIMENT1, EXPERIMENT2
from paper.visualizations import (
    visualize_a1,
    visualize_a2,
    visualize_a3,
    visualize_a4,
    visualize_a5,
    visualize_a6,
    visualize_a7,
)


def _render(tmp_path: Path, name: str, rows: list[dict[str, object]], render) -> None:
    source = tmp_path / f"{name}.csv"
    output = tmp_path / f"{name}.png"
    write_csv(source, rows, tuple(rows[0]))
    assert render(source, output) == output
    assert output.stat().st_size > 0


def test_a1_draws_measured_zero_with_an_outline() -> None:
    figure, ax = plt.subplots()
    visualize_a1._draw(ax, ({(3, 3, "linear_dependency"): (0, 1)},), "linear_dependency")
    try:
        assert len(ax.patches) == 2
        assert ax.patches[-1].get_facecolor()[-1] == 0
    finally:
        plt.close(figure)


@pytest.mark.filterwarnings("ignore:This figure includes Axes")
def test_all_paper_visualizers_write_pngs_from_synthetic_csvs(tmp_path: Path) -> None:
    _render(
        tmp_path,
        "a1",
        [
            {"problem": problem, "n": 3, "r": 2, "invariant": invariant, "num_valid": 1, "num_rejected": 0}
            for problem, invariant in (
                ("pm_stb", "linear_dependency"),
                ("pm_stb", "signatures"),
                ("pm_css", "linear_dependency"),
                ("pm_css", "signatures"),
                ("lc_stb", "local_invariant"),
            )
        ],
        visualize_a1.render,
    )
    _render(
        tmp_path,
        "a2",
        [
            {"problem": problem, "n": 3, "r": 2, "num_valid": 1, "num_censored": 0, "mean_pairwise_refinement": 0.5}
            for problem in ("pm_stb", "pm_css")
        ],
        visualize_a2.render,
    )
    _render(
        tmp_path,
        "a3",
        [
            {
                "problem": problem,
                "invariant": invariant,
                "n": 3,
                "r": 2,
                "invariant_mean_seconds": 0.1,
                "backend_algorithm": "pm_stb_sat",
                "backend_mean_seconds": 1.0,
                "backend_selection": "completed",
                "relative_runtime": 0.1,
            }
            for problem, invariant in (
                ("pm_stb", "linear_dependency"),
                ("pm_css", "linear_dependency"),
                ("pm_stb", "signatures"),
                ("pm_css", "signatures"),
                ("lc_stb", "local_invariant"),
            )
        ],
        visualize_a3.render,
    )
    _render(
        tmp_path,
        "a4",
        [
            {
                "algorithm": algorithm,
                "n": 3,
                "r": 2,
                "mean_total_seconds": 1.0,
                "num_successful": 1,
                "num_timeouts": 0,
                "num_memory_limited": 0,
                "num_errors": int(algorithm == "lc_stb_graph_iso"),
                "num_unexpected": 0,
            }
            for algorithm in ("pm_stb_graph_iso", "lc_stb_graph_iso", "pm_css_matroid")
        ],
        visualize_a4.render,
    )
    _render(
        tmp_path,
        "a5",
        [
            {
                "problem": problem,
                "n": 3,
                "r": 2,
                "winner": f"{problem}_sat",
                "runner_up": "",
                "speed_ratio": "",
                "num_eligible_algorithms": 1,
                "selection": "timeout_fallback" if problem == "pm_css" else "completed",
                "excluded_algorithms": "pm_stb_aut (missing data)" if problem == "pm_stb" else "",
            }
            for problem in ("pm_stb", "pm_css", "lc_stb")
        ],
        visualize_a5.render,
    )
    _render(
        tmp_path,
        "a6",
        [
            {
                "variant": variant,
                "n": 3,
                "r": 2,
                "mean_seconds": 1.0,
                "hx_hz_log_scale_improvement_percentage": 10 if variant == "pm_stb_sat_on_css" else "",
                "num_successful": 1,
                "num_timeouts": 0,
                "num_memory_limited": 0,
                "num_errors": 0,
            }
            for variant in ("pm_stb_sat_on_stabilizer", "pm_css_sat_on_css", "pm_stb_sat_on_css")
        ],
        visualize_a6.render,
    )
    _render(
        tmp_path,
        "a7",
        [
            {
                "experiment": experiment,
                "condition": condition,
                "n": 8,
                "r": 4,
                "median_base_decisions": 10,
                "median_invalid_mapping_decisions": 5,
            }
            for experiment, conditions in (
                (EXPERIMENT1, ("A", "B1", "B2", "C")),
                (EXPERIMENT2, ("clean", "mixed")),
            )
            for condition in conditions
        ],
        visualize_a7.render,
    )
