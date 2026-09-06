from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from benchmarks.experiments.generators_structured import load_named_code
from paper.benchmarks import collect_a8
from paper.experiments.extract_a8 import extract
from paper.visualizations.visualize_a8 import render
from src.core.css_code import CSSCode

SMALL_CODES = ("bell", "3q_rep", "5q_prf")


@pytest.mark.parametrize("code_name", ("steane", "5q_prf"))
def test_code_serialization_round_trips(code_name: str) -> None:
    code = load_named_code(code_name)
    restored = collect_a8.decode_code(collect_a8.encode_code(code), code.n)
    assert isinstance(restored, CSSCode) == isinstance(code, CSSCode)
    assert (restored.n, restored.k) == (code.n, code.k)
    assert np.array_equal(restored.symplectic % 2, code.symplectic % 2)


@pytest.mark.parametrize("problem", tuple(collect_a8.HYBRIDS))
@pytest.mark.parametrize("positive", (True, False))
def test_generated_instances_are_deterministic_and_correctly_labeled(problem: str, positive: bool) -> None:
    for code_name in SMALL_CODES:
        if problem == "pm_css" and code_name in collect_a8.NON_CSS_CODES:
            continue
        first = collect_a8.generate_instance(problem, code_name, positive, 89)
        second = collect_a8.generate_instance(problem, code_name, positive, 89)
        assert first == second
        left = collect_a8.decode_code(first["left"], first["n"])
        right = collect_a8.decode_code(first["right"], first["n"])
        assert (left.n, left.k) == (right.n, right.k)
        assert collect_a8.certified_inequivalent(problem, left, right) is not positive


def test_pm_hybrids_share_the_positive_css_instance() -> None:
    stb = collect_a8.generate_instance("pm_stb", "steane", True, 89)
    css = collect_a8.generate_instance("pm_css", "steane", True, 89)
    assert (stb["left"], stb["right"]) == (css["left"], css["right"])


def test_read_trace_reports_last_stage_of_a_killed_call(tmp_path: Path) -> None:
    log = tmp_path / "trace.log"
    log.write_text("CI\nEI\nS\n", encoding="utf-8")
    assert collect_a8.read_trace(log) == (["CI", "EI", "S"], "")
    log.write_text("CI\nSAT\n#decided_by SAT\n", encoding="utf-8")
    assert collect_a8.read_trace(log) == (["CI", "SAT"], "SAT")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_collect_extract_render_pipeline_resumes_from_csvs(tmp_path: Path) -> None:
    kwargs = dict(codes=("bell", "5q_prf"), seeds=(89, 7), output_directory=tmp_path,
                  generation_timeout=60.0, timeout=60.0, memory_limit_bytes=4 * 1024**3,
                  verbose=False)
    collect_a8.collect(("pm_stb", "pm_css"), **kwargs)

    raw = _rows(tmp_path / "pm_stb_raw.csv")
    assert len(raw) == 8
    assert {row["status"] for row in raw} == {"success"}
    assert all(row["decided_by"] for row in raw)
    assert len(_rows(tmp_path / "pm_css_raw.csv")) == 4  # 5q_prf is not CSS
    instances = _rows(tmp_path / "pm_stb_instances.csv")
    assert len(instances) == 8

    # Re-running measures nothing new; deleting the raw file reuses the cached instances.
    collect_a8.collect(("pm_stb",), **kwargs)
    assert len(_rows(tmp_path / "pm_stb_raw.csv")) == 8
    (tmp_path / "pm_stb_raw.csv").unlink()
    collect_a8.collect(("pm_stb",), **kwargs)
    assert len(_rows(tmp_path / "pm_stb_raw.csv")) == 8
    assert _rows(tmp_path / "pm_stb_instances.csv") == instances

    cells = extract(tmp_path, tmp_path / "by_cell.csv")
    assert len(cells) == 6
    bell = next(cell for cell in cells if cell["problem"] == "pm_stb" and cell["code"] == "bell" and cell["positive"])
    assert bell["num_cases"] == 2 and bell["num_successful"] == 2
    assert bell["primary_decider"] and bell["mean_seconds"] > 0
    output = render(tmp_path / "by_cell.csv", tmp_path / "a8.png")
    assert output.stat().st_size > 0


def test_generation_failure_is_recorded_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(problem: str, code_name: str, positive: bool, seed: int) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(collect_a8, "generate_instance", broken)
    collect_a8.collect(("lc_stb",), codes=("bell",), seeds=(1,), output_directory=tmp_path,
                       generation_timeout=30.0, timeout=30.0, memory_limit_bytes=2 * 1024**3,
                       verbose=False)
    raw = _rows(tmp_path / "lc_stb_raw.csv")
    assert [row["status"] for row in raw] == ["generation_error", "generation_error"]
    assert "boom" in raw[0]["error"]
    assert [row["status"] for row in _rows(tmp_path / "lc_stb_instances.csv")] == ["generation_error"] * 2
