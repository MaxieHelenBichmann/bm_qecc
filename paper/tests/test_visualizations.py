"""Tests for the fixed-schema paper visualizations."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from paper.visualizations.common import load_rows
from paper.visualizations.plot_signature_space import render


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_stale_collector_schema_is_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    _write(path, ("problem", "median_q_pairs"), [{"problem": "pm_stb", "median_q_pairs": 1}])
    with pytest.raises(ValueError, match="obsolete schema"):
        load_rows(path, ("problem", "mean_q_pairs"))


def test_signature_plot_writes_one_png(tmp_path: Path) -> None:
    path = tmp_path / "by_cell.csv"
    fields = ("problem", "n", "k", "r", "num_valid", "mean_q_pairs")
    rows = [
        {"problem": problem, "n": 3, "k": 1, "r": 2, "num_valid": 20, "mean_q_pairs": value}
        for problem in ("pm_stb", "pm_css")
        for value in (2 / 3,)
    ]
    _write(path, fields, rows)
    output = render(path, tmp_path / "signature_space.png")
    assert output.is_file()
    assert list(tmp_path.glob("*.png")) == [output]
