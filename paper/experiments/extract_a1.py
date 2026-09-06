"""Extract A1 rejection counts and overall rates from raw decisions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, as_bool, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "invariant_rejections.csv"
OUTPUT_DIRECTORY = RESULTS_DIR / "a1"
CELL_FIELDS = ("problem", "n", "k", "r", "invariant", "num_requested", "num_valid", "num_rejected", "rejection_percentage", "num_censored")
OVERALL_FIELDS = ("problem", "invariant", "num_valid", "num_rejected", "rejection_percentage")


def extract(input_file: Path = INPUT, output_directory: Path = OUTPUT_DIRECTORY) -> list[dict[str, Any]]:
    component_rows = read_csv(input_file, ("problem", "instance_id", "n", "k", "r", "invariant", "rejected", "status"))
    by_instance: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in component_rows:
        by_instance[(row["problem"], row["instance_id"])].append(row)
    combined: list[dict[str, Any]] = []
    for group in by_instance.values():
        sample = group[0]
        is_valid = all(row["status"] == "success" for row in group)
        combined.append({
            **sample,
            "invariant": "combined",
            "status": "success" if is_valid else "censored",
            "rejected": any(as_bool(row["rejected"]) for row in group) if is_valid else "",
        })
    all_rows: list[dict[str, Any]] = [*component_rows, *combined]
    groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[(row["problem"], int(row["n"]), int(row["k"]), row["invariant"])].append(row)
    cells = []
    for (problem, n, k, invariant), group in sorted(groups.items()):
        valid_rows = [row for row in group if row["status"] == "success"]
        rejected = sum(as_bool(row["rejected"]) for row in valid_rows)
        cells.append({
            "problem": problem, "n": n, "k": k, "r": n - k,
            "invariant": invariant, "num_requested": len(group),
            "num_valid": len(valid_rows), "num_rejected": rejected,
            "rejection_percentage": 100 * rejected / len(valid_rows) if valid_rows else "",
            "num_censored": len(group) - len(valid_rows),
        })
    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        overall_groups[(row["problem"], row["invariant"])].append(row)
    overall = []
    for (problem, invariant), group in sorted(overall_groups.items()):
        valid_rows = [row for row in group if row["status"] == "success"]
        rejected = sum(as_bool(row["rejected"]) for row in valid_rows)
        overall.append({
            "problem": problem, "invariant": invariant, "num_valid": len(valid_rows),
            "num_rejected": rejected,
            "rejection_percentage": 100 * rejected / len(valid_rows) if valid_rows else "",
        })
    write_csv(output_directory / "by_cell.csv", cells, CELL_FIELDS)
    write_csv(output_directory / "overall.csv", overall, OVERALL_FIELDS)
    return cells


if __name__ == "__main__":
    extract()
