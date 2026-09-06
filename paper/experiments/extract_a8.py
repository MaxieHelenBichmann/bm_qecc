"""Aggregate the raw A8 hybrid rows into one row per (problem, code, label)."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, as_bool, read_csv, write_csv

INPUT_DIRECTORY = COLLECTED_DATA_DIR / "hybrids"
OUTPUT = RESULTS_DIR / "a8" / "by_cell.csv"

PROBLEMS = ("pm_stb", "pm_css", "lc_stb")
CODE_LABELS = (
    ("bell", "Bell pair"), ("3q_rep", "3-bit repetition"),
    ("5q_prf", "5-qubit perfect"), ("steane", "Steane"), ("shor", "Shor"),
    ("carbon", "Carbon"), ("hamming_15", "Hamming-15"),
    ("15q_optimal", "15-qubit optimal"), ("tetrahedral", "Tetrahedral"),
    ("golay", "Golay"), ("rot_surf_d5", "Rotated surface d=5"),
    ("hamming_31", "Hamming-31"), ("coco_488", "CoCo-488"),
    ("coco_666", "CoCo-666"), ("bb_72", "BB-72"), ("bb_90", "BB-90"),
    ("bb_108", "BB-108"), ("bb_144", "BB-144"),
)
CODE_ORDER = {name: index for index, (name, _) in enumerate(CODE_LABELS)}
CODE_LABEL = dict(CODE_LABELS)
STAGES = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE", "trivial")

REQUIRED = (
    "problem", "code", "positive", "seed", "n", "k", "status", "runtime_seconds",
    "decided_by", "stuck_at", "timeout_seconds",
)
FIELDS = (
    "problem", "code", "code_label", "n", "k", "r", "positive", "num_cases",
    "num_successful", "num_unexpected", "num_timeouts", "num_memory_limited",
    "num_errors", "num_generation_errors", "mean_seconds", "stddev_seconds",
    "maximum_seconds", "primary_decider", "primary_decider_count",
    "secondary_decider", "secondary_decider_count", "deciders", "stuck_at",
    "timeout_seconds",
)


def distribution(values: list[str]) -> list[tuple[str, int]]:
    """Count stage tags, most frequent first, ties in pipeline order."""
    counts = Counter(value for value in values if value)
    order = {stage: index for index, stage in enumerate(STAGES)}
    return sorted(counts.items(), key=lambda item: (-item[1], order.get(item[0], len(order))))


def extract(input_directory: Path = INPUT_DIRECTORY, output_file: Path = OUTPUT) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, bool], list[dict[str, str]]] = defaultdict(list)
    for problem in PROBLEMS:
        path = input_directory / f"{problem}_raw.csv"
        if not path.is_file():
            print(f"warning: {path} is missing; run collect_a8 for {problem}", file=sys.stderr)
            continue
        for row in read_csv(path, REQUIRED):
            groups[(row["problem"], row["code"], as_bool(row["positive"]))].append(row)
    if not groups:
        raise FileNotFoundError(f"no A8 raw files in {input_directory}")

    output: list[dict[str, Any]] = []
    for (problem, code, positive), rows in groups.items():
        # Runtime statistics follow the other collectors: completed calls and
        # timeouts (capped at the budget) count, memory and execution errors do not.
        timed = [float(row["runtime_seconds"]) for row in rows
                 if row["status"] in {"success", "unexpected", "timeout"}]
        deciders = distribution([row["decided_by"] for row in rows])
        (primary, primary_count), (secondary, secondary_count) = (deciders + [("", 0), ("", 0)])[:2]
        generated = [row for row in rows if row["status"] != "generation_error"]
        n, k = (int(generated[0]["n"]), int(generated[0]["k"])) if generated else (None, None)
        output.append({
            "problem": problem, "code": code, "code_label": CODE_LABEL.get(code, code),
            "n": n, "k": k, "r": n - k if n is not None and k is not None else None,
            "positive": positive,
            "num_cases": len(rows),
            "num_successful": sum(row["status"] == "success" for row in rows),
            "num_unexpected": sum(row["status"] == "unexpected" for row in rows),
            "num_timeouts": sum(row["status"] == "timeout" for row in rows),
            "num_memory_limited": sum(row["status"] == "memory_limited" for row in rows),
            "num_errors": sum(row["status"] == "error" for row in rows),
            "num_generation_errors": len(rows) - len(generated),
            "mean_seconds": mean(timed) if timed else "",
            "stddev_seconds": stdev(timed) if len(timed) > 1 else (0.0 if timed else ""),
            "maximum_seconds": max(timed) if timed else "",
            "primary_decider": primary, "primary_decider_count": primary_count,
            "secondary_decider": secondary, "secondary_decider_count": secondary_count,
            "deciders": ";".join(f"{stage}:{count}" for stage, count in deciders),
            "stuck_at": ";".join(f"{stage}:{count}" for stage, count in
                                 distribution([row["stuck_at"] for row in rows])),
            "timeout_seconds": float(rows[0]["timeout_seconds"]),
        })
    output.sort(key=lambda row: (CODE_ORDER.get(row["code"], len(CODE_ORDER)), row["code"],
                                 PROBLEMS.index(row["problem"]), not row["positive"]))
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
