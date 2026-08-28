"""Collect raw signature transformation-space measurements.

Usage::

    python3 -m paper.benchmarks.collect_signature_space

There are no CLI arguments. For PM-STB and PM-CSS, the script generates both
positive and certified-negative pairs whose signatures match and records
``sum(|s_i|^2)/n^2``. Every result is appended immediately to
``paper/data/collected/signature_space.csv``. Restarting skips keys already
present, while A2 aggregates both labels later.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.experiments.run import run
from paper.benchmarks.utils.config import (
    COLLECTED_DATA_DIR,
    DIMENSIONS,
    MEMORY_LIMIT_BYTES,
    SEEDS,
    TIMEOUT_SECONDS,
    append_csv_row,
    completed_csv_keys,
    csv_key,
    execution_status,
)
from paper.benchmarks.utils.generation import signature_pair
from paper.benchmarks.utils.invariants import evaluate_signature

OUTPUT_FILE = COLLECTED_DATA_DIR / "signature_space.csv"
PROBLEMS = ("pm_stb", "pm_css")
FIELDS = (
    "problem", "positive", "seed", "n", "k", "r", "class_sizes", "q_pairs",
    "status", "timeout", "memory_limited", "error",
)


def signature_metric(class_sizes: Sequence[int], n: int) -> float:
    return sum(size * size for size in class_sizes) / (n * n)


def collect(
    *,
    dimensions: Sequence[tuple[int, int]] = DIMENSIONS,
    seeds: Sequence[int] = SEEDS,
    output_file: Path = OUTPUT_FILE,
) -> list[dict[str, Any]]:
    key_fields = ("problem", "n", "k", "seed", "positive")
    completed = completed_csv_keys(output_file, key_fields)
    rows: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for positive in (True, False):
            label = "positive" if positive else "negative"
            print(f"signature space: {problem} {label}", flush=True)
            for n, k in dimensions:
                for seed in seeds:
                    key = csv_key(problem, n, k, seed, positive)
                    if key in completed:
                        continue
                    base = {
                        "problem": problem, "positive": positive, "seed": seed,
                        "n": n, "k": k, "r": n - k,
                    }
                    try:
                        pair = signature_pair(problem, n, k, seed, positive)
                    except Exception as exc:  # noqa: BLE001 - benchmark data
                        row = {
                            **base, "class_sizes": "", "q_pairs": None,
                            "status": "generation_error", "timeout": False,
                            "memory_limited": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        append_csv_row(output_file, row, FIELDS)
                        rows.append(row)
                        completed.add(key)
                        continue

                    result = run(
                        evaluate_signature,
                        (problem, *pair),
                        None,
                        timeout=TIMEOUT_SECONDS,
                        max_memory_bytes=MEMORY_LIMIT_BYTES,
                    )
                    status = execution_status(result)
                    sizes: list[int] = []
                    error = result.error or ""
                    if status == "success":
                        compatible, sizes, partner_sizes = result.result
                        if not compatible or sorted(sizes) != sorted(partner_sizes):
                            status = "error"
                            error = "generated pair does not have matching signatures"
                    row = {
                        **base,
                        "class_sizes": " ".join(map(str, sizes)),
                        "q_pairs": signature_metric(sizes, n) if status == "success" else None,
                        "status": status, "timeout": result.timed_out,
                        "memory_limited": result.memory_exceeded, "error": error,
                    }
                    append_csv_row(output_file, row, FIELDS)
                    rows.append(row)
                    completed.add(key)

    print(f"appended {len(rows)} new signature measurements to {output_file}", flush=True)
    return rows


if __name__ == "__main__":
    collect()
