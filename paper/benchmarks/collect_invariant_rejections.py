"""Collect raw invariant decisions on certified inequivalent pairs.

Usage::

    python3 -m paper.benchmarks.collect_invariant_rejections

There are no CLI arguments. For each fixed ``(n, k, seed)`` the script draws
two independent codes, certifies them as inequivalent with the corresponding
SAT backend, and records whether each relevant invariant rejects the pair.
Every result is appended immediately to
``paper/data/collected/invariant_rejections.csv``. Restarting the script skips
keys already present, while the A1 experiment performs all aggregation later.
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
from paper.benchmarks.utils.generation import certified_negative_pair
from paper.benchmarks.utils.invariants import INVARIANTS, evaluate_invariant

OUTPUT_FILE = COLLECTED_DATA_DIR / "invariant_rejections.csv"
PROBLEMS = ("pm_stb", "pm_css", "lc_stb")
FIELDS = (
    "problem", "instance_id", "seed", "n", "k", "r", "invariant",
    "rejected", "status", "timeout", "memory_limited", "error",
)


def collect(
    *,
    dimensions: Sequence[tuple[int, int]] = DIMENSIONS,
    seeds: Sequence[int] = SEEDS,
    output_file: Path = OUTPUT_FILE,
) -> list[dict[str, Any]]:
    key_fields = ("problem", "n", "k", "seed", "invariant")
    completed = completed_csv_keys(output_file, key_fields)
    rows: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        print(f"invariant rejections: {problem}", flush=True)
        for n, k in dimensions:
            for seed in seeds:
                instance_id = f"{problem}-n{n}k{k}-s{seed}"
                missing = [
                    invariant
                    for invariant in INVARIANTS[problem]
                    if csv_key(problem, n, k, seed, invariant) not in completed
                ]
                if not missing:
                    continue
                try:
                    pair = certified_negative_pair(problem, n, k, seed)
                except Exception as exc:  # noqa: BLE001 - recorded benchmark data
                    for invariant in missing:
                        row = {
                            "problem": problem, "instance_id": instance_id,
                            "seed": seed, "n": n, "k": k, "r": n - k,
                            "invariant": invariant, "rejected": None,
                            "status": "generation_error", "timeout": False,
                            "memory_limited": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        append_csv_row(output_file, row, FIELDS)
                        rows.append(row)
                        completed.add(csv_key(problem, n, k, seed, invariant))
                    continue

                for invariant in missing:
                    result = run(
                        evaluate_invariant,
                        (invariant, problem, *pair),
                        None,
                        timeout=TIMEOUT_SECONDS,
                        max_memory_bytes=MEMORY_LIMIT_BYTES,
                    )
                    status = execution_status(result)
                    row = {
                        "problem": problem, "instance_id": instance_id,
                        "seed": seed, "n": n, "k": k, "r": n - k,
                        "invariant": invariant,
                        "rejected": not bool(result.result) if status == "success" else None,
                        "status": status, "timeout": result.timed_out,
                        "memory_limited": result.memory_exceeded,
                        "error": result.error or "",
                    }
                    append_csv_row(output_file, row, FIELDS)
                    rows.append(row)
                    completed.add(csv_key(problem, n, k, seed, invariant))

    print(f"appended {len(rows)} new raw decisions to {output_file}", flush=True)
    return rows


if __name__ == "__main__":
    collect()
