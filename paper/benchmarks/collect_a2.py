"""Collect signature-partition sizes on typical random codes.

For each fixed ``(n, k, seed)`` a random non-CSS or CSS code is generated, 
and its signature partition is computed. 
The raw metric is ``sum(|s_i|^2) / n^2``, where ``|s_i|`` are the signature
class sizes. It is the probability that two qubits sampled independently with
replacement have the same signature; the extractor converts it to the
fraction of distinct pairs separated by the signature. 

Every result is appended immediately to ``paper/data/collected/signature_space.csv``. 
Restarting skips completed ``(problem, n, k, seed)`` keys. Generation and 
signature-computation failures are retained as explicit rows. The file is opened only 
in append mode; existing measurements are never rewritten or deleted.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.experiments.run import RunResult, run
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.experiments.utils import random_css_code, random_stabilizer_code
from benchmarks.thesis.thesis_prototypes import measurement_dimensions
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids import p_css, p_stab

ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 42
NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
DIMENSIONS = tuple(
    sorted(
        measurement_dimensions(30, 47),
        key=lambda dimension: (
            dimension[0] - dimension[1],
            dimension[0],
            dimension[1],
        ),
    )
)
TIMEOUT_SECONDS = 5_400.0
MEMORY_LIMIT_BYTES = 13 * 1024**3

OUTPUT_FILE = ROOT / "paper" / "data" / "collected" / "signature_space.csv"
PROBLEMS = ("pm_stb",)
FIELDS = (
    "problem",
    "seed",
    "n",
    "k",
    "r",
    "x_rank",
    "class_sizes",
    "q_pairs",
    "status",
    "timeout",
    "memory_limited",
    "error",
)
Code = CSSCode | StabilizerCode


# CSV persistence -------------------------------------------------------------------------------

def execution_status(result: RunResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.memory_exceeded:
        return "memory_limited"
    if result.error is not None:
        return "error"
    return "success"


def csv_key(*values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def completed_csv_keys(path: Path, key_fields: Sequence[str]) -> set[tuple[str, ...]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(key_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} has an incompatible header; missing {sorted(missing)}"
            )
        return {
            tuple(row[field] for field in key_fields)
            for row in reader
            if all(row.get(field) is not None for field in key_fields)
        }


def append_csv_row(
    path: Path,
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Random-code generation ------------------------------------------------------------------------

def generate_random_code(
    problem: str,
    n: int,
    k: int,
    seed: int,
) -> tuple[Code, int | None]:
    """Generate one unconditioned code from the repository's random ensemble."""
    if problem == "pm_css":
        rng = np.random.default_rng(seed)
        x_rank = int(rng.integers(0, n - k + 1))
        code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return random_css_code(n, k, rx=x_rank, seed=code_seed), x_rank
    if problem == "pm_stb":
        return random_stabilizer_code(n, k, seed=seed), None
    raise ValueError(f"unknown signature problem {problem!r}")


# Signature computation -------------------------------------------------------------------------

def _partition_sizes(partition: Mapping[Any, Sequence[int]] | None) -> list[int]:
    return sorted((len(group) for group in (partition or {}).values()), reverse=True)


def evaluate_signature_partition(problem: str, code: Code) -> list[int]:
    """Compute one code's partition through the production signature routine."""
    # Passing the same matrices selects each routine's single-computation
    # self-comparison path while keeping one canonical signature implementation.
    row_basis = p_stab._row_basis
    if problem == "pm_css":
        if not isinstance(code, CSSCode):
            raise TypeError("pm_css signatures require a CSSCode")
        hx = row_basis(code.Hx)
        hz = row_basis(code.Hz)
        compatible, partition, _ = p_css.preserved_punctured_hull_weight_enumerator(
            hx, hz, hx, hz
        )
    elif problem == "pm_stb":
        matrix = row_basis(code.symplectic)
        compatible, partition, _ = p_stab.preserved_punctured_hull_weight_enumerator(
            matrix, matrix
        )
    else:
        raise ValueError(f"unknown signature problem {problem!r}")

    if not compatible or partition is None:
        raise RuntimeError("a code was unexpectedly incompatible with itself")
    return _partition_sizes(partition)


def signature_metric(class_sizes: Sequence[int], n: int) -> float:
    return sum(size * size for size in class_sizes) / (n * n)


# Collection ------------------------------------------------------------------------------------

def collect(
    *,
    dimensions: Sequence[tuple[int, int]] = DIMENSIONS,
    seeds: Sequence[int] = SEEDS,
    output_file: Path = OUTPUT_FILE,
) -> list[dict[str, Any]]:
    key_fields = ("problem", "n", "k", "seed")
    completed = completed_csv_keys(output_file, key_fields)
    rows: list[dict[str, Any]] = []

    for problem in PROBLEMS:
        print(f"signature partitions: {problem}", flush=True)
        for n, k in dimensions:
            for seed in seeds:
                key = csv_key(problem, n, k, seed)
                if key in completed:
                    continue

                base = {
                    "problem": problem,
                    "seed": seed,
                    "n": n,
                    "k": k,
                    "r": n - k,
                    "x_rank": "",
                }
                try:
                    code, x_rank = generate_random_code(problem, n, k, seed)
                    base["x_rank"] = "" if x_rank is None else x_rank
                except Exception as exc:  # noqa: BLE001 - recorded benchmark data
                    row = {
                        **base,
                        "class_sizes": "",
                        "q_pairs": None,
                        "status": "generation_error",
                        "timeout": False,
                        "memory_limited": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    append_csv_row(output_file, row, FIELDS)
                    rows.append(row)
                    completed.add(key)
                    continue

                result = run(
                    evaluate_signature_partition,
                    (problem, code),
                    None,
                    timeout=TIMEOUT_SECONDS,
                    max_memory_bytes=MEMORY_LIMIT_BYTES,
                )
                status = execution_status(result)
                sizes = list(result.result) if status == "success" else []
                row = {
                    **base,
                    "class_sizes": " ".join(map(str, sizes)),
                    "q_pairs": (
                        signature_metric(sizes, n) if status == "success" else None
                    ),
                    "status": status,
                    "timeout": result.timed_out,
                    "memory_limited": result.memory_exceeded,
                    "error": result.error or "",
                }
                append_csv_row(output_file, row, FIELDS)
                rows.append(row)
                completed.add(key)

    print(f"appended {len(rows)} new signature measurements to {output_file}", flush=True)
    return rows


if __name__ == "__main__":
    collect()
