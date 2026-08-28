"""Collect raw signature transformation-space measurements.

Usage::

    python3 -m paper.benchmarks.collect_signature_space

For PM-STB and PM-CSS, the script generates both positive and certified-negative 
pairs whose signatures match and records ``sum(|s_i|^2)/n^2``. Practical feasibility 
(runtime and memory consumption) is not important here, only in the case of an error
it is recorded. Every result is appended immediately to 
``paper/data/collected/signature_space.csv``. Restarting skips keys already
present, while A2 aggregates both labels later. 
The generated instances aim to be as random as possible (w.r.t. partition sizes)
while still being verifiable (in)equivalent and quick to generate.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.experiments.generators_random import (
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from benchmarks.experiments.run import RunResult, run
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.thesis.thesis_prototypes import measurement_dimensions
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids import p_css, p_stab

ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 42
NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
DIMENSIONS = tuple(measurement_dimensions())
TIMEOUT_SECONDS = 5_400.0
CERTIFICATION_TIMEOUT_SECONDS = 600.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
CSS_SAT_MAX_R = 9
CSS_MATROID_MAX_N = 28
PM_STB_SIGNATURE_MATCH_MAX_N = 20

OUTPUT_FILE = ROOT / "paper" / "data" / "collected" / "signature_space.csv"
PROBLEMS = ("pm_stb", "pm_css")
FIELDS = (
    "problem", "positive", "seed", "n", "k", "r", "class_sizes", "q_pairs",
    "status", "timeout", "memory_limited", "error",
)
CodePair = tuple[StabilizerCode, StabilizerCode]


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


# Signature computation -------------------------------------------------------------------------

def _partition_sizes(partition: dict[Any, list[int]] | None) -> list[int]:
    return sorted((len(group) for group in (partition or {}).values()), reverse=True)


def _prepared(problem: str, left: StabilizerCode, right: StabilizerCode) -> tuple:
    row_basis = p_stab._row_basis
    if problem == "pm_css":
        if not isinstance(left, CSSCode) or not isinstance(right, CSSCode):
            raise TypeError("pm_css signatures require CSSCode inputs")
        return (
            row_basis(left.Hx),
            row_basis(left.Hz),
            row_basis(right.Hx),
            row_basis(right.Hz),
        )
    return row_basis(left.symplectic), row_basis(right.symplectic)


def evaluate_signature(
    problem: str,
    left: StabilizerCode,
    right: StabilizerCode,
) -> tuple[bool, list[int], list[int]]:
    matrices = _prepared(problem, left, right)
    if problem == "pm_stb":
        compatible, first, second = p_stab.preserved_punctured_hull_weight_enumerator(
            *matrices
        )
    elif problem == "pm_css":
        compatible, first, second = p_css.preserved_punctured_hull_weight_enumerator(
            *matrices
        )
    else:
        raise ValueError(f"no signature invariant for {problem}")
    return bool(compatible), _partition_sizes(first), _partition_sizes(second)


def signature_metric(class_sizes: Sequence[int], n: int) -> float:
    return sum(size * size for size in class_sizes) / (n * n)


# Pair generation and certification --------------------------------------------------------------

def _attempt_seed(population: str, n: int, k: int, seed: int, attempt: int) -> int:
    value = f"{population}|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _independent_candidate(problem: str, n: int, k: int, seed: int) -> CodePair:
    if problem == "pm_css":
        return NonPEqCodePairGenerator.css_codes_independent_candidate(n, k, seed)
    return NonPEqCodePairGenerator.stabilizer_codes_independent_candidate(n, k, seed)


def _css_rank_mismatch(pair: CodePair) -> bool:
    left, right = pair
    return (
        isinstance(left, CSSCode)
        and isinstance(right, CSSCode)
        and (left.Hx.shape[0], left.Hz.shape[0])
        != (right.Hx.shape[0], right.Hz.shape[0])
    )


def _css_certifier(n: int, k: int) -> Callable[..., bool] | None:
    if n - k <= CSS_SAT_MAX_R:
        return are_peq_css_sat
    if n <= CSS_MATROID_MAX_N:
        return are_peq_css_matroid
    return None


def _certified_inequivalent(problem: str, pair: CodePair, n: int, k: int) -> bool:
    if problem == "pm_css" and _css_rank_mismatch(pair):
        return True
    certifier = _css_certifier(n, k) if problem == "pm_css" else are_peq_stab_sat
    if certifier is None:
        raise RuntimeError(f"no independent CSS certifier configured for [[{n},{k}]]")
    result = run(
        certifier,
        pair,
        False,
        timeout=CERTIFICATION_TIMEOUT_SECONDS,
        max_memory_bytes=MEMORY_LIMIT_BYTES,
    )
    if result.timed_out:
        raise RuntimeError("inequivalence certification timed out")
    if result.memory_exceeded:
        raise RuntimeError("inequivalence certification exceeded memory limit")
    if result.error is not None:
        raise RuntimeError(f"inequivalence certification failed: {result.error}")
    return result.result is False


def certified_negative_pair(
    problem: str,
    n: int,
    k: int,
    seed: int,
    *,
    max_attempts: int = 1_000,
) -> CodePair:
    """Return an inequivalent pair whose two signatures match."""
    population = f"{problem}_negative_matching=True"
    use_certified_css_generator = problem == "pm_css" and _css_certifier(n, k) is None
    for attempt in range(max_attempts):
        attempt_seed = _attempt_seed(population, n, k, seed, attempt)
        pair = (
            NonPEqCodePairGenerator.css_codes_cascaded(n, k, attempt_seed) # verifying CSS code inequivalence is too slow for larger codes, so we use a generator that produces inequivalent codes by construction 
            if use_certified_css_generator
            else _independent_candidate(problem, n, k, attempt_seed)
        )
        if not evaluate_signature(problem, *pair)[0]:
            continue
        if use_certified_css_generator or _certified_inequivalent(problem, pair, n, k):
            return pair
    raise RuntimeError(
        f"could not generate a signature-matching {problem} negative for "
        f"[[{n},{k}]], seed {seed}"
    )


def signature_pair(
    problem: str,
    n: int,
    k: int,
    seed: int,
    positive: bool,
) -> CodePair:
    """Return a positive or certified-negative pair with matching signatures."""
    if positive:
        if problem == "pm_css":
            return PEqCodePairGenerator.css_codes_basis_changed(n, k, seed)
        return PEqCodePairGenerator.stabilizer_codes_permuted(n, k, seed)
    if problem == "pm_stb" and n > PM_STB_SIGNATURE_MATCH_MAX_N:
        return NonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection( # guaranteeing matching signatures for larger inequivalent stabilizer codes is too slow, so we use a generator guarantees match by construction
            n, k, seed
        )
    return certified_negative_pair(problem, n, k, seed)


# Collection ------------------------------------------------------------------------------------

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
