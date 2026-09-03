"""Collect raw invariant decisions on certified inequivalent pairs.

For each fixed ``(n, k, seed)``, PM-STB and LC-STB apply a small, configurable number of
random Clifford gates to one source code, while PM-CSS applies a short physical-CNOT
circuit that preserves the CSS form. The X- and Z-check ranks therefore match too.
Each candidate is certified with an admissible exact backend (SAT) before the script
records whether the relevant invariants reject it. PM-CSS uses a SAT- and 
matroid-based  verification method, and a scalable certified CSS code pair generator
due to runtime constraints.

Every result is appended immediately to
``paper/data/collected/invariant_rejections.csv``. Practical feasibility
(runtime and memory consumption) is not important here, so it can be run on any platform.
The invariants and input generation are still run a resource limit and errors are recorded, 
to not unnecessarily exhaust resources. 
Restarting skips keys already present, while the A1 experiment performs all aggregation later.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.experiments.generators_random import NonPEqCodePairGenerator
from benchmarks.experiments.run import RunResult, run
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.thesis.thesis_prototypes import measurement_dimensions
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids import lc_stb, p_css, p_stab

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
STABILIZER_CLIFFORD_GATE_STEPS = 2
CSS_CNOT_GATE_STEPS = 2

OUTPUT_FILE = ROOT / "paper" / "data" / "collected" / "invariant_rejections.csv"
PROBLEMS = ("pm_stb", "pm_css", "lc_stb")
INVARIANTS = {
    "pm_stb": ("linear_dependency", "signatures"),
    "pm_css": ("linear_dependency", "signatures"),
    "lc_stb": ("local_invariant",),
}
FIELDS = (
    "problem", "instance_id", "seed", "n", "k", "r", "invariant",
    "rejected", "status", "timeout", "memory_limited", "error",
)
CodePair = tuple[StabilizerCode, StabilizerCode]
CERTIFIERS: dict[str, Callable[..., bool]] = {
    "pm_stb": are_peq_stab_sat,
    "lc_stb": are_lceq_sat,
}


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


# Invariants ------------------------------------------------------------------------------------

def _prepared(problem: str, left: StabilizerCode, right: StabilizerCode) -> tuple:
    row_basis = p_stab._row_basis
    if problem == "pm_css":
        if not isinstance(left, CSSCode) or not isinstance(right, CSSCode):
            raise TypeError("pm_css invariants require CSSCode inputs")
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
) -> bool:
    matrices = _prepared(problem, left, right)
    if problem == "pm_stb":
        compatible, _, _ = p_stab.preserved_punctured_hull_weight_enumerator(*matrices)
    elif problem == "pm_css":
        compatible, _, _ = p_css.preserved_punctured_hull_weight_enumerator(*matrices)
    else:
        raise ValueError(f"no signature invariant for {problem}")
    return bool(compatible)


def evaluate_invariant(
    name: str,
    problem: str,
    left: StabilizerCode,
    right: StabilizerCode,
) -> bool:
    if name == "signatures":
        return evaluate_signature(problem, left, right)
    matrices = _prepared(problem, left, right)
    if name == "linear_dependency" and problem == "pm_stb":
        return bool(p_stab.preserved_linear_dependencies(*matrices))
    if name == "linear_dependency" and problem == "pm_css":
        return bool(p_css.preserved_linear_dependencies(*matrices))
    if name == "local_invariant" and problem == "lc_stb":
        return bool(lc_stb.preserved_low_degree_local_invariant(*matrices))
    raise ValueError(f"unknown invariant {name!r} for {problem!r}")


# Negative-pair generation ----------------------------------------------------------------------

def _attempt_seed(problem: str, n: int, k: int, seed: int, attempt: int) -> int:
    population = f"{problem}_negative_matching=False"
    value = f"{population}|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _candidate_pair(problem: str, n: int, k: int, seed: int) -> CodePair:
    if problem == "pm_css":
        # same dimensions of the check matrices to emulate practically relevant instances and not make the problem too trivial
        rx = seed % (n - k + 1)
        return NonPEqCodePairGenerator.css_codes_cnot_candidate(
            n,
            k,
            seed,
            rx=rx,
            gate_steps=CSS_CNOT_GATE_STEPS,
        )
    # keeps the two stabilizer codes related somehow to emulate practically relevant instances
    return NonPEqCodePairGenerator.stabilizer_codes_clifford_candidate(
        n,
        k,
        seed,
        gate_steps=STABILIZER_CLIFFORD_GATE_STEPS,
    )


def _css_certifier(n: int, k: int) -> Callable[..., bool] | None:
    if n - k <= CSS_SAT_MAX_R:
        return are_peq_css_sat
    if n <= CSS_MATROID_MAX_N:
        return are_peq_css_matroid
    return None


def _certified_inequivalent(problem: str, pair: CodePair, n: int, k: int) -> bool:
    certifier = _css_certifier(n, k) if problem == "pm_css" else CERTIFIERS[problem]
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
    use_certified_css_generator = problem == "pm_css" and _css_certifier(n, k) is None
    for attempt in range(max_attempts):
        attempt_seed = _attempt_seed(problem, n, k, seed, attempt)
        pair = (
            NonPEqCodePairGenerator.css_codes_cascaded(n, k, attempt_seed)
            if use_certified_css_generator
            else _candidate_pair(problem, n, k, attempt_seed)
        )
        if use_certified_css_generator or _certified_inequivalent(problem, pair, n, k):
            return pair
    raise RuntimeError(
        f"could not generate a certified {problem} negative for [[{n},{k}]], seed {seed}"
    )


# Collection ------------------------------------------------------------------------------------

def collect(
    *,
    dimensions: Sequence[tuple[int, int]] = DIMENSIONS,
    seeds: Sequence[int] = SEEDS,
    output_file: Path = OUTPUT_FILE,
    problems: Sequence[str] = PROBLEMS,
) -> list[dict[str, Any]]:
    unknown = set(problems) - set(PROBLEMS)
    if unknown:
        raise ValueError(f"unknown A1 problems: {sorted(unknown)}")
    key_fields = ("problem", "n", "k", "seed", "invariant")
    completed = completed_csv_keys(output_file, key_fields)
    rows: list[dict[str, Any]] = []
    for problem in problems:
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
