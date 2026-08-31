"""Collect raw SAT decision statistics for the A7 CSS-structure experiment.

For each configured ``(n, k, seed)``, the first part generates positive pairs
for four conditions: 
    unrestricted general stabilizer codes (A), 
    general codes with two actual but hidden row-operation blocks (B1), 
    the same B1 pair with the two blocks exposed as ``R1`` and ``R2`` (B2), 
    and balanced CSS codes with independent ``Rx`` and ``Rz`` (C). 
    
Both B2 blocks contain X and Z information, whereas each C block sees only one Pauli component. 

In addition to the ordinary positive solve, fresh formulas force mappings outside the known witness
permutation. UNSAT decision counts measure how long these wrong mappings survive before contradiction. 

The second part compares clean (separated) CSS tableaus with independently row-mixed presentations 
of the same stabilizer groups. Both are solved with the full-tableau encoding, so
this isolates the effect of hiding the CSS generator split through invertible row transformations.

Every solve is appended immediately to
``paper/data/collected/a7_sat_css_structure.csv``. Practical feasibility
(runtime and memory consumption) is not important here, so it can be run on any platform.
Restarting skips keys already present, while the A7 extractor performs all
aggregation later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import z3

from benchmarks.experiments.utils import (
    _random_permutation,
    _random_row_space_base_change,
    _random_tableau_row_space_base_change,
    random_css_code,
    random_stabilizer_code,
)
from src.algorithms.p_css.p_css_sat import _build_peq_css_sat_solver
from src.algorithms.p_stb.p_stab_sat import _build_peq_stab_sat_solver
from src.core.css_code import CSSCode
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "data" / "collected" / "a7_sat_css_structure.csv"

MASTER_SEED = 42
K = 2
NUM_SAMPLES = 10
NUM_PROBES = 3
TIMEOUT_SECONDS = 300.0
EXPERIMENT1_N = (16, 18, 20)
EXPERIMENT2_N = (14, 16, 18)
VERBOSE = True

EXPERIMENT1 = "permutation_survival"
EXPERIMENT2 = "row_mixing"

RAW_FIELDS = (
    "experiment",
    "condition",
    "sample",
    "seed",
    "n",
    "k",
    "r",
    "rx",
    "rz",
    "measurement",
    "probe",
    "source",
    "target",
    "witness_target",
    "result",
    "timed_out",
    "timeout_seconds",
    "build_seconds",
    "solve_seconds",
    "decisions",
    "conflicts",
    "propagations",
    "rlimit_count",
    "assertions",
    "row_operation_variables",
)
KEY_FIELDS = (
    "experiment",
    "condition",
    "sample",
    "seed",
    "n",
    "k",
    "measurement",
    "probe",
)


@dataclass(frozen=True)
class Condition:
    experiment: str
    name: str
    builder: Callable[[], z3.Solver]
    witness: tuple[int, ...]
    row_operation_variables: int
    probe_mappings: bool


def _append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _completed_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(KEY_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} has an obsolete schema; missing {sorted(missing)}"
            )
        return {tuple(row[field] for field in KEY_FIELDS) for row in reader}


def _row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _sample_seed(master_seed: int, experiment: str, n: int, sample: int) -> int:
    payload = f"A7|{master_seed}|{experiment}|{n}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for target, source in enumerate(permutation):
        inverse[source] = target
    return tuple(inverse)


def _stabilizer_partner(
    left: StabilizerCode,
    permutation: Sequence[int],
    row_seed: int,
    block_sizes: tuple[int, int] | None,
) -> StabilizerCode:
    matrix = np.asarray(left.symplectic, dtype=np.int8).copy()
    if block_sizes is None:
        matrix = _random_row_space_base_change(matrix, seed=row_seed)
    else:
        first, second = block_sizes
        matrix[:first] = _random_row_space_base_change(matrix[:first], seed=row_seed)
        matrix[first:] = _random_row_space_base_change(
            matrix[first:], seed=row_seed + 1
        )
    columns = list(permutation) + [qubit + left.n for qubit in permutation]
    return StabilizerCode(StabilizerTableau(matrix[:, columns]))


def _css_partner(
    left: CSSCode, permutation: Sequence[int], row_seed: int
) -> CSSCode:
    hx = _random_row_space_base_change(left.Hx, seed=row_seed)[:, permutation]
    hz = _random_row_space_base_change(left.Hz, seed=row_seed + 1)[:, permutation]
    return CSSCode(hx, hz)


def _exactly_one(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    return z3.PbEq([(variable, 1) for variable in variables], 1)


def _xor(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    value = z3.BoolVal(False)
    for variable in variables:
        value = z3.Xor(value, variable)
    return value


def _column_value(
    column: np.ndarray, variables: Sequence[z3.BoolRef]
) -> z3.BoolRef:
    return z3.And(
        *[
            variable if bit else z3.Not(variable)
            for bit, variable in zip(column, variables, strict=True)
        ]
    )


def _build_block_general_solver(
    left: StabilizerCode,
    right: StabilizerCode,
    block_sizes: tuple[int, int],
) -> z3.Solver:
    """Encode a general tableau with two explicit row-operation matrices."""
    solver = z3.Solver()
    n = left.n
    first, second = block_sizes
    r = first + second
    auxiliary = [
        z3.Bool(f"aux_{row}_{column}")
        for row in range(r)
        for column in range(2 * n)
    ]
    permutation = [
        z3.Bool(f"p_{source}_{target}")
        for source in range(n)
        for target in range(n)
    ]
    for source in range(n):
        solver.add(
            _exactly_one(
                [permutation[source * n + target] for target in range(n)]
            )
        )
    for target in range(n):
        solver.add(
            _exactly_one(
                [permutation[source * n + target] for source in range(n)]
            )
        )

    for source in range(n):
        for target in range(n):
            x_variables = [
                auxiliary[row * (2 * n) + target] for row in range(r)
            ]
            z_variables = [
                auxiliary[row * (2 * n) + target + n] for row in range(r)
            ]
            solver.add(
                z3.Implies(
                    permutation[source * n + target],
                    z3.And(
                        _column_value(left.symplectic[:, source], x_variables),
                        _column_value(left.symplectic[:, source + n], z_variables),
                    ),
                )
            )

    for block, (offset, size) in enumerate(((0, first), (first, second)), 1):
        coefficients = [
            z3.Bool(f"r{block}_{row}_{column}")
            for row in range(size)
            for column in range(size)
        ]
        for local_row in range(size):
            output_row = offset + local_row
            for column in range(2 * n):
                contributions = [
                    coefficients[local_row * size + contribution]
                    for contribution in range(size)
                    if right.symplectic[offset + contribution, column]
                ]
                solver.add(
                    auxiliary[output_row * (2 * n) + column]
                    == _xor(contributions)
                )
    return solver


def _experiment1_conditions(n: int, k: int, seed: int) -> list[Condition]:
    rng = np.random.default_rng(seed)
    r = n - k
    first, second = r // 2, r - r // 2
    permutation = tuple(
        int(value)
        for value in _random_permutation(n, seed=int(rng.integers(0, 2**31)))
    )
    witness = _inverse_permutation(permutation)

    general_left = random_stabilizer_code(
        n, k, seed=int(rng.integers(0, 2**31))
    )
    full_right = _stabilizer_partner(
        general_left, permutation, int(rng.integers(0, 2**31)), None
    )
    block_right = _stabilizer_partner(
        general_left,
        permutation,
        int(rng.integers(0, 2**31)),
        (first, second),
    )

    css_left = random_css_code(
        n, k, rx=first, seed=int(rng.integers(0, 2**31))
    )
    css_right = _css_partner(
        css_left, permutation, int(rng.integers(0, 2**31))
    )

    return [
        Condition(
            EXPERIMENT1,
            "A",
            lambda: _build_peq_stab_sat_solver(general_left, full_right),
            witness,
            r * r,
            True,
        ),
        Condition(
            EXPERIMENT1,
            "B1",
            lambda: _build_peq_stab_sat_solver(general_left, block_right),
            witness,
            r * r,
            True,
        ),
        Condition(
            EXPERIMENT1,
            "B2",
            lambda: _build_block_general_solver(
                general_left, block_right, (first, second)
            ),
            witness,
            first * first + second * second,
            True,
        ),
        Condition(
            EXPERIMENT1,
            "C",
            lambda: _build_peq_css_sat_solver(css_left, css_right),
            witness,
            first * first + second * second,
            True,
        ),
    ]


def _experiment2_conditions(n: int, k: int, seed: int) -> list[Condition]:
    rng = np.random.default_rng(seed)
    r = n - k
    first = r // 2
    permutation = tuple(
        int(value)
        for value in _random_permutation(n, seed=int(rng.integers(0, 2**31)))
    )
    witness = _inverse_permutation(permutation)
    css_left = random_css_code(
        n, k, rx=first, seed=int(rng.integers(0, 2**31))
    )
    css_right = _css_partner(
        css_left, permutation, int(rng.integers(0, 2**31))
    )
    clean_left = StabilizerCode(css_left.generators)
    clean_right = StabilizerCode(css_right.generators)
    mixed_left = StabilizerCode(
        _random_tableau_row_space_base_change(
            css_left.generators,
            seed=int(rng.integers(0, 2**31)),
            steps=30 * r,
        )
    )
    mixed_right = StabilizerCode(
        _random_tableau_row_space_base_change(
            css_right.generators,
            seed=int(rng.integers(0, 2**31)),
            steps=30 * r,
        )
    )
    return [
        Condition(
            EXPERIMENT2,
            "clean",
            lambda: _build_peq_stab_sat_solver(clean_left, clean_right),
            witness,
            r * r,
            False,
        ),
        Condition(
            EXPERIMENT2,
            "mixed",
            lambda: _build_peq_stab_sat_solver(mixed_left, mixed_right),
            witness,
            r * r,
            False,
        ),
    ]


def _solver_statistics(solver: z3.Solver) -> dict[str, int | float]:
    statistics = solver.statistics()
    return {key: statistics.get_key_value(key) for key in statistics.keys()}


def _measure(
    condition: Condition,
    *,
    sample: int,
    seed: int,
    n: int,
    k: int,
    timeout_seconds: float,
    measurement: str,
    probe: int | str = "",
    source: int | str = "",
    target: int | str = "",
) -> dict[str, Any]:
    build_start = perf_counter()
    solver = condition.builder()
    build_seconds = perf_counter() - build_start
    if source != "" and target != "":
        solver.add(z3.Bool(f"p_{source}_{target}"))
    solver.set(timeout=max(1, round(timeout_seconds * 1_000)))
    solver.set(random_seed=(seed + int(probe or 0)) % (2**31 - 1))
    solve_start = perf_counter()
    result = solver.check()
    solve_seconds = perf_counter() - solve_start
    statistics = _solver_statistics(solver)
    timed_out = result == z3.unknown and solver.reason_unknown() == "timeout"
    r = n - k
    rx = r // 2
    return {
        "experiment": condition.experiment,
        "condition": condition.name,
        "sample": sample,
        "seed": seed,
        "n": n,
        "k": k,
        "r": r,
        "rx": rx,
        "rz": r - rx,
        "measurement": measurement,
        "probe": probe,
        "source": source,
        "target": target,
        "witness_target": condition.witness[int(source)] if source != "" else "",
        "result": str(result),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "build_seconds": f"{build_seconds:.9f}",
        "solve_seconds": f"{solve_seconds:.9f}",
        "decisions": int(statistics.get("decisions", 0)),
        "conflicts": int(statistics.get("conflicts", 0)),
        "propagations": int(statistics.get("propagations", 0)),
        "rlimit_count": int(statistics.get("rlimit count", 0)),
        "assertions": len(solver.assertions()),
        "row_operation_variables": condition.row_operation_variables,
    }


def _probe_assignments(
    n: int, witness: Sequence[int], probes: int, seed: int
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed + 1)
    sources = rng.choice(n, size=min(probes, n), replace=False)
    alternatives = [int(rng.integers(0, n - 1)) for _ in sources]
    result = []
    for source_value, alternative in zip(sources, alternatives, strict=True):
        source = int(source_value)
        witness_target = witness[source]
        target = alternative + (alternative >= witness_target)
        result.append((source, target))
    return result


def _collect_conditions(
    conditions: Sequence[Condition],
    *,
    sample: int,
    seed: int,
    n: int,
    k: int,
    probes: int,
    timeout_seconds: float,
    output: Path,
    completed: set[tuple[str, ...]],
    verbose: bool,
) -> None:
    assignments = _probe_assignments(n, conditions[0].witness, probes, seed)
    for condition in conditions:
        specifications: list[tuple[str, int | str, int | str, int | str]] = [
            ("base", "", "", "")
        ]
        if condition.probe_mappings:
            specifications.extend(
                ("invalid_mapping", probe, source, target)
                for probe, (source, target) in enumerate(assignments)
            )
        for measurement, probe, source, target in specifications:
            identity = {
                "experiment": condition.experiment,
                "condition": condition.name,
                "sample": sample,
                "seed": seed,
                "n": n,
                "k": k,
                "measurement": measurement,
                "probe": probe,
            }
            if _row_key(identity) in completed:
                continue
            row = _measure(
                condition,
                sample=sample,
                seed=seed,
                n=n,
                k=k,
                timeout_seconds=timeout_seconds,
                measurement=measurement,
                probe=probe,
                source=source,
                target=target,
            )
            _append_csv_row(output, row)
            completed.add(_row_key(row))
            if verbose:
                print(
                    f"A7 {condition.experiment} [[{n},{k}]] sample={sample} "
                    f"{condition.name} {measurement}: {row['result']}, "
                    f"decisions={row['decisions']}",
                    flush=True,
                )


def collect(
    *,
    experiment1_n: Sequence[int] = EXPERIMENT1_N,
    experiment2_n: Sequence[int] = EXPERIMENT2_N,
    k: int = K,
    samples: int = NUM_SAMPLES,
    probes: int = NUM_PROBES,
    master_seed: int = MASTER_SEED,
    timeout_seconds: float = TIMEOUT_SECONDS,
    output: Path = OUTPUT,
    verbose: bool = VERBOSE,
) -> None:
    if samples < 1 or probes < 0 or timeout_seconds <= 0:
        raise ValueError("samples and timeout must be positive; probes cannot be negative")
    completed = _completed_keys(output)
    experiments = (
        (EXPERIMENT1, experiment1_n, _experiment1_conditions),
        (EXPERIMENT2, experiment2_n, _experiment2_conditions),
    )
    for experiment, ns, factory in experiments:
        for n in ns:
            if not 0 <= k <= n - 2:
                raise ValueError(f"require 0 <= k <= n-2, got [[{n},{k}]]")
            if experiment == EXPERIMENT1 and (n - k) % 2:
                raise ValueError(
                    f"Experiment 1 requires balanced CSS ranks; n-k must be even, "
                    f"got [[{n},{k}]]"
                )
            for sample in range(samples):
                seed = _sample_seed(master_seed, experiment, n, sample)
                _collect_conditions(
                    factory(n, k, seed),
                    sample=sample,
                    seed=seed,
                    n=n,
                    k=k,
                    probes=probes,
                    timeout_seconds=timeout_seconds,
                    output=output,
                    completed=completed,
                    verbose=verbose,
                )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment1-n", nargs="+", type=int, default=EXPERIMENT1_N
    )
    parser.add_argument(
        "--experiment2-n", nargs="+", type=int, default=EXPERIMENT2_N
    )
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--probes", type=int, default=NUM_PROBES)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = _parse_args()
    collect(
        experiment1_n=arguments.experiment1_n,
        experiment2_n=arguments.experiment2_n,
        k=arguments.k,
        samples=arguments.samples,
        probes=arguments.probes,
        master_seed=arguments.seed,
        timeout_seconds=arguments.timeout,
        output=arguments.output,
        verbose=not arguments.quiet,
    )
