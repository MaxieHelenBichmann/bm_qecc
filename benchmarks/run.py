"""Command-line benchmark runner."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.algorithms.lc_css_bruteforce import is_lceq_css_bruteforce
from src.algorithms.lc_css_graph_state import is_lceq_css_graph_state
from src.algorithms.lc_css_kls import is_lceq_css_kls
from src.algorithms.lc_css_orbit import is_lceq_css_orbit
from src.algorithms.lc_eq_graph_state import are_lceq_graph_state
from src.algorithms.p_css_bruteforce import are_peq_css_bruteforce
from src.algorithms.p_css_classical import are_peq_css_classical
from src.algorithms.p_css_graph_iso import are_peq_css_graph_iso
from src.algorithms.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_stab_bruteforce import are_peq_stab_bruteforce
from src.algorithms.p_stab_sat import are_peq_stab_sat

from src.core.stabilizer_code import StabilizerCode
from src.core.css_code import CSSCode

from .utils import (
    lc_equivalent_code,
    permutation_equivalent_css_code,
    random_permuted_stabilizer_pair,
    random_permuted_css_pair,
    random_non_permuted_stabilizer_pair,
    random_non_permuted_css_pair,
)

@dataclass(frozen=True)
class Case:
    """One benchmark input."""

    name: str
    inputs: tuple[StabilizerCode, ...]
    expected_p: bool | None = None
    expected_lc: bool | None = None


@dataclass(frozen=True)
class Result:
    """One benchmark measurement."""
    algorithm: str
    case: str
    n: int
    k: int
    seconds: float
    result: bool | None
    expected: bool | None
    success: bool
    error: str = ""


Algorithm = Callable[..., bool]

ALGORITHMS: dict[str, Algorithm] = {
    "pm_css_bruteforce": are_peq_css_bruteforce,
    "pm_css_classical": are_peq_css_classical,
    "pm_css_graph_iso": are_peq_css_graph_iso,
    "pm_css_matroid": are_peq_css_matroid,
    "pm_stb_bruteforce": are_peq_stab_bruteforce,
    "pm_stb_sat": are_peq_stab_sat,
    "lc_equ_graph_state": are_lceq_graph_state,
    "lc_css_bruteforce": is_lceq_css_bruteforce,
    "lc_css_graph_state": is_lceq_css_graph_state,
    "lc_css_kls": is_lceq_css_kls,
    "lc_css_orbit": is_lceq_css_orbit,
}

def case_supports_algorithm(case: Case, algorithm_name: str) -> bool:
    """Return whether a case has an expectation and compatible inputs for an algorithm."""
    if algorithm_name.startswith("pm_css") and all(isinstance(code, CSSCode) for code in case.inputs)and len(case.inputs) == 2 and case.expected_p is not None:
        return True
    if algorithm_name.startswith("pm_stb") and len(case.inputs) == 2 and case.expected_p is not None:
        return True
    if algorithm_name.startswith("lc_equ") and len(case.inputs) == 2 and case.expected_lc is not None:
        return True
    if algorithm_name.startswith("lc_css") and len(case.inputs) == 1 and case.expected_lc is not None:
        return True
    return False


def all_algorithm_names() -> list[str]:
    """Return all registered algorithm names."""
    return sorted({name for name in ALGORITHMS.keys()})


def default_cases(seed: int) -> list[Case]:
    """Return test cases."""
    bell_pair = CSSCode(Hz=np.array([[1, 1]], dtype=np.int8))

    three_bit_repetition = CSSCode.from_file("data/three_bit_repetition")
    steane = CSSCode.from_file("data/steane")
    five_qubit_perfect = StabilizerCode.from_file("data/five_qubit_perfect")
    carbon = CSSCode.from_file("data/carbon")
    golay = CSSCode.from_file("data/golay")
    hamming_15 = CSSCode.from_file("data/hamming_15")
    rotated_surface_d5 = CSSCode.from_file("data/rotated_surface_d5")
    shor = CSSCode.from_file("data/shor")
    tetrahedral = CSSCode.from_file("data/tetrahedral")

    random_case_stab1, random_case_stab2 = random_permuted_stabilizer_pair(4, 2, seed=seed)
    random_case_css1, random_case_css2 = random_permuted_css_pair(8, 3, seed=seed + 1)
    random_case_non_permuted_css1 , random_case_non_permuted_css2 = random_non_permuted_css_pair(10, 4, seed=seed + 2)
    random_case_non_permuted_stab1 , random_case_non_permuted_stab2 = random_non_permuted_stabilizer_pair(10, 4, seed=seed + 3)


    # ---------------------

    case_bell_pair_same = Case(
            name="bell_pair_same",
            inputs=(bell_pair, bell_pair),
            expected_p=True,
            expected_lc=True,
    )

    case_three_qubits_permuted = Case(
            name="three_qubits_permuted", 
            inputs=(three_bit_repetition, permutation_equivalent_css_code(three_bit_repetition, seed=seed + 1)),
            expected_p=True,
            expected_lc=None,
    )

    case_steane_permuted = Case(
            name="steane_permuted", 
            inputs=(steane, permutation_equivalent_css_code(steane, seed=seed + 2)),
            expected_p=True,
            expected_lc=None,
    )

    case_shor_permuted = Case(
            name="shor_permuted", 
            inputs=(shor, permutation_equivalent_css_code(shor, seed=seed + 7)),
            expected_p=True,
            expected_lc=None,
    )

    case_carbon_permuted = Case(
            name="carbon_permuted",
            inputs=(carbon, permutation_equivalent_css_code(carbon, seed=seed + 3)),
            expected_p=True,
            expected_lc=None,
    )

    case_tetrahedral_permuted = Case(
            name="tetrahedral_permuted", 
            inputs=(tetrahedral, permutation_equivalent_css_code(tetrahedral, seed=seed + 8)),
            expected_p=True,
            expected_lc=None,
    )

    case_hamming_15_permuted = Case(
            name="hamming_15_permuted", 
            inputs=(hamming_15, permutation_equivalent_css_code(hamming_15, seed=seed + 5)),
            expected_p=True,
            expected_lc=None,
    )

    case_golay_permuted = Case(
            name="golay_permuted", 
            inputs=(golay, permutation_equivalent_css_code(golay, seed=seed + 4)),
            expected_p=True,
            expected_lc=None,
    )

    case_rotated_surface_d5_permuted = Case(
            name="rotated_surface_d5_permuted", 
            inputs=(rotated_surface_d5, permutation_equivalent_css_code(rotated_surface_d5, seed=seed + 6)),
            expected_p=True,
            expected_lc=None,
    )

    case_random_permuted_css = Case(
            name="random_permuted_stb",
            inputs=(random_case_stab1, random_case_stab2),
            expected_p=True,
            expected_lc=None,
    )

    case_random_permuted_stb = Case(
            name="random_permuted_css", 
            inputs=(random_case_css1, random_case_css2),
            expected_p=True,
            expected_lc=None,
    )

    case_random_non_permuted_css = Case(
            name="random_non_permuted_css",
            inputs=(random_case_non_permuted_css1, random_case_non_permuted_css2),
            expected_p=False,
            expected_lc=None,
    )

    case_random_non_permuted_stb = Case(
            name="random_non_permuted_stb", 
            inputs=(random_case_non_permuted_stab1, random_case_non_permuted_stab2),
            expected_p=False,
            expected_lc=None,
    )

    case_five_qubits_lc_only = Case(
            name="five_qubits_lc_only",
            inputs=(five_qubit_perfect, lc_equivalent_code(five_qubit_perfect, seed=seed + 3)),
            expected_p=None,
            expected_lc=True,
    )

    return [
        case_bell_pair_same, # n = 2 , k = 0
        case_three_qubits_permuted, # n = 3 , k = 1
        case_steane_permuted, # n = 7 , k = 1
        case_shor_permuted, # n = 9 , k = 1
        case_carbon_permuted, # n = 12 , k = 2
        case_tetrahedral_permuted, # n = 15 , k = 1
        case_hamming_15_permuted, # n = 15 , k = 7
        case_golay_permuted, # n = 23 , k = 1
        case_rotated_surface_d5_permuted, # n = 25 , k = 1
        case_random_permuted_css, # n = 4 , k = 2
        case_random_permuted_stb, # n = 8 , k = 3
        case_random_non_permuted_css,  # n = 10 , k = 4
        case_random_non_permuted_stb, # n = 10 , k = 4
        case_five_qubits_lc_only,  # n = 5 , k = 1

    ]


def run_case(algorithm_name: str, algorithm: Algorithm, case: Case, repeats: int) -> Result:
    """Run one algorithm on one case and return the average runtime."""
    total_seconds = 0.0
    last_result: bool | None = None
    expected = case.expected_p if algorithm_name.startswith("pm") else (case.expected_lc if algorithm_name.startswith("lc") else None)

    try:
        for _ in range(repeats):
            start = perf_counter()
            last_result = algorithm(*case.inputs)
            total_seconds += perf_counter() - start
        success = expected is not None and last_result == expected
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=total_seconds / repeats,
            result=last_result,
            expected=expected,
            success=success,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=0.0,
            result=None,
            expected=expected,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_benchmarks(cases: Sequence[Case], algorithm_names: Sequence[str], repeats: int) -> list[Result]:
    """Run selected algorithms on cases with the matching problem type."""
    results: list[Result] = []
    selected_names = set(algorithm_names)

    for case in cases:
        for algorithm_name in sorted(selected_names & ALGORITHMS.keys()):
            if not case_supports_algorithm(case, algorithm_name):
                continue
            results.append(run_case(algorithm_name, ALGORITHMS[algorithm_name], case, repeats))
    return results


def write_csv(results: Sequence[Result], seed: int, output: Path) -> None:
    """Write benchmark results to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "algorithm": result.algorithm,
            "case": result.case,
            "n": result.n,
            "k": result.k,
            "seconds": f"{result.seconds:.9f}",
            "result": "" if result.result is None else result.result,
            "expected": "" if result.expected is None else result.expected,
            "success": result.success,
            "error": result.error,
        }
        for result in results
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([seed])
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions; the average runtime is recorded.")
    parser.add_argument(
        "--algorithm",
        choices=all_algorithm_names(),
        action="append",
        help="Algorithm to run. Can be passed multiple times. Defaults to all implemented algorithms.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/latest.csv"), help="CSV output path.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

   
    algorithm_names = args.algorithm or all_algorithm_names()
    results = run_benchmarks(default_cases(seed=args.seed), algorithm_names, args.repeats)
    write_csv(results, args.seed, args.output)
    print(f"Benchmark results for global seed {args.seed}:\n")

    for result in results:
        status = "ok" if result.success else "failed"
        print(
            f"{status:6} {result.algorithm:24} {result.case:42} "
            f"n={result.n:<2} k={result.k:<2} t={result.seconds:.6f}s"
        )
        if result.error:
            print(f"       {result.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
