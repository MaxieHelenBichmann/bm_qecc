"""Command-line benchmark runner."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from unittest import case

import numpy as np

from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce
from src.algorithms.lc_css.lc_css_kls import is_lceq_css_kls
from src.algorithms.lc_css.lc_css_orbit import is_lceq_css_orbit
from src.algorithms.lc_css.lc_css_sat import is_lceq_css_sat
from src.algorithms.lc_eq.lc_eq_graph_state import are_lceq_graph_state
from src.algorithms.lc_eq.lc_eq_bruteforce import are_lceq_bruteforce
from src.algorithms.lc_eq.lc_eq_sat import are_lceq_sat
from src.algorithms.p_css.p_css_bruteforce import are_peq_css_bruteforce
from src.algorithms.p_css.p_css_classical import are_peq_css_classical
from src.algorithms.p_css.p_css_graph_iso import are_peq_css_graph_iso
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stab.p_stab_bruteforce import are_peq_stab_bruteforce
from src.algorithms.p_stab.p_stab_aut import are_peq_stab_aut
from src.algorithms.p_stab.p_stab_classical import are_peq_stab_classical
from src.algorithms.p_stab.p_stab_graph_iso import are_peq_stab_graph_iso
from src.algorithms.p_stab.p_stab_sat import are_peq_stab_sat

from src.core.stabilizer_code import StabilizerCode
from src.core.css_code import CSSCode

from .utils import (
    lc_equivalent_code,
    non_permutation_equivalent_css_code,
    non_permutation_equivalent_stabilizer_code,
    permutation_equivalent_code,
    permutation_equivalent_css_code,
    random_permuted_stabilizer_pair,
    random_permuted_css_pair,
    random_non_permuted_stabilizer_pair,
    random_non_permuted_css_pair,
    random_css_code,
)

N_STATS = 7
MEAS_STATS = [
    (3,1),
    (5,2),
    (7,2),
    (10,4),
    (13,3),
    (15,2),
    (17,5),
    (20,2)
]

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

@dataclass(frozen=True)
class Statistic:
    """One statistic result."""

    algorithm: str
    measurement: str
    times: list[float]
    mean: float
    stddev: float
    maximum: float


Algorithm = Callable[..., bool]

ALGORITHMS: dict[str, Algorithm] = {
    "pm_css_bruteforce": are_peq_css_bruteforce,
    "pm_css_classical": are_peq_css_classical,
    "pm_css_graph_iso": are_peq_css_graph_iso,
    "pm_css_matroid": are_peq_css_matroid,
    "pm_css_sat": are_peq_css_sat,
    "pm_stb_aut": are_peq_stab_aut,
    "pm_stb_bruteforce": are_peq_stab_bruteforce,
    "pm_stb_classical": are_peq_stab_classical,
    "pm_stb_graph_iso": are_peq_stab_graph_iso,
    "pm_stb_sat": are_peq_stab_sat,
    "lc_equ_graph_state": are_lceq_graph_state,
    "lc_equ_bruteforce": are_lceq_bruteforce,
    "lc_equ_sat": are_lceq_sat,
    "lc_css_bruteforce": is_lceq_css_bruteforce,
    "lc_css_kls": is_lceq_css_kls,
    "lc_css_orbit": is_lceq_css_orbit,
    "lc_css_sat": is_lceq_css_sat,
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def generated_stabilizer_pair(n: int, k: int, suffix: str) -> tuple[StabilizerCode, StabilizerCode] | None:
    """Load a generated stabilizer pair from data/ if both files exist."""
    base = f"random_stab_{n}_{k}"
    paths = (
        DATA_DIR / f"{base}1_{suffix}.txt",
        DATA_DIR / f"{base}2_{suffix}.txt",
    )
    if not all(path.exists() for path in paths):
        return None
    code1_path, code2_path = paths
    return StabilizerCode.from_file(code1_path), StabilizerCode.from_file(code2_path)

def generated_css_pair(n: int, k: int, suffix: str) -> tuple[CSSCode, CSSCode] | None:
    """Load a generated CSS pair from data/ if both files exist."""
    base = f"random_css_{n}_{k}"
    paths = (
        DATA_DIR / f"{base}1_{suffix}.txt",
        DATA_DIR / f"{base}2_{suffix}.txt",
    )
    if not all(path.exists() for path in paths):
        return None
    code1_path, code2_path = paths
    return CSSCode.from_file(code1_path), CSSCode.from_file(code2_path)


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


def resolve_algorithm_names(selectors: Sequence[str] | None) -> list[str]:
    """Expand exact algorithm names, shell wildcards, or regexes to registered names."""
    if not selectors:
        return all_algorithm_names()

    selected_names: set[str] = set()
    invalid_selectors: list[str] = []
    algorithm_names = all_algorithm_names()

    for selector in selectors:
        if selector in ALGORITHMS:
            selected_names.add(selector)
            continue

        matches = [name for name in algorithm_names if fnmatch.fnmatchcase(name, selector)]
        if not matches:
            try:
                pattern = re.compile(selector)
            except re.error as exc:
                invalid_selectors.append(f"{selector!r} (invalid regex: {exc})")
                continue
            matches = [name for name in algorithm_names if pattern.search(name)]

        if matches:
            selected_names.update(matches)
        else:
            invalid_selectors.append(f"{selector!r} (no matches)")

    if invalid_selectors:
        available = ", ".join(algorithm_names)
        invalid = ", ".join(invalid_selectors)
        raise ValueError(f"Unknown algorithm selector(s): {invalid}. Available algorithms: {available}")

    return sorted(selected_names)

def random_non_permuted_css_case(n: int, k: int, case_seed: int, use_cached: bool = True) -> Case:
        if use_cached:
            pair = generated_css_pair(n, k, "non_peq")
            code1, code2 = pair or random_non_permuted_css_pair(n, k, seed=case_seed)
        else:
            code1, code2 = random_non_permuted_css_pair(n, k, seed=case_seed)
        return Case(
            name=f"random_non_permuted_css_{n}_{case_seed}",
            inputs=(code1, code2),
            expected_p=False,
            expected_lc=None,
        )

def random_permuted_css_case(n: int, k: int, case_seed: int, use_cached: bool = True) -> Case:
    if use_cached:
        pair = generated_css_pair(n, k, "peq")
        code1, code2 = pair or random_permuted_css_pair(n, k, seed=case_seed)
    else:
        code1, code2 = random_permuted_css_pair(n, k, seed=case_seed)
    return Case(
        name=f"random_permuted_css_{n}_{k}_{case_seed}",
        inputs=(code1, code2),
        expected_p=True,
        expected_lc=None,
    )

def random_non_permuted_stabilizer_case(n: int, k: int, case_seed: int, use_cached: bool = True) -> Case:
    if use_cached:
        pair = generated_stabilizer_pair(n, k, "non_peq")
        code1, code2 = pair or random_non_permuted_stabilizer_pair(n, k, seed=case_seed)
    else:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=case_seed)
    return Case(
        name=f"random_non_permuted_stb_{n}_{k}_{case_seed}",
        inputs=(code1, code2),
        expected_p=False,
        expected_lc=None,
    )
    
def random_permuted_stabilizer_case(n: int, k: int, case_seed: int, use_cached: bool = True) -> Case:
    if use_cached:
        pair = generated_stabilizer_pair(n, k, "peq")
        code1, code2 = pair or random_permuted_stabilizer_pair(n, k, seed=case_seed)
    else:
        code1, code2 = random_permuted_stabilizer_pair(n, k, seed=case_seed)
    return Case(
        name=f"random_permuted_stb_{n}_{k}_{case_seed}",
        inputs=(code1, code2),
        expected_p=True,
        expected_lc=None,
    )

def random_lcc_css_case(n: int, k: int, case_seed: int) -> Case:
    code = random_css_code(n, k, seed=case_seed)
    return Case(
        name=f"random_lcc_css_{n}_{k}_{case_seed}",
        inputs=(lc_equivalent_code(code, seed=case_seed + 420),),
        expected_p=None,
        expected_lc=True,
    )

def random_lcc_eq_case(n: int, k: int, case_seed: int) -> Case:
    code1 = random_css_code(n, k, seed=case_seed)
    return Case(
        name=f"random_lcc_eq_{n}_{k}_{case_seed}",
        inputs=(code1, lc_equivalent_code(code1, seed=case_seed + 420)),
        expected_p=None,
        expected_lc=True,
    )

def known_non_permuted_css_case(code1: CSSCode, case_seed: int) -> Case:
        code2 = non_permutation_equivalent_css_code(code1, seed=case_seed)
        return Case(
            name=f"known_non_permuted_css_{case_seed}",
            inputs=(code1, code2),
            expected_p=False,
            expected_lc=None,
        )

def known_permuted_css_case(code1: CSSCode, case_seed: int) -> Case:
    code2 = permutation_equivalent_css_code(code1, seed=case_seed)
    return Case(
        name=f"known_permuted_css_{case_seed}",
        inputs=(code1, code2),
        expected_p=True,
        expected_lc=None,
    ) 

def known_non_permuted_stabilizer_case(code1: StabilizerCode, case_seed: int) -> Case:
    code2 = non_permutation_equivalent_stabilizer_code(code1, seed=case_seed)
    return Case(
        name=f"known_non_permuted_stb_{case_seed}",
        inputs=(code1, code2),
        expected_p=False,
        expected_lc=None,
    )
    
def known_permuted_stabilizer_case(code1: StabilizerCode, case_seed: int) -> Case:
    code2 = permutation_equivalent_code(code1, seed=case_seed)
    return Case(
        name=f"known_permuted_stb_{case_seed}",
        inputs=(code1, code2),
        expected_p=True,
        expected_lc=None,
    )

def known_lcc_css_case(code: StabilizerCode, case_seed: int) -> Case:
    return Case(
        name=f"known_lcc_css_{case_seed}",
        inputs=(lc_equivalent_code(code, seed=case_seed),),
        expected_p=None,
        expected_lc=True,
    )

def known_lcc_eq_case(code: StabilizerCode, case_seed: int) -> Case:
    code2 = lc_equivalent_code(code, seed=case_seed + 420)
    return Case(
        name=f"known_lcc_eq_{case_seed}",
        inputs=(code, code2),
        expected_p=None,
        expected_lc=True,
    )


def seeded_measurements(seed: int, algorithm: str, random: bool) -> list[tuple[str, list[Case]]]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000, size=N_STATS)
    measurements_pos : list[tuple[str, list[Case]]] = []
    measurements_neg : list[tuple[str, list[Case]]] = []

    if random:
        for n, k in MEAS_STATS:
            if algorithm.startswith("pm_css"):
                non_permuted_css = [random_non_permuted_css_case(n, k, s, False) for s in seeds]
                permuted_css = [random_permuted_css_case(n, k, s+69, False) for s in seeds]
                measurements_neg.append((f"non_permuted_css_{n}_{k}", non_permuted_css))
                measurements_pos.append((f"permuted_css_{n}_{k}", permuted_css))
            elif algorithm.startswith("pm_stb"):
                non_permuted_stabilizer = [random_non_permuted_stabilizer_case(n, k, s+1337, False) for s in seeds]
                permuted_stabilizer = [random_permuted_stabilizer_case(n, k, s+420, False) for s in seeds]
                measurements_neg.append((f"non_permuted_stab_{n}_{k}", non_permuted_stabilizer))
                measurements_pos.append((f"permuted_stab_{n}_{k}", permuted_stabilizer))
            elif algorithm.startswith("lc_equ"):
                lc_eq = [random_lcc_eq_case(n, k, s) for s in seeds]
                measurements_pos.append((f"lc_eq_{n}_{k}", lc_eq))
            elif algorithm.startswith("lc_css"):
                lc_cases = [random_lcc_css_case(n, k, s+13) for s in seeds]
                measurements_pos.append((f"lc_css_{n}_{k}", lc_cases))
    else:
        if algorithm.startswith("pm_css"):
            for name, code in [
                                ("bell_pair", bell_pair), # n = 2 , k = 0
                                ("3bit_repetition", three_bit_repetition), # n = 3 , k = 1 
                                ("steane", steane), # n = 7 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rotated_surface_d5", rotated_surface_d5)
                               ]:
                measurements_neg.append((name, [known_non_permuted_css_case(code, s) for s in seeds]))
                measurements_pos.append((name, [known_permuted_css_case(code, s) for s in seeds]))

        elif algorithm.startswith("pm_stb"):
            for name, code in [
                                ("bell_pair", bell_pair), # n = 2 , k = 0
                                ("3bit_repetition", three_bit_repetition), # n = 3 , k = 1
                                ("steane", steane), # n = 7 , k = 1
                                ("five_qubit_perfect", five_qubit_perfect), # n = 5 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rotated_surface_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements_neg.append((name, [known_non_permuted_stabilizer_case(code, s) for s in seeds]))
                measurements_pos.append((name, [known_permuted_stabilizer_case(code, s) for s in seeds]))
        elif algorithm.startswith("lc_equ"):
              for name, code in [
                                ("bell_pair", bell_pair), # n = 2 , k = 0
                                ("3bit_repetition", three_bit_repetition), # n = 3 , k = 1 
                                ("steane", steane), # n = 7 , k = 1
                                ("five_qubit_perfect", five_qubit_perfect), # n = 5 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rotated_surface_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements_pos.append((name, [known_lcc_eq_case(code, s) for s in seeds]))
        elif algorithm.startswith("lc_css"):
              for name, code in [
                                ("bell_pair", bell_pair),  # n = 2 , k = 0
                                ("3bit_repetition", three_bit_repetition), # n = 3 , k = 1
                                ("steane", steane), # n = 7 , k = 1
                                ("five_qubit_perfect", five_qubit_perfect), # n = 5 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rotated_surface_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements_pos.append((name, [known_lcc_css_case(code, s) for s in seeds]))

    return measurements_pos + measurements_neg
    
def default_cases(seed: int, random: bool = False) -> list[Case]:
    """Return test cases."""
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

    case_five_qubits_lc_only = Case(
            name="five_qubits_lc_only",
            inputs=(five_qubit_perfect, lc_equivalent_code(five_qubit_perfect, seed=seed + 3)),
            expected_p=None,
            expected_lc=True,
    )

    case_shor_lc_only = Case(
            name="shor_lc_only",
            inputs=(shor, lc_equivalent_code(shor, seed=seed + 69)),
            expected_p=None, 
            expected_lc=True,
    )

    case_shor_lc_css = Case(
            name="shor_lc_css",
            inputs=tuple([lc_equivalent_code(shor, seed=seed + 1337)]),
            expected_p=None, 
            expected_lc=True,
    )

    case_three_qubits_lc_css = Case(
            name="three_qubits_lc_css",
            inputs=tuple([lc_equivalent_code(three_bit_repetition, seed=seed + 1337)]),
            expected_p=None, 
            expected_lc=True,
    )

    case_steane_lc_css = Case(
            name="steane_lc_css",
            inputs=tuple([lc_equivalent_code(steane, seed=seed + 1337)]),
            expected_p=None, 
            expected_lc=True,
    )

    known_permuted = [
        case_bell_pair_same, # n = 2 , k = 0
        case_three_qubits_permuted, # n = 3 , k = 1
        case_steane_permuted, # n = 7 , k = 1
        case_shor_permuted, # n = 9 , k = 1
        case_carbon_permuted, # n = 12 , k = 2
        case_tetrahedral_permuted, # n = 15 , k = 1
        case_hamming_15_permuted, # n = 15 , k = 7
        case_golay_permuted, # n = 23 , k = 1
        case_rotated_surface_d5_permuted, # n = 25 , k = 1
    ]

    random_permuted_css = [
        random_permuted_css_case(3, 1, seed + 1),
        random_permuted_css_case(4, 2, seed + 1),
        random_permuted_css_case(5, 2, seed + 2),
        random_permuted_css_case(6, 3, seed + 3),
        random_permuted_css_case(7, 2, seed + 7),
        random_permuted_css_case(8, 3, seed + 69), 
        random_permuted_css_case(9, 5, seed + 420),
        random_permuted_css_case(10, 4, seed), 
    ]

    random_non_permuted_css = [
        random_non_permuted_css_case(3, 1, seed + 1),
        random_non_permuted_css_case(5, 2, seed + 20), 
        random_non_permuted_css_case(7, 2, seed + 42),
        random_non_permuted_css_case(9, 5, seed + 1337),
    ]

    known_lc = [
        case_five_qubits_lc_only, # n = 5 , k = 1
        case_shor_lc_only, # n = 9 , k = 1 
    ]

    known_lc_css = [
        case_three_qubits_lc_css, # n = 3 , k = 1
        case_steane_lc_css, # n = 7 , k = 1
        case_shor_lc_css, # n = 9 , k = 1
    ]

    random_lc_css = [
        random_lcc_css_case(10, 4, seed + 69),
    ]

    return random_permuted_css + random_non_permuted_css + random_lc_css if random else known_permuted + known_lc + known_lc_css


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


def run_raw_benchmarks(cases: Sequence[Case], algorithm_names: Sequence[str], repeats: int, verbose: bool = True) -> list[Result]:
    """Run selected algorithms on cases with the matching problem type."""
    results: list[Result] = []
    selected_names = set(algorithm_names)

    for algorithm_name in sorted(selected_names & ALGORITHMS.keys()):
        if verbose:
            print(f"Running benchmark for algorithm: {algorithm_name}")
        result_algorithm = []
        for case in cases:
            if not case_supports_algorithm(case, algorithm_name):
                continue
            if verbose:
                print(f"    Running case: {case.name}...")
            result_algorithm.append(run_case(algorithm_name, ALGORITHMS[algorithm_name], case, repeats))

        if verbose:
            print_results(result_algorithm)

        results.extend(result_algorithm)

    return results

def run_stat_benchmarks(algorithm_names: Sequence[str], repeats: int, seed: int, verbose: bool = True, random: bool = False) -> list[Statistic]:
    """Run selected algorithms on cases with the matching problem type."""
    statistics: list[Statistic] = []
    selected_names = set(algorithm_names)

    for algorithm_name in sorted(selected_names & ALGORITHMS.keys()):
        if verbose:
            print(f"Running benchmark for algorithm: {algorithm_name}")
        stats_algorithm = []
        for measurement_name, measurement in seeded_measurements(seed=seed, algorithm=algorithm_name, random=random):
            if verbose:
                print(f"    Running measurement: {measurement_name}")
            results: list[Result] = []
            for case in measurement:
                if not case_supports_algorithm(case, algorithm_name):
                    continue
                if verbose:
                    print(f"        Running case: {case.name}...")
                results.append(run_case(algorithm_name, ALGORITHMS[algorithm_name], case, repeats))

            stat = compute_statistics(results, algorithm_name, measurement_name)

            if stat is not None:
                stats_algorithm.append(stat)

        if verbose:
            print_statistics(stats_algorithm)

        statistics.extend(stats_algorithm)
    
    return statistics

def compute_statistics(results: Sequence[Result], algorithm_name: str, measurement_name: str) -> Statistic | None:
    """Compute mean and standard deviation of runtimes for each algorithm and case."""
    times = []

    for result in results:
        if not result.success:
            print(f"Warning: Skipping failed case {result.case} for algorithm {algorithm_name} in statistics.")
            continue
        if result.algorithm != algorithm_name:
            continue

        times.append(result.seconds)

    if not times:
        return None
    
    mean = np.mean(times)
    stddev = np.std(times, ddof=1) if len(times) > 1 else 0.0

    return Statistic(
        algorithm=algorithm_name,
        measurement=measurement_name,
        times=times,
        mean=mean,
        stddev=stddev,
        maximum=max(times),
    )
    
def write_stats(stats: Sequence[Statistic], seed: int, output: Path) -> None:
    """Write benchmark statistics to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_short = [
        {
            "algorithm": stat.algorithm,
            "measurement": stat.measurement,
            "mean_seconds": f"{stat.mean:.9f}",
            "stddev_seconds": f"{stat.stddev:.9f}",
            "maximum_seconds": f"{stat.maximum:.9f}",
        }
        for stat in stats
    ]

    if len(rows_short) == 0:
        return

    with output.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([seed])
        writer = csv.DictWriter(file, fieldnames=list(rows_short[0].keys()))
        writer.writeheader()
        writer.writerows(rows_short)

    # ---
    output_raw = output.with_name(output.stem + "_raw" + output.suffix)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    rows_raw = [
        {
            "seed": seed,
            "algorithm": stat.algorithm,
            "measurement": stat.measurement,
            "sample": i,
            "time_seconds": f"{time:.9f}",
        }
        for stat in stats
        for i, time in enumerate(stat.times, start=1)
    ]

    if len(rows_raw) == 0:
        return

    with output_raw.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_raw[0].keys()))
        writer.writeheader()
        writer.writerows(rows_raw)

def write_bms(results: Sequence[Result], seed: int, output: Path) -> None:
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

    if len(rows) == 0:
        return

    with output.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([seed])
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_statistics(statistics: Sequence[Statistic]) -> None:
    """Print benchmark statistics to console."""
    if len(statistics) == 0:
        print("No statistics to show.")
        return
    
    print()
    print(f"Benchmark statistics:\n")

    for stat in statistics:
        print(
            f"{stat.algorithm:24} {stat.measurement:42} "
            f"mean={stat.mean:.6f}s stddev={stat.stddev:.6f}s max={stat.maximum:.6f}s"
        )

def print_results(results: Sequence[Result]) -> None:
    """Print benchmark results to console."""
    if len(results) == 0:
        print("No cases ran, no results to show.")
        return

    print(f"Benchmark results:\n")

    for result in results:
        status = "ok" if result.success else "failed"
        print(
            f"{status:6} {result.algorithm:24} {result.case:42} "
            f"n={result.n:<2} k={result.k:<2} t={result.seconds:.6f}s"
        )
        if result.error:
            print(f"       {result.error}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Available algorithms: " + ", ".join(all_algorithm_names()),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions; the average runtime is recorded.")
    parser.add_argument(
        "--algorithm",
        action="append",
        metavar="SELECTOR",
        help=(
            "Algorithm name, shell wildcard, or regex to run. Can be passed multiple times. "
            "Examples: pm_css_bruteforce, 'pm_css*', 'lc_(equ|css).*'. "
            "Defaults to all implemented algorithms."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/latest.csv"), help="CSV output path.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--stats", action="store_true", default=False, help="Execute the algorithm on the cases with different seeds and print the statistics of the runtime.")
    parser.add_argument("--stats_output", type=Path, default=Path("results/statistics.csv"), help="CSV output path for statistics.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed results updates.")
    parser.add_argument("--random", action="store_true", default=False, help="Use randomly generated cases instead of fixed ones.")
    args = parser.parse_args(argv)
    try:
        args.algorithm = resolve_algorithm_names(args.algorithm)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    if args.stats:
        statistics = run_stat_benchmarks(args.algorithm, args.repeats, args.seed, args.verbose, args.random)
        write_stats(statistics, args.seed, args.stats_output)
        return 0
                                      
    else:
        results = run_raw_benchmarks(default_cases(seed=args.seed, random=args.random), args.algorithm, args.repeats, verbose=args.verbose)
        write_bms(results, args.seed, args.output)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
