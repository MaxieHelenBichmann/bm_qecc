"""Command-line benchmark runner."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import multiprocessing as mp
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Empty
from time import perf_counter

import numpy as np

from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce
from src.algorithms.lc_css.lc_css_kls import is_lceq_css_kls
from src.algorithms.lc_css.lc_css_cliff_orbit import is_lceq_css_cliff_orbit
from src.algorithms.lc_css.lc_css_lc_orbit import is_lceq_css_lc_orbit
from src.algorithms.lc_css.lc_css_sat import is_lceq_css_sat
from src.algorithms.lc_eq.lc_eq_graph_state import are_lceq_graph_state
from src.algorithms.lc_eq.lc_eq_graph_state_small_k import are_lceq_graph_state_small_k
from src.algorithms.lc_eq.lc_eq_bruteforce import are_lceq_bruteforce
from src.algorithms.lc_eq.lc_eq_sat import are_lceq_sat
from src.algorithms.lc_eq.lc_eq_graph_iso import are_lceq_graph_iso
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

from src.hybrids.p_css import are_peq_css
from src.hybrids.p_stab import are_peq_stab
from src.hybrids.lc_eq import are_lceq
from src.hybrids.lc_css import is_lceq_css

from src.invariants.lc_eq.lc_invariants import (
    preserved_local_weight_distribution,
    preserved_low_degree_local_invariant
)

from src.invariants.p_eq.pm_invariants import (
    preserved_weight_enumerator,
    preserved_pauli_weight_enumerator,
    preserved_linear_dependencies,
)


from src.core.stabilizer_code import StabilizerCode
from src.core.css_code import CSSCode

from .utils import (
    lc_equivalent_code,
    non_lc_equivalent_code,
    non_permutation_equivalent_css_code,
    non_permutation_equivalent_stabilizer_code,
    permutation_equivalent_code,
    permutation_equivalent_css_code,
    random_permuted_stabilizer_pair,
    random_permuted_css_pair,
    random_non_permuted_stabilizer_pair,
    random_non_permuted_css_pair,
    random_css_code,
    random_stabilizer_code,
)

N_STATS = 10
MEAS_STATS = [
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    17,
    20
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
class Measurement:
    """All meta-data for a seeded measurement. Later usable for statistics."""
    algorithm: str
    name: str | None
    n: int
    k: int
    positive: bool
    density: float | None
    symmetry: float | None

@dataclass(frozen=True)
class Statistic:
    """One statistic result."""
    meta: Measurement
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
    "lc_equ_graph_state_small_k": are_lceq_graph_state_small_k,
    "lc_equ_bruteforce": are_lceq_bruteforce,
    "lc_equ_graph_iso": are_lceq_graph_iso,
    "lc_equ_sat": are_lceq_sat,
    "lc_css_bruteforce": is_lceq_css_bruteforce,
    "lc_css_kls": is_lceq_css_kls,
    "lc_css_cliff_orbit": is_lceq_css_cliff_orbit,
    "lc_css_lc_orbit": is_lceq_css_lc_orbit,
    "lc_css_sat": is_lceq_css_sat,

    "pm_css_hybrid": are_peq_css,
    "pm_stb_hybrid": are_peq_stab,
    "lc_eq_hybrid": are_lceq,
    "lc_css_hybrid": is_lceq_css,
}

LC_INVARIANTS: dict[str, Algorithm] = {
    "lc_local_weight_distribution": preserved_local_weight_distribution,
    "lc_local_weight_distribution_s2": partial(preserved_local_weight_distribution, max_subset_size=2),
    "lc_local_weight_distribution_s4": partial(preserved_local_weight_distribution, max_subset_size=4),
    "lc_low_degree_local_invariant": preserved_low_degree_local_invariant,
    "lc_low_degree_local_invariant_s2": partial(preserved_low_degree_local_invariant, max_subset_size=2),
    "lc_low_degree_local_invariant_s4": partial(preserved_low_degree_local_invariant, max_subset_size=4),
}

PM_INVARIANTS: dict[str, Algorithm] = {
    "pm_weight_enumerator": preserved_weight_enumerator,
    "pm_pauli_weight_enumerator": preserved_pauli_weight_enumerator,
    "pm_linear_dependencies": preserved_linear_dependencies,
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _algorithm_worker(algorithm: Algorithm, inputs: tuple[StabilizerCode, ...], queue: mp.Queue) -> None:
    """Run one benchmark repeat in a child process."""
    try:
        queue.put(("result", algorithm(*inputs)))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


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
        if algorithm_name == "lc_equ_graph_state_small_k":
            return all(isinstance(code, StabilizerCode) and code.k < 2 for code in case.inputs)
        return True
    if algorithm_name.startswith("lc_css") and len(case.inputs) == 1 and case.expected_lc is not None:
        if algorithm_name == "lc_css_lc_orbit":
            return case.inputs[0].k < 2
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

def non_permuted_css_case(seed: int, dim: tuple[int, int] | None = None, code: CSSCode | None = None, use_cached: bool = True) -> Case:
    if dim is not None:
        n, k = dim
        if use_cached:
            pair = generated_css_pair(n, k, "non_peq")
            code1, code2 = pair or random_non_permuted_css_pair(n, k, seed=seed)
        else:
            code1, code2 = random_non_permuted_css_pair(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        code1 = code
        code2 = non_permutation_equivalent_css_code(code1, seed=seed)

    return Case(
        name=f"non_permuted_css_{n}_{k}_{seed}",
        inputs=(code1, code2),
        expected_p=False,
        expected_lc=None,
    )

def permuted_css_case(seed: int, dim: tuple[int, int] | None = None, code: CSSCode | None = None, use_cached: bool = True) -> Case:
    if dim is not None:
        n, k = dim
        if use_cached:
            pair = generated_css_pair(n, k, "peq")
            code1, code2 = pair or random_permuted_css_pair(n, k, seed=seed)
        else:
            code1, code2 = random_permuted_css_pair(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        code1 = code
        code2 = permutation_equivalent_css_code(code1, seed=seed)

    return Case(
        name=f"permuted_css_{n}_{k}_{seed}",
        inputs=(code1, code2),
        expected_p=True,
        expected_lc=None,
    )

def non_permuted_stabilizer_case(seed: int, dim: tuple[int, int] | None = None, code: StabilizerCode | None = None, use_cached: bool = True) -> Case:
    if dim is not None:
        n, k = dim
        if use_cached:
            pair = generated_stabilizer_pair(n, k, "non_peq")
            code1, code2 = pair or random_non_permuted_stabilizer_pair(n, k, seed=seed)
        else:
            code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        code1 = code
        code2 = non_permutation_equivalent_stabilizer_code(code1, seed=seed)

    return Case(
        name=f"non_permuted_stb_{n}_{k}_{seed}",
        inputs=(code1, code2),
        expected_p=False,
        expected_lc=None,
    )
    
def permuted_stabilizer_case(seed: int, dim: tuple[int, int] | None = None, code: StabilizerCode | None = None, use_cached: bool = True) -> Case:
    if dim is not None:
        n, k = dim
        if use_cached:
            pair = generated_stabilizer_pair(n, k, "peq")
            code1, code2 = pair or random_permuted_stabilizer_pair(n, k, seed=seed)
        else:
            code1, code2 = random_permuted_stabilizer_pair(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        code1 = code
        code2 = permutation_equivalent_code(code1, seed=seed)

    return Case(
        name=f"permuted_stb_{n}_{k}_{seed}",
        inputs=(code1, code2),
        expected_p=True,
        expected_lc=None,
    )

def lcc_css_case(seed: int, dim: tuple[int, int] | None = None, code: CSSCode | None = None) -> Case:
    if dim is not None:
        n, k = dim
        code = random_css_code(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        
    return Case(
        name=f"lcc_css_{n}_{k}_{seed}",
        inputs=(lc_equivalent_code(code, seed=seed + 420),),
        expected_p=None,
        expected_lc=True,
    )

def non_lcc_css_case(seed: int, dim: tuple[int, int] | None = None, code: CSSCode | None = None) -> Case:
    if dim is not None:
        n, k = dim
        code = random_css_code(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k

    return Case(
        name=f"non_lcc_css_{n}_{k}_{seed}",
        inputs=(non_lc_equivalent_code(code, seed=seed + 69),),
        expected_p=None,
        expected_lc=False,
    )

def lcc_eq_case(seed: int, dim: tuple[int, int] | None = None, code: StabilizerCode | None = None) -> Case:
    if dim is not None:
        n, k = dim
        code1 = random_stabilizer_code(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        code1 = code

    return Case(
        name=f"lcc_eq_{n}_{k}_{seed}",
        inputs=(code1, lc_equivalent_code(code1, seed=seed + 1337)),
        expected_p=None,
        expected_lc=True,
    )

def non_lcc_eq_case(seed: int, dim: tuple[int, int] | None = None, code: StabilizerCode | None = None) -> Case:
    if dim is not None:
        n, k = dim
        code1 = random_stabilizer_code(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        code1 = code

    return Case(
        name=f"non_lcc_eq_{n}_{k}_{seed}",
        inputs=(code1, non_lc_equivalent_code(code1, seed=seed + 69)),
        expected_p=None,
        expected_lc=False,
    )

def seeded_measurements(seed: int, algorithm: str, random: bool) -> list[tuple[Measurement, list[Case]]]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000, size=N_STATS)
    measurements : list[tuple[Measurement, list[Case]]] = []
    sizes = [(n, i) for n in MEAS_STATS for i in range(0, n, 1 if n < 7 else 2)]

    if random:
        for n, k in sizes:
            if algorithm.startswith("pm_css"):
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [non_permuted_css_case(seed=s, dim=(n, k), use_cached=False) for s in seeds]
                                    ))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [permuted_css_case(seed=s, dim=(n, k), use_cached=False) for s in seeds]))
            elif algorithm.startswith("pm_stb"):
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [non_permuted_stabilizer_case(seed=s, dim=(n, k), use_cached=False) for s in seeds]))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [permuted_stabilizer_case(seed=s, dim=(n, k), use_cached=False) for s in seeds]))
            elif algorithm.startswith("lc_equ"):
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [lcc_eq_case(seed=s, dim=(n, k)) for s in seeds]))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [non_lcc_eq_case(seed=s, dim=(n, k)) for s in seeds]))
            elif algorithm.startswith("lc_css"):
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                    [lcc_css_case(seed=s, dim=(n, k)) for s in seeds]))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                    [non_lcc_css_case(seed=s, dim=(n, k)) for s in seeds]))
    else:
        if algorithm.startswith("pm_css"):
            for name, code in [
                                ("bell", bell_pair), # n = 2 , k = 0
                                ("3q_rep", three_bit_repetition), # n = 3 , k = 1 
                                ("steane", steane), # n = 7 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rot_surf_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ),
                                    [permuted_css_case(seed=s, code=code) for s in seeds]
                                    ))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ),
                                    [non_permuted_css_case(seed=s, code=code) for s in seeds]
                                    ))

        elif algorithm.startswith("pm_stb"):
            for name, code in [
                                ("bell", bell_pair), # n = 2 , k = 0
                                ("3q_rep", three_bit_repetition), # n = 3 , k = 1
                                ("5q_prf", five_qubit_perfect), # n = 5 , k = 1
                                ("steane", steane), # n = 7 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rot_surf_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [permuted_stabilizer_case(seed=s, code=code) for s in seeds]))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [non_permuted_stabilizer_case(seed=s, code=code) for s in seeds]))
        elif algorithm.startswith("lc_equ"):
            for name, code in [
                                ("bell", bell_pair), # n = 2 , k = 0
                                ("3q_rep", three_bit_repetition), # n = 3 , k = 1 
                                ("5q_prf", five_qubit_perfect), # n = 5 , k = 1
                                ("steane", steane), # n = 7 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rot_surf_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [lcc_eq_case(seed=s, code=code) for s in seeds]))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [non_lcc_eq_case(seed=s, code=code) for s in seeds]))
        elif algorithm.startswith("lc_css"):
              for name, code in [
                                ("bell", bell_pair),  # n = 2 , k = 0
                                ("3q_rep", three_bit_repetition), # n = 3 , k = 1
                                ("5q_prf", five_qubit_perfect), # n = 5 , k = 1
                                ("steane", steane), # n = 7 , k = 1
                                ("shor", shor),  # n = 9 , k = 1
                                ("carbon", carbon), # n = 12 , k = 2
                                ("tetrahedral", tetrahedral), # n = 15 , k = 1
                                ("hamming_15", hamming_15), # n = 15 , k = 7
                                ("golay", golay),  # n = 23 , k = 1
                                ("rot_surf_d5", rotated_surface_d5) # n = 25 , k = 1
                               ]:
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [lcc_css_case(seed=s, code=code) for s in seeds]))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    [non_lcc_css_case(seed=s, code=code) for s in seeds]))

    return measurements
    
def pm_invariant_cases(seed: int) -> list[Case]:
    return [
        permuted_stabilizer_case(seed=seed + 1, dim=(2, 0)),
        permuted_stabilizer_case(seed=seed + 2, dim=(3, 1)),
        permuted_stabilizer_case(seed=seed + 3, code=five_qubit_perfect),
        permuted_stabilizer_case(seed=seed + 4, dim=(5, 2)),
        permuted_stabilizer_case(seed=seed + 7, dim=(7, 2)),
        permuted_stabilizer_case(seed=seed + 8, dim=(9, 5)),
        permuted_stabilizer_case(seed=seed + 9, dim=(10, 4)),
        permuted_stabilizer_case(seed=seed + 10, dim=(10, 4)),
        permuted_stabilizer_case(seed=seed + 16, dim=(15, 8)),
        permuted_stabilizer_case(seed=seed + 18, dim=(12, 6)),
        permuted_stabilizer_case(seed=seed + 19, dim=(15, 8)),
        permuted_stabilizer_case(seed=seed + 24, dim=(23, 10)),
        permuted_stabilizer_case(seed=seed + 25, dim=(25, 12)),
        permuted_stabilizer_case(seed=seed + 26, dim=(18, 6)),
        permuted_stabilizer_case(seed=seed + 27, dim=(20, 8)),
        permuted_stabilizer_case(seed=seed + 28, dim=(20, 4)),
        non_permuted_stabilizer_case(seed=seed + 1, dim=(7, 2), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 2, dim=(9, 5), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 3, dim=(3, 1), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 5, dim=(5, 2), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 6, dim=(4, 1), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 10, dim=(7, 2), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 11, dim=(9, 5), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 12, dim=(10, 4), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 13, dim=(7, 3), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 15, dim=(10, 4), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 20, dim=(12, 6), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 21, dim=(15, 7), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 22, dim=(12, 6), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 23, dim=(15, 8), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 1, dim=(16, 8), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 2, dim=(18, 8), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 3, dim=(16, 8), use_cached=False),
        non_permuted_stabilizer_case(seed=seed + 5, dim=(18, 8), use_cached=False),
    ]

def lc_invariant_cases(seed: int) -> list[Case]:
    return [
        lcc_eq_case(seed=seed + 1, dim=(2, 0)),
        lcc_eq_case(seed=seed + 2, dim=(3, 0)),
        lcc_eq_case(seed=seed + 3, dim=(3, 1)),
        lcc_eq_case(seed=seed + 4, dim=(4, 2)),
        lcc_eq_case(seed=seed + 10, dim=(5, 1)),
        lcc_eq_case(seed=seed + 11, dim=(5, 2)),
        lcc_eq_case(seed=seed + 16, dim=(6, 5)),
        lcc_eq_case(seed=seed + 21, dim=(7, 6)),
        lcc_eq_case(seed=seed + 22, dim=(8, 7)),
        lcc_eq_case(seed=seed + 23, dim=(7, 1)),
        lcc_eq_case(seed=seed + 24, dim=(9, 1)),
        non_lcc_eq_case(seed=seed + 1, dim=(3, 0)),
        non_lcc_eq_case(seed=seed + 2, dim=(3, 1)),
        non_lcc_eq_case(seed=seed + 3, dim=(4, 1)),
        non_lcc_eq_case(seed=seed + 4, dim=(4, 2)),
        non_lcc_eq_case(seed=seed + 5, dim=(4, 2)),
        non_lcc_eq_case(seed=seed + 10, dim=(5, 1)),
        non_lcc_eq_case(seed=seed + 11, dim=(5, 2)),
        non_lcc_eq_case(seed=seed + 12, dim=(6, 3)),
        non_lcc_eq_case(seed=seed + 18, dim=(7, 3)),
        non_lcc_eq_case(seed=seed + 19, dim=(8, 4)),
        non_lcc_eq_case(seed=seed + 20, dim=(9, 5)),
        non_lcc_eq_case(seed=seed + 25, dim=(10, 5)),
        non_lcc_eq_case(seed=seed + 26, dim=(12, 6)),
    ]


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
        permuted_css_case(seed=seed + 1, dim=(3, 1)),
        permuted_css_case(seed=seed + 1, dim=(4, 2)),
        permuted_css_case(seed=seed + 2, dim=(5, 2)),
        permuted_css_case(seed=seed + 3, dim=(6, 3)),
        permuted_css_case(seed=seed + 7, dim=(7, 2)),
        permuted_css_case(seed=seed + 69, dim=(8, 3)), 
        permuted_css_case(seed=seed + 420, dim=(9, 5)),
        permuted_css_case(seed=seed, dim=(10, 4)),
    ]

    random_permuted_stb = [
        permuted_stabilizer_case(seed=seed + 1, dim=(3, 1)),
        permuted_stabilizer_case(seed=seed + 1, dim=(4, 2)),
        permuted_stabilizer_case(seed=seed + 2, dim=(5, 2)),
        permuted_stabilizer_case(seed=seed + 3, dim=(6, 3)),
        permuted_stabilizer_case(seed=seed + 7, dim=(7, 2)),
        permuted_stabilizer_case(seed=seed + 69, dim=(8, 3)), 
        permuted_stabilizer_case(seed=seed + 420, dim=(9, 5)),
        permuted_stabilizer_case(seed=seed, dim=(10, 4)),
        permuted_stabilizer_case(seed=seed + 12, dim=(10, 4)),
        permuted_stabilizer_case(seed=seed + 55, dim=(10, 4)),
        permuted_stabilizer_case(seed=seed + 4, dim=(13, 4)),
        permuted_stabilizer_case(seed=seed + 6, dim=(13, 4)),
        permuted_stabilizer_case(seed=seed + 9, dim=(15, 4)),


    ]

    random_non_permuted_css = [
        non_permuted_css_case(seed=seed + 3, dim=(3, 1)),
        non_permuted_css_case(seed=seed + 20, dim=(5, 2)),
        non_permuted_css_case(seed=seed + 42, dim=(7, 2)),
        non_permuted_css_case(seed=seed + 1337, dim=(9, 5)),
    ]

    random_non_permuted_stb = [
        non_permuted_stabilizer_case(seed=seed + 1, dim=(3, 1)),
        non_permuted_stabilizer_case(seed=seed + 20, dim=(5, 2)),
        non_permuted_stabilizer_case(seed=seed + 42, dim=(7, 2)),
        non_permuted_stabilizer_case(seed=seed + 1337, dim=(9, 5)),
        non_permuted_stabilizer_case(seed=seed + 69, dim=(10, 4)),
        non_permuted_stabilizer_case(seed=seed + 420, dim=(10, 4)),
        non_permuted_stabilizer_case(seed=seed + 4, dim=(13, 4)),
        non_permuted_stabilizer_case(seed=seed + 6, dim=(13, 4)),
        non_permuted_stabilizer_case(seed=seed + 9, dim=(15, 4))
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
        lcc_css_case(seed=seed + 69, dim=(10, 4)),
    ]

    return random_permuted_stb + random_non_permuted_stb if random else known_permuted + known_lc + known_lc_css


def _run_algorithm_once(
    algorithm: Algorithm,
    inputs: tuple[StabilizerCode, ...],
    timeout: float | None,
) -> tuple[float, bool | None, str]:
    """Run one benchmark repeat, optionally killing it after timeout seconds."""
    if timeout is None:
        start = perf_counter()
        result = algorithm(*inputs)
        return perf_counter() - start, result, ""

    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    queue: mp.Queue = context.Queue()
    process = context.Process(target=_algorithm_worker, args=(algorithm, inputs, queue))

    start = perf_counter()
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        return timeout, None, f"TimeoutError: exceeded {timeout:.6g}s"

    elapsed = perf_counter() - start
    try:
        kind, payload = queue.get_nowait()
    except Empty:
        return elapsed, None, f"RuntimeError: child process exited with code {process.exitcode}"
    finally:
        queue.close()

    if kind == "error":
        return elapsed, None, payload

    return elapsed, payload, ""


def run_case(algorithm_name: str, algorithm: Algorithm, case: Case, repeats: int, timeout: float | None = None) -> Result:
    """Run one algorithm on one case and return the average runtime."""
    total_seconds = 0.0
    last_result: bool | None = None
    errors: list[str] = []
    expected = case.expected_p if algorithm_name.startswith("pm") else (case.expected_lc if algorithm_name.startswith("lc") else None)

    try:
        for _ in range(repeats):
            seconds, result, error = _run_algorithm_once(algorithm, case.inputs, timeout)
            total_seconds += seconds
            if error:
                errors.append(error)
            else:
                last_result = result
        success = not errors and expected is not None and last_result == expected
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=total_seconds / repeats,
            result=last_result,
            expected=expected,
            success=success,
            error="; ".join(errors),
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


def run_raw_benchmarks(
    cases: Sequence[Case],
    algorithm_names: Sequence[str],
    repeats: int,
    timeout: float | None = None,
    verbose: bool = True,
) -> list[Result]:
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
            result_algorithm.append(run_case(algorithm_name, ALGORITHMS[algorithm_name], case, repeats, timeout))

        if verbose:
            print_results(result_algorithm)

        results.extend(result_algorithm)

    return results

def run_stat_benchmarks(
    algorithm_names: Sequence[str],
    repeats: int,
    seed: int,
    output: Path,
    timeout: float | None = None,
    verbose: bool = True,
    random: bool = False,
) -> None:
    """Run selected algorithms on cases with the matching problem type."""
    selected_names = set(algorithm_names)

    for algorithm_name in sorted(selected_names & ALGORITHMS.keys()):
        if verbose:
            print(f"Running benchmark for algorithm: {algorithm_name}")
        stats_algorithm = []
        for measurement, measurement_cases in seeded_measurements(seed=seed, algorithm=algorithm_name, random=random):
            if verbose:
                print(f"    Running measurement for n={measurement.n} k={measurement.k}:")
            results: list[Result] = []
            for case in measurement_cases:
                if not case_supports_algorithm(case, algorithm_name):
                    continue
                if verbose:
                    print(f"        Running case: {case.name}...")
                results.append(run_case(algorithm_name, ALGORITHMS[algorithm_name], case, repeats, timeout))

            stat = compute_statistics(results, measurement)

            if stat is not None:
                stats_algorithm.append(stat)
                write_stat(stat, seed=seed, output=output)

        if verbose:
            print_statistics(stats_algorithm)

def run_inv_benchmarks(
    pm: bool,
    repeats: int,
    seed: int,
    timeout: float | None = None,
    verbose: bool = True,
) -> list[Result]:
    """Run invariants on cases with the matching problem type."""
    if pm:
        INVS = PM_INVARIANTS
        cases = pm_invariant_cases(seed=seed)
    else:        
        INVS = LC_INVARIANTS
        cases =  lc_invariant_cases(seed=seed)
        
    results = []
    for inv_name in sorted(INVS.keys()):
        if verbose:
            print(f"Running benchmark for invariant: {inv_name}")
        result_inv = []
        for case in cases:
            result_inv.append(run_case(inv_name, INVS[inv_name], case, repeats, timeout))

        if verbose:
            print_results(result_inv)
            print()

        results.extend(result_inv)

    return results


def prefixed_output_path(output: Path, prefix: str) -> Path:
    """Return output with a benchmark-family prefix on the filename."""
    if output.name.startswith(f"{prefix}_"):
        return output
    return output.with_name(f"{prefix}_{output.name}")

def result_timed_out(result: Result) -> bool:
    """Return whether a result failed because at least one repeat timed out."""
    return "TimeoutError:" in result.error


def compute_statistics(results: Sequence[Result], measurement: Measurement) -> Statistic | None:
    """Compute mean and standard deviation of runtimes for each algorithm and case."""
    times = []

    for result in results:
        if result.algorithm != measurement.algorithm:
            continue

        if not result.success and not result_timed_out(result):
            print(f"Warning: Skipping failed case {result.case} for algorithm {result.algorithm} in statistics.")
            continue

        times.append(result.seconds)

    if not times:
        return None
    
    mean = np.mean(times)
    stddev = np.std(times, ddof=1) if len(times) > 1 else 0.0

    return Statistic(
        meta=measurement,
        times=times,
        mean=mean,
        stddev=stddev,
        maximum=max(times),
    )
    
def write_stat(stat: Statistic, seed: int, output: Path) -> None:
    """Append one benchmark statistic to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "algorithm": stat.meta.algorithm,
        "name": stat.meta.name,
        "n": stat.meta.n,
        "k": stat.meta.k,
        "positive": stat.meta.positive,
        "density": stat.meta.density,
        "symmetry": stat.meta.symmetry,
        "mean_seconds": f"{stat.mean:.9f}",
        "stddev_seconds": f"{stat.stddev:.9f}",
        "maximum_seconds": f"{stat.maximum:.9f}",
    }
    write_header = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8") as file:
        if write_header:
            csv.writer(file).writerow([seed])
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

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

    for stat in [s for s in statistics if s.meta.positive] + [s for s in statistics if not s.meta.positive]:
        print(
            f"{stat.meta.algorithm:26} {'--' if stat.meta.name is None else stat.meta.name:11} n={stat.meta.n:<2} k={stat.meta.k:<2}   {'POS' if stat.meta.positive else 'NEG'} "
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
            f"{status:6} {result.algorithm:26} {result.case:42} "
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
    parser.add_argument("--inv", action="store_true", default=False, help="Benchmark more complex invariants.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed results updates.")
    parser.add_argument("--random", action="store_true", default=False, help="Use randomly generated cases instead of fixed ones.")
    parser.add_argument("--timeout", type=float, default=None, help="Maximum seconds allowed for each repeat before it is stopped.")
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
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")

    if args.stats:
        if args.output.exists():
            args.output.unlink()
            
        run_stat_benchmarks(args.algorithm, args.repeats, args.seed, args.output, args.timeout, args.verbose, args.random)
        return 0
    
    if args.inv:
        result_pm = run_inv_benchmarks(True, args.repeats, args.seed, args.timeout, args.verbose)
        result_lc = run_inv_benchmarks(False, args.repeats, args.seed, args.timeout, args.verbose)
        write_bms(result_pm, args.seed, prefixed_output_path(args.output, "pm"))
        write_bms(result_lc, args.seed, prefixed_output_path(args.output, "lc"))
        return 0
    else:
        results = run_raw_benchmarks(
            default_cases(seed=args.seed, random=args.random),
            args.algorithm,
            args.repeats,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        write_bms(results, args.seed, args.output)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
