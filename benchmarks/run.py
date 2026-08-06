"""Command-line benchmark runner."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import multiprocessing as mp
import os
import re
import signal
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from queue import Empty
from time import perf_counter

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only
    resource = None

import numpy as np

from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce
from src.algorithms.lc_css.lc_css_kls import is_lceq_css_kls
from src.algorithms.lc_css.lc_css_cliff_orbit import is_lceq_css_cliff_orbit
from src.algorithms.lc_css.lc_css_lc_orbit import is_lceq_css_lc_orbit
from src.algorithms.lc_css.lc_css_sat import is_lceq_css_sat
from src.algorithms.lc_stb.lc_stb_lse import are_lceq_graph_state
from src.algorithms.lc_stb.lc_stb_bruteforce import are_lceq_bruteforce
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.lc_stb.lc_stb_kls import are_lceq_kls
from src.algorithms.lc_stb.lc_stb_graph_iso import are_lceq_graph_iso
from src.algorithms.p_css.p_css_bruteforce import are_peq_css_bruteforce
from src.algorithms.p_css.p_css_classical import are_peq_css_classical
from src.algorithms.p_css.p_css_graph_iso import are_peq_css_graph_iso
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_bruteforce import are_peq_stab_bruteforce
from src.algorithms.p_stb.p_stab_aut import are_peq_stab_aut
from src.algorithms.p_stb.p_stab_classical import are_peq_stab_classical
from src.algorithms.p_stb.p_stab_graph_iso import are_peq_stab_graph_iso
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat

from src.hybrids.p_css import are_peq_css
from src.hybrids.p_stab import are_peq_stab
from src.hybrids.lc_stb import are_lceq
from src.hybrids.lc_css import is_lceq_css

from src.invariants.lc_invariants import (
    preserved_local_weight_distribution,
    preserved_low_degree_local_invariant
)

from src.invariants.pm_invariants import (
    preserved_weight_enumerator,
    preserved_pauli_weight_enumerator,
    preserved_linear_dependencies,
)


from src.core.stabilizer_code import StabilizerCode
from src.core.css_code import CSSCode

from .utils import (
    RandomizeError,
    lc_equivalent_code,
    non_lc_css_code,
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

MEAS_STATS = list(range(3, 26)) + list(range(26, 31, 2)) + list(range(32, 51, 5))
N_STATS = 10
N_INVARIANT_STATS = 5
PM_INVARIANT_NS = list(range(2, 26)) + [30, 31, 37, 72, 90, 108, 144]
LC_INVARIANT_NS = list(range(2, 13)) + [15, 23, 25, 30, 31, 37]
KNOWN_INVARIANT_KS = {
    15: {1, 3, 7},
    23: {1},
    25: {1},
    30: {8},
    31: {1, 21},
    37: {1},
    72: {12},
    90: {8},
    108: {8},
    144: {12},
}
MAX_TOL_TIMEOUTS = 3
MAX_TOL_MEMORY_ERRORS = 2
MEMORY_POLL_INTERVAL_SECONDS = 0.2

bell_pair = CSSCode(Hx=np.array([[1, 1]], dtype=np.int8), Hz=np.array([[1, 1]], dtype=np.int8))
three_bit_repetition = CSSCode.from_file("data/three_bit_repetition")
steane = CSSCode.from_file("data/steane")
carbon = CSSCode.from_file("data/carbon")
golay = CSSCode.from_file("data/golay")
hamming_15 = CSSCode.from_file("data/hamming_15")
hamming_31 = CSSCode.from_file("data/hamming_31")
rotated_surface_d5 = CSSCode.from_file("data/rotated_surface_d5")
shor = CSSCode.from_file("data/shor")
tetrahedral = CSSCode.from_file("data/tetrahedral")
bring = CSSCode.from_file("data/bring")
coco_488 = CSSCode.from_file("data/coco_488") 
coco_666 = CSSCode.from_file("data/coco_666") 
bb_72 = CSSCode.from_file("data/bb_72")
bb_90 = CSSCode.from_file("data/bb_90")
bb_108 = CSSCode.from_file("data/bb_108")
bb_144 = CSSCode.from_file("data/bb_144")

five_qubit_perfect = StabilizerCode.from_file("data/five_qubit_perfect")
gottesman = StabilizerCode.from_file("data/eight_qubit_gottesman")
fifteen_qubit_optimal = StabilizerCode.from_file("data/fifteen_qubit_optimal")

NAMED_CODES = [
            ("bell", bell_pair), # n = 2 , k = 0
            ("3q_rep", three_bit_repetition), # n = 3 , k = 1
            ("5q_prf", five_qubit_perfect), # n = 5 , k = 1
            ("steane", steane), # n = 7 , k = 1
            ("gottesman", gottesman), # n = 8 , k = 3
            ("shor", shor),  # n = 9 , k = 1
            ("carbon", carbon), # n = 12 , k = 2
            ("tetrahedral", tetrahedral), # n = 15 , k = 1
            ("15q_optimal", fifteen_qubit_optimal), # n = 15 , k = 3
            ("hamming_15", hamming_15), # n = 15 , k = 7
            ("golay", golay),  # n = 23 , k = 1
            ("rot_surf_d5", rotated_surface_d5), # n = 25 , k = 1
            ("bring", bring), # n = 30 , k = 8
            ("coco_488", coco_488), # n = 31 , k = 1
            ("hamming_31", hamming_31), # n = 31 , k = 21
            ("coco_666", coco_666), # n = 37 , k = 1
            ("bb_72", bb_72), # n = 72, k = 12
            ("bb_90", bb_90), # n = 90, k = 8
            ("bb_108", bb_108), # n = 108, k = 8
            ("bb_144", bb_144) # n = 144 , k = 12
            ]

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
    status: str = "ok"

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
    num_cases: int
    num_successful: int
    num_timeouts: int
    num_memory_limited: int


AlgorithmResult = bool | None | list[int] | list[str]
Algorithm = Callable[..., AlgorithmResult]

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
    "lc_stb_lse": are_lceq_graph_state,
    "lc_stb_bruteforce": are_lceq_bruteforce,
    "lc_stb_graph_iso": are_lceq_graph_iso,
    "lc_stb_kls": are_lceq_kls,
    "lc_stb_sat": are_lceq_sat,
    "lc_css_bruteforce": is_lceq_css_bruteforce,
    "lc_css_kls": is_lceq_css_kls,
    "lc_css_cliff_orbit": is_lceq_css_cliff_orbit,
    "lc_css_lc_orbit": is_lceq_css_lc_orbit,
    "lc_css_sat": is_lceq_css_sat,

    "pm_css_hybrid": are_peq_css,
    "pm_stb_hybrid": are_peq_stab,
    "lc_stb_hybrid": are_lceq,
    "lc_css_hybrid": is_lceq_css,
}

HYBRID_ALGORITHMS = (
    "pm_css_hybrid",
    "pm_stb_hybrid",
    "lc_stb_hybrid",
    "lc_css_hybrid",
)

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

def _algorithm_worker(
    algorithm: Algorithm,
    inputs: tuple[StabilizerCode, ...],
    queue: mp.Queue,
    memory_limit_bytes: int | None,
) -> None:
    """Run one benchmark repeat in a child process."""
    if hasattr(os, "setsid"):
        os.setsid()
    if memory_limit_bytes is not None:
        _set_memory_limit(memory_limit_bytes)
    try:
        queue.put(("result", algorithm(*inputs)))
    except MemoryError:
        queue.put(("error", "MemoryError: exceeded memory limit"))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _set_memory_limit(memory_limit_bytes: int) -> None:
    """Limit address-space allocations in the worker process."""
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = memory_limit_bytes if hard == resource.RLIM_INFINITY else min(hard, memory_limit_bytes)
    new_soft = memory_limit_bytes if soft == resource.RLIM_INFINITY else min(soft, memory_limit_bytes)
    if new_hard != resource.RLIM_INFINITY:
        new_soft = min(new_soft, new_hard)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
    except (OSError, ValueError):
        # Some platforms reject lowering RLIMIT_AS below already-mapped memory.
        # The parent-side RSS monitor remains active in that case.
        pass


def _read_process_group_rss_bytes(process_group_id: int) -> int | None:
    """Read the combined resident memory usage of a process group."""
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-g", str(process_group_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    rss_values_kib: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            rss_values_kib.append(int(line.strip()))
        except ValueError:
            continue

    return sum(rss_values_kib) * 1024 if rss_values_kib else None


def _terminate_process_group(process: mp.Process) -> None:
    """Terminate a benchmark worker and any subprocesses it spawned."""
    if process.pid is None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    process.join(5)
    if process.is_alive():
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
        process.join()


def generated_stabilizer_pair(
    n: int,
    k: int,
    suffix: str,
    seed: int | None = None,
) -> tuple[StabilizerCode, StabilizerCode] | None:
    """Load a generated stabilizer pair from data/ if both files exist."""
    base = f"random_stab_{n}_{k}"
    candidate_paths = [
        (
            DATA_DIR / f"{base}1_{suffix}.txt",
            DATA_DIR / f"{base}2_{suffix}.txt",
        )
    ]
    if seed is not None:
        candidate_paths.insert(
            0,
            (
                DATA_DIR / f"random_stab_{suffix}_{n}_{k}_{seed}_1.txt",
                DATA_DIR / f"random_stab_{suffix}_{n}_{k}_{seed}_2.txt",
            ),
        )
    for paths in candidate_paths:
        if all(path.exists() for path in paths):
            code1_path, code2_path = paths
            return StabilizerCode.from_file(code1_path), StabilizerCode.from_file(code2_path)
    return None

def generated_css_pair(n: int, k: int, suffix: str, seed: int | None = None) -> tuple[CSSCode, CSSCode] | None:
    """Load a generated CSS pair from data/ if both files exist."""
    candidate_paths = [
        (
            DATA_DIR / f"random_css_{n}_{k}1_{suffix}.txt",
            DATA_DIR / f"random_css_{n}_{k}2_{suffix}.txt",
        )
    ]
    if seed is not None:
        candidate_paths.insert(
            0,
            (
                DATA_DIR / f"random_css_{suffix}_{n}_{k}_{seed}_1.txt",
                DATA_DIR / f"random_css_{suffix}_{n}_{k}_{seed}_2.txt",
            ),
        )
        candidate_paths.insert(
            1,
            (
                DATA_DIR / f"random_css_{n}_{k}_{seed}1_{suffix}.txt",
                DATA_DIR / f"random_css_{n}_{k}_{seed}2_{suffix}.txt",
            ),
        )

    for paths in candidate_paths:
        if all(path.exists() for path in paths):
            code1_path, code2_path = paths
            return CSSCode.from_file(code1_path), CSSCode.from_file(code2_path)

    return None

def case_supports_algorithm(case: Case, algorithm_name: str) -> bool:
    """Return whether a case has an expectation and compatible inputs for an algorithm."""
    if algorithm_name.startswith("pm_css") and all(isinstance(code, CSSCode) for code in case.inputs)and len(case.inputs) == 2 and case.expected_p is not None:
        return True
    if algorithm_name.startswith("pm_stb") and len(case.inputs) == 2 and case.expected_p is not None:
        return True
    if algorithm_name.startswith("lc_stb") and len(case.inputs) == 2 and case.expected_lc is not None:
        if algorithm_name == "lc_stb_graph_state":
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
            pair = generated_css_pair(n, k, "non_peq", seed=seed)
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
            pair = generated_css_pair(n, k, "peq", seed=seed)
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
            pair = generated_stabilizer_pair(n, k, "non_peq", seed=seed)
            code1, code2 = pair or random_non_permuted_stabilizer_pair(n, k, seed=seed)
        else:
            code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=seed)
    else:
        if code is None:
            raise ValueError("Either dim or code must be provided")
        n, k = code.n, code.k
        pair = generated_stabilizer_pair(n, k, "non_peq", seed=seed) if use_cached else None
        if pair is None:
            code1 = code
            code2 = non_permutation_equivalent_stabilizer_code(code1, seed=seed)
        else:
            code1, code2 = pair

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

    cached_path = DATA_DIR / "lc" / f"non_lcc_css_{n}_{k}_{seed}.txt"
    if cached_path.is_file():
        negative = StabilizerCode.from_file(cached_path)
        if (negative.n, negative.k) != (n, k):
            raise ValueError(
                f"Cached LC-CSS negative {cached_path} has parameters "
                f"[[{negative.n}, {negative.k}]], expected [[{n}, {k}]]."
            )
    else:
        negative = non_lc_css_code(code, seed=seed + 69)

    return Case(
        name=f"non_lcc_css_{n}_{k}_{seed}",
        inputs=(negative,),
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


_LC_CSS_NEGATIVE_EXCLUDED_DIMENSIONS = {(3, 0), (3, 1), (4, 0), (8,6), (8,0), (10,8), (12,10)}
_LC_CSS_NEGATIVE_EXCLUDED_SEEDS = {
    (4, 1): {85},
    (4, 2): {773, 654, 438, 433, 858, 85, 697, 201, 94},
    (7, 4): {201, 94},
    (9, 6): {89, 697}
}


def supports_lc_css_negative_case(n: int, k: int, seed: int) -> bool:
    """Whether the generator can provide a certified LC-CSS negative case."""
    if k == n - 1:
        return False
    if (n, k) in _LC_CSS_NEGATIVE_EXCLUDED_DIMENSIONS:
        return False
    return int(seed) not in _LC_CSS_NEGATIVE_EXCLUDED_SEEDS.get((n, k), set())


def seeded_cases(
    seeds: Sequence[int],
    generator: Callable[[int], Case],
) -> list[Case]:
    """Generate seeded cases, skipping seeds the generator cannot satisfy."""
    cases = []
    for seed in seeds:
        try:
            cases.append(generator(int(seed)))
        except RandomizeError:
            continue
    return cases

def seeded_measurements(seed: int, algorithm: str, random: bool, nmin: int | None = None, nmax: int | None = None) -> list[tuple[Measurement, list[Case]]]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000, size=N_STATS)
    measurements : list[tuple[Measurement, list[Case]]] = []

    sizes = [
        (n, i)
        for n in [n for n in MEAS_STATS if (nmin is None or n >= nmin) and (nmax is None or n <= nmax)]
        for i in sorted(set(range(0, n,  1 if n < 7 else 2 if n < 15 else 4 if n < 30 else 5)) | {4, 8})
        if i < n
    ]

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
                                    seeded_cases(seeds, lambda s: non_permuted_css_case(seed=s, dim=(n, k), use_cached=n>17))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: permuted_css_case(seed=s, dim=(n, k), use_cached=False))))
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
                                    seeded_cases(seeds, lambda s: non_permuted_stabilizer_case(seed=s, dim=(n, k), use_cached=False))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: permuted_stabilizer_case(seed=s, dim=(n, k), use_cached=False))))
            elif algorithm.startswith("lc_stb"):
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: lcc_eq_case(seed=s, dim=(n, k)))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=None, 
                                        n=n, 
                                        k=k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: non_lcc_eq_case(seed=s, dim=(n, k)))))
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
                    seeded_cases(seeds, lambda s: lcc_css_case(seed=s, dim=(n, k)))))
                negative_seeds = [
                    int(s) for s in seeds if supports_lc_css_negative_case(n, k, int(s))
                ]
                if negative_seeds:
                    measurements.append((Measurement(
                                            algorithm=algorithm,
                                            name=None,
                                            n=n,
                                            k=k,
                                            positive=False,
                                            density=None,
                                            symmetry=None
                                        ),
                        seeded_cases(negative_seeds, lambda s: non_lcc_css_case(seed=s, dim=(n, k)))))
    else:
        if algorithm.startswith("pm_css"):
            for name, code in [ (name, code) for name, code in NAMED_CODES if (nmin is None or code.n >= nmin) and (nmax is None or code.n <= nmax) ]:
                if not isinstance(code, CSSCode):
                    continue
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ),
                                    seeded_cases(seeds, lambda s: permuted_css_case(seed=s, code=code))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ),
                                    seeded_cases(seeds, lambda s: non_permuted_css_case(seed=s, code=code))
                                    ))

        elif algorithm.startswith("pm_stb"):
            for name, code in [ (name, code) for name, code in NAMED_CODES if (nmin is None or code.n >= nmin) and (nmax is None or code.n <= nmax) ]:
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: permuted_stabilizer_case(seed=s, code=code))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: non_permuted_stabilizer_case(seed=s, code=code))))
        elif algorithm.startswith("lc_stb"):
            for name, code in [ (name, code) for name, code in NAMED_CODES if (nmin is None or code.n >= nmin) and (nmax is None or code.n <= nmax) ]:
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: lcc_eq_case(seed=s, code=code))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: non_lcc_eq_case(seed=s, code=code))))
        elif algorithm.startswith("lc_css"):
              for name, code in [ (name, code) for name, code in NAMED_CODES if (nmin is None or code.n >= nmin) and (nmax is None or code.n <= nmax) ]:
                if not isinstance(code, CSSCode):
                    continue
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=True, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: lcc_css_case(seed=s, code=code))))
                measurements.append((Measurement(
                                        algorithm=algorithm, 
                                        name=name, 
                                        n=code.n, 
                                        k=code.k, 
                                        positive=False, 
                                        density=None, 
                                        symmetry=None
                                    ), 
                                    seeded_cases(seeds, lambda s: non_lcc_css_case(seed=s, code=code))))

    return measurements
    
def invariant_dimensions(pm: bool) -> list[tuple[int, int]]:
    """Return a regular, bounded (n, k) grid for invariant benchmarks."""
    ns = PM_INVARIANT_NS if pm else LC_INVARIANT_NS
    dimensions = []

    for n in ns:
        if n > 50:
            # At these sizes, use only dimensions backed by an existing named
            # code; eagerly generating arbitrary random pairs is too costly.
            ks = KNOWN_INVARIANT_KS.get(n, set())
        elif n < 7:
            ks = range(0, n)
        elif n < 15:
            ks = set(range(0, n, 2)) | {n // 2}
        elif n < 30:
            ks = set(range(0, n, 4)) | {n // 2}
        else:
            ks = {0, 1, n // 4, n // 2, 3 * n // 4, n - 1}

        ks = set(ks) | KNOWN_INVARIANT_KS.get(n, set())
        dimensions.extend((n, k) for k in sorted(ks) if k < n)

    return dimensions


def invariant_measurements(seed: int, invariant: str, pm: bool) -> Iterator[tuple[Measurement, list[Case]]]:
    """Yield five reproducible positive/negative cases per invariant dimension."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, size=N_INVARIANT_STATS)
    for n, k in invariant_dimensions(pm):
        named_code = next(
            (code for _, code in NAMED_CODES if n > 50 and (code.n, code.k) == (n, k)),
            None,
        )
        for positive in (True, False):
            measurement = Measurement(invariant, None, n, k, positive, None, None)
            if pm:
                if positive:
                    cases = [
                        permuted_stabilizer_case(
                            seed=int(s), code=named_code
                        ) if named_code is not None else permuted_stabilizer_case(
                            seed=int(s), dim=(n, k)
                        )
                        for s in seeds
                    ]
                else:
                    cases = [
                        replace(
                            non_permuted_stabilizer_case(seed=int(s), code=named_code)
                            if named_code is not None else non_permuted_stabilizer_case(
                                seed=int(s), dim=(n, k), use_cached=False
                            ),
                            expected_p=None,
                        )
                        for s in seeds
                    ]
            else:
                if positive:
                    cases = [lcc_eq_case(seed=int(s), dim=(n, k)) for s in seeds]
                else:
                    cases = [
                        replace(non_lcc_eq_case(seed=int(s), dim=(n, k)), expected_lc=None)
                        for s in seeds
                    ]
            yield measurement, cases

def default_cases(seed: int, random: bool = False, nmin: int | None = None, nmax: int | None = None) -> list[Case]:
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

    if random:
        default_cases = random_permuted_css + random_non_permuted_css +random_permuted_stb + random_non_permuted_stb + random_lc_css
    else:
        default_cases = known_permuted + known_lc + known_lc_css

    return [case for case in default_cases if (nmin is None or case.inputs[0].n >= nmin) and (nmax is None or case.inputs[0].n <= nmax)]


def _run_algorithm_once(
    algorithm: Algorithm,
    inputs: tuple[StabilizerCode, ...],
    timeout: float | None,
    memory_limit_bytes: int | None,
) -> tuple[float, AlgorithmResult, str]:
    """Run one benchmark repeat, optionally killing it after timeout or memory pressure."""
    if timeout is None and memory_limit_bytes is None:
        start = perf_counter()
        result = algorithm(*inputs)
        return perf_counter() - start, result, "ok"

    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    queue: mp.Queue = context.Queue()
    process = context.Process(target=_algorithm_worker, args=(algorithm, inputs, queue, memory_limit_bytes))

    start = perf_counter()
    deadline = None if timeout is None else start + timeout
    process.start()
    while process.is_alive():
        now = perf_counter()
        if deadline is not None and now >= deadline:
            _terminate_process_group(process)
            queue.close()
            return timeout, None, "timeout"

        # The worker starts a new process group, so this includes subprocesses
        # such as GAP in pm_stb_aut as well as the Python worker itself.
        rss_bytes = _read_process_group_rss_bytes(process.pid) if process.pid is not None else None
        if rss_bytes is not None:
            if memory_limit_bytes is not None and rss_bytes >= memory_limit_bytes:
                elapsed = perf_counter() - start
                _terminate_process_group(process)
                queue.close()
                return (
                    elapsed,
                    None,
                    "memory_limit",
                )

        sleep_for = MEMORY_POLL_INTERVAL_SECONDS
        if deadline is not None:
            sleep_for = min(sleep_for, max(0.0, deadline - now))
        process.join(sleep_for)

    if process.is_alive():
        _terminate_process_group(process)
        queue.close()
        return timeout or perf_counter() - start, None, "timeout"

    elapsed = perf_counter() - start
    try:
        kind, payload = queue.get_nowait()
    except Empty:
        return elapsed, None, "error"
    finally:
        queue.close()

    if kind == "error":
        status = "memory_limit" if payload.startswith("MemoryError:") else "error"
        return elapsed, None, status

    return elapsed, payload, "ok"


def decision_result(result: AlgorithmResult) -> bool:
    """Convert a Boolean or witness result into a Boolean decision."""
    if isinstance(result, bool):
        return result
    return result is not None


def run_case(
    algorithm_name: str,
    algorithm: Algorithm,
    case: Case,
    timeout: float | None = None,
    memory_limit_bytes: int | None = None,
) -> Result:
    """Run one algorithm on one case."""
    expected = case.expected_p if algorithm_name.startswith("pm") else (case.expected_lc if algorithm_name.startswith("lc") else None)

    try:
        seconds, raw_result, status = _run_algorithm_once(
            algorithm,
            case.inputs,
            timeout,
            memory_limit_bytes,
        )
        result = decision_result(raw_result) if status == "ok" else None
        if status == "ok" and expected is not None and result != expected:
            status = "wrong"
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=seconds,
            result=result,
            expected=expected,
            status=status,
        )
    except Exception:  # noqa: BLE001
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=0.0,
            result=None,
            expected=expected,
            status="error",
        )


def run_raw_benchmarks(
    cases: Sequence[Case],
    algorithm_names: Sequence[str],
    timeout: float | None = None,
    memory_limit_bytes: int | None = None,
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
            result_algorithm.append(
                run_case(algorithm_name, ALGORITHMS[algorithm_name], case, timeout, memory_limit_bytes)
            )

        if verbose:
            print_results(result_algorithm)

        results.extend(result_algorithm)

    return results

def run_stat_benchmarks(
    algorithm_names: Sequence[str],
    seed: int,
    output: Path,
    timeout: float | None = None,
    memory_limit_bytes: int | None = None,
    verbose: bool = True,
    random: bool = False,
    nmin: int | None = None,
    nmax: int | None = None,
) -> None:
    """Run selected algorithms on cases with the matching problem type."""
    selected_names = set(algorithm_names)

    for algorithm_name in sorted(selected_names & ALGORITHMS.keys()):
        if verbose:
            print(f"Running benchmark for algorithm: {algorithm_name}")
        stats_algorithm = []
        for measurement, measurement_cases in seeded_measurements(seed=seed, algorithm=algorithm_name, random=random, nmin=nmin, nmax=nmax):
            if verbose:
                print(f"    Running measurement for n={measurement.n} k={measurement.k}:")
            results: list[Result] = []
            timeout_counter = 0 # avoid running 10 seeds that run into timeouts either way
            memory_counter = 0 # avoid running seeds that are expected to hit the same memory limit

            for case in measurement_cases:
                if not case_supports_algorithm(case, algorithm_name):
                    continue
                if timeout_counter >= MAX_TOL_TIMEOUTS or memory_counter >= MAX_TOL_MEMORY_ERRORS:
                    break
                if verbose:
                    print(f"        Running case: {case.name}...")

                results.append(
                    run_case(algorithm_name, ALGORITHMS[algorithm_name], case, timeout, memory_limit_bytes)
                )

                if result_timed_out(results[-1]):
                    timeout_counter += 1
                if result_memory_limited(results[-1]):
                    memory_counter += 1

            stat = compute_statistics(results, measurement)

            stats_algorithm.append(stat)
            write_stat(stat, seed=seed, output=output, timeout=timeout, memory_limit_bytes=memory_limit_bytes)

        if verbose:
            print_statistics(stats_algorithm)

def run_inv_benchmarks(
    pm: bool,
    seed: int,
    output: Path,
    timeout: float | None = None,
    memory_limit_bytes: int | None = None,
    verbose: bool = True,
) -> None:
    """Run seeded invariant measurements and write aggregate statistics."""
    invariants = PM_INVARIANTS if pm else LC_INVARIANTS
    for inv_name in sorted(invariants.keys()):
        if verbose:
            print(f"Running benchmark for invariant: {inv_name}")
        statistics = []
        for measurement, cases in invariant_measurements(seed, inv_name, pm):
            results = []
            timeout_counter = 0
            memory_counter = 0
            for case in cases:
                if timeout_counter >= MAX_TOL_TIMEOUTS or memory_counter >= MAX_TOL_MEMORY_ERRORS:
                    break
                results.append(run_case(inv_name, invariants[inv_name], case, timeout, memory_limit_bytes))
                timeout_counter += int(result_timed_out(results[-1]))
                memory_counter += int(result_memory_limited(results[-1]))
            statistic = compute_statistics(results, measurement)
            statistics.append(statistic)
            write_stat(statistic, seed, output, timeout, memory_limit_bytes)
        if verbose:
            print_statistics(statistics)


def prefixed_output_path(output: Path, prefix: str) -> Path:
    """Return output with a benchmark-family prefix on the filename."""
    if output.name.startswith(f"{prefix}_"):
        return output
    return output.with_name(f"{prefix}_{output.name}")

def result_timed_out(result: Result) -> bool:
    """Return whether a result failed because at least one repeat timed out."""
    return result.status == "timeout"


def result_memory_limited(result: Result) -> bool:
    """Return whether a result failed because it hit the memory guard."""
    return result.status == "memory_limit"


def compute_statistics(results: Sequence[Result], measurement: Measurement) -> Statistic:
    """Compute mean and standard deviation of runtimes for each algorithm and case."""
    times = []
    num_timeouts = 0
    num_memory_limited = 0

    for result in results:
        if result.algorithm != measurement.algorithm:
            continue

        if result_timed_out(result):
            num_timeouts += 1
        elif result_memory_limited(result):
            num_memory_limited += 1

        if result.status not in {"ok", "timeout"}:
            print(f"Warning: Skipping failed case {result.case} for algorithm {result.algorithm} in statistics.")
            continue

        times.append(result.seconds)
    
    mean = float(np.mean(times)) if times else float("nan")
    stddev = np.std(times, ddof=1) if len(times) > 1 else 0.0

    return Statistic(
        meta=measurement,
        times=times,
        mean=mean,
        stddev=stddev,
        maximum=max(times) if times else float("nan"),
        num_cases=len(results),
        num_successful=sum(1 for result in results if result.status == "ok"),
        num_timeouts=num_timeouts,
        num_memory_limited=num_memory_limited,
    )
    
def _metadata_row(seed: int, timeout: float | None, memory_limit_bytes: int | None) -> list[str | int | float]:
    return [
        seed,
        "" if timeout is None else timeout,
        "" if memory_limit_bytes is None else memory_limit_bytes,
    ]


def write_stat(
    stat: Statistic,
    seed: int,
    output: Path,
    timeout: float | None,
    memory_limit_bytes: int | None,
) -> None:
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
        "mean_seconds": "" if np.isnan(stat.mean) else f"{stat.mean:.9f}",
        "stddev_seconds": "" if np.isnan(stat.stddev) else f"{stat.stddev:.9f}",
        "maximum_seconds": "" if np.isnan(stat.maximum) else f"{stat.maximum:.9f}",
        "num_cases": stat.num_cases,
        "num_successful": stat.num_successful,
        "num_timeouts": stat.num_timeouts,
        "num_memory_limited": stat.num_memory_limited,
    }
    write_header = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8") as file:
        if write_header:
            csv.writer(file).writerow(_metadata_row(seed, timeout, memory_limit_bytes))
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def write_bms(
    results: Sequence[Result],
    seed: int,
    output: Path,
    timeout: float | None,
    memory_limit_bytes: int | None,
) -> None:
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
            "status": result.status,
        }
        for result in results
    ]

    if len(rows) == 0:
        return

    with output.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(_metadata_row(seed, timeout, memory_limit_bytes))
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_statistics(statistics: Sequence[Statistic]) -> None:
    """Print benchmark statistics to console."""
    if len(statistics) == 0:
        print("No statistics to show.")
        return
    
    print()
    print("Benchmark statistics:\n")

    for stat in [s for s in statistics if s.meta.positive] + [s for s in statistics if not s.meta.positive]:
        mean = "--" if np.isnan(stat.mean) else f"{stat.mean:.6f}s"
        stddev = "--" if np.isnan(stat.stddev) else f"{stat.stddev:.6f}s"
        maximum = "--" if np.isnan(stat.maximum) else f"{stat.maximum:.6f}s"
        print(
            f"{stat.meta.algorithm:26} {'--' if stat.meta.name is None else stat.meta.name:11} n={stat.meta.n:<2} k={stat.meta.k:<2}   {'POS' if stat.meta.positive else 'NEG'} "
            f"mean={mean} stddev={stddev} max={maximum} "
            f"timeouts={stat.num_timeouts} memory={stat.num_memory_limited} ntotal={stat.num_cases} nsuccess={stat.num_successful}"
        )

def print_results(results: Sequence[Result]) -> None:
    """Print benchmark results to console."""
    if len(results) == 0:
        print("No cases ran, no results to show.")
        return

    print("Benchmark results:\n")

    for result in results:
        print(
            f"{result.status:12} {result.algorithm:26} {result.case:42} "
            f"n={result.n:<2} k={result.k:<2} t={result.seconds:.6f}s status={result.status}"
        )


def parse_memory_limit(value: str) -> int:
    """Parse human-readable memory sizes like 4096M, 32G, or raw bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|b)?\s*", value, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError("expected a size like 4096M, 32G, 16GiB, or raw bytes")

    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    bytes_value = int(number * multipliers[unit])
    if bytes_value <= 0:
        raise argparse.ArgumentTypeError("--memory-limit must be greater than 0")
    return bytes_value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Available algorithms: " + ", ".join(all_algorithm_names()),
    )
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
    parser.add_argument("--nmin", type=int, help="Minimum value for n.")
    parser.add_argument("--nmax", type=int, help="Maximum value for n.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed results updates.")
    parser.add_argument("--random", action="store_true", default=False, help="Use randomly generated cases instead of fixed ones.")
    parser.add_argument("--timeout", type=float, default=None, help="Maximum seconds allowed for each repeat before it is stopped.")
    parser.add_argument(
        "--memory-limit",
        type=parse_memory_limit,
        default=None,
        help=(
            "Maximum memory for each benchmark child, e.g. 4096M, 32G, or 16GiB. "
            "The child receives an address-space limit and is killed if observed RSS reaches the limit."
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--raw",
        action="store_const",
        const="raw",
        dest="mode",
        help="Run raw benchmarks on default known or random cases. This is the default mode.",
    )
    modes.add_argument(
        "--stats",
        action="store_const",
        const="stats",
        dest="mode",
        help="Run statistical benchmarks on seeded fixed or random cases.",
    )
    modes.add_argument(
        "--structured-hybrid-stats",
        "--hybrid-stats",
        action="store_const",
        const="structured_hybrid_stats",
        dest="mode",
        help=(
            "Run statistical benchmarks on the structured/named codes, using only hybrid "
            "algorithms. Select a hybrid with --algorithm; all hybrids are used by default."
        ),
    )
    modes.add_argument(
        "--inv",
        action="store_const",
        const="inv",
        dest="mode",
        help="Run invariant benchmarks on fixed cases.",
    )
    parser.set_defaults(mode="raw")
    args = parser.parse_args(argv)
    algorithm_selectors = args.algorithm
    if args.mode == "inv" and args.algorithm:
        parser.error("--algorithm can only be used with raw or stats benchmarks.")
    if args.mode in {"inv", "structured_hybrid_stats"} and args.random:
        parser.error("--random cannot be used with invariant or structured hybrid benchmarks.")
    if args.mode == "inv" and (args.nmin is not None or args.nmax is not None):
        parser.error("--nmin and --nmax can only be used with raw or stats benchmarks.")
    try:
        args.algorithm = (
            list(HYBRID_ALGORITHMS)
            if args.mode == "structured_hybrid_stats" and algorithm_selectors is None
            else resolve_algorithm_names(algorithm_selectors)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "structured_hybrid_stats":
        non_hybrids = sorted(set(args.algorithm) - set(HYBRID_ALGORITHMS))
        if non_hybrids:
            parser.error(
                "--structured-hybrid-stats only accepts hybrid algorithms: "
                + ", ".join(HYBRID_ALGORITHMS)
            )
    if args.nmin is not None and args.nmin < 1:
        parser.error("--nmin must be at least 1.")
    if args.nmax is not None and args.nmax < 1:
        parser.error("--nmax must be at least 1.")
    if args.nmin is not None and args.nmax is not None and args.nmin > args.nmax:
        parser.error("--nmin cannot be greater than --nmax.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")

    if args.mode in {"stats", "structured_hybrid_stats"}:
        s = (
            "structured"
            if args.mode == "structured_hybrid_stats" or not args.random
            else "random"
        )
        print(f"STATISTICAL BENCHMARKS for {s} cases")
        if args.output.exists():
            args.output.unlink()
            
        run_stat_benchmarks(
            args.algorithm,
            args.seed,
            args.output,
            args.timeout,
            args.memory_limit,
            args.verbose,
            False if args.mode == "structured_hybrid_stats" else args.random,
            args.nmin,
            args.nmax,
        )
        return 0
    
    if args.mode == "inv":
        print("INVARIANT BENCHMARKS")
            
        pm_output = prefixed_output_path(args.output, "pm")
        lc_output = prefixed_output_path(args.output, "lc")
        for output in (pm_output, lc_output):
            if output.exists():
                output.unlink()
        run_inv_benchmarks(True, args.seed, pm_output, args.timeout, args.memory_limit, args.verbose)
        run_inv_benchmarks(False, args.seed, lc_output, args.timeout, args.memory_limit, args.verbose)
        return 0
 
    raw_case_kind = "random" if args.random else "structured"
    print(f"RAW BENCHMARKS for {raw_case_kind} cases")
    results = run_raw_benchmarks(
        default_cases(seed=args.seed, random=args.random, nmin=args.nmin, nmax=args.nmax),
        args.algorithm,
        timeout=args.timeout,
        memory_limit_bytes=args.memory_limit,
        verbose=args.verbose,
    )
    write_bms(results, args.seed, args.output, args.timeout, args.memory_limit)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
