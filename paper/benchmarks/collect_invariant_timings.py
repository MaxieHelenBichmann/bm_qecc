"""Collect explicit runtimes for every invariant used in A3.

Usage::

    python3 -m paper.benchmarks.collect_invariant_timings

There are no CLI arguments. Edit ``INVARIANT_N_RANGES`` and the constants below
to change per-invariant ranges, master seed, cases per cell, timeout, memory
limit, or verbosity. Each invariant is timed on positive and negative random
instances and appended to ``paper/data/collected/invariant_timings.csv`` using
the standard statistics schema. The A3 experiment later chooses the fastest
valid backend separately at every parameter setting and computes the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.experiments.statistics import BenchmarkCase, run_statistics
from benchmarks.thesis.thesis_prototypes import RandomCaseGenerator, measurement_dimensions
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids import lc_stb, p_css, p_stab

ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 42
NUM_SEEDS = 5
TIMEOUT_SECONDS = 5_400.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
VERBOSE = True
OUTPUT_FILE = ROOT / "paper" / "data" / "collected" / "invariant_timings.csv"

INVARIANTS = {
    "pm_stb": ("linear_dependency", "signatures"),
    "pm_css": ("linear_dependency", "signatures"),
    "lc_stb": ("local_invariant",),
}


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


INVARIANT_N_RANGES: dict[tuple[str, str], tuple[int, int]] = {
    (problem, invariant): (3, 47)
    for problem, invariants in INVARIANTS.items()
    for invariant in invariants
}


@dataclass(frozen=True)
class InvariantAlgorithm:
    problem: str
    invariant: str

    @property
    def __name__(self) -> str:
        return f"{self.problem}_{self.invariant}"

    def __call__(self, left: Any, right: Any) -> bool:
        evaluate_invariant(self.invariant, self.problem, left, right)
        # A3 measures completion, not whether the invariant accepts the pair.
        return True


@dataclass(frozen=True)
class InvariantCaseGenerator:
    problem: str
    invariant: str
    n: int
    k: int
    positive: bool

    seed_upper_bound = 1_000

    @property
    def algorithm_name(self) -> str:
        return f"{self.problem}_{self.invariant}"

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "negative"
        return f"random_{self.algorithm_name}_{self.n}_{self.k}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "generator": self.__name__,
            "name": self.problem,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        # Any prototype name with the right prefix selects the shared random
        # population for that equivalence problem.
        base = RandomCaseGenerator(f"{self.problem}_sat", self.n, self.k, self.positive)
        case = base(seed)
        return BenchmarkCase(case.inputs, True, self.metadata)


def collect() -> None:
    for (problem, invariant), (nmin, nmax) in INVARIANT_N_RANGES.items():
        algorithm = InvariantAlgorithm(problem, invariant)
        for n, k in measurement_dimensions(nmin, nmax):
            for positive in (True, False):
                if VERBOSE:
                    label = "positive" if positive else "negative"
                    print(f"{algorithm.__name__} [[{n},{k}]] {label}", flush=True)
                run_statistics(
                    algorithm,
                    InvariantCaseGenerator(problem, invariant, n, k, positive),
                    MASTER_SEED,
                    NUM_SEEDS,
                    OUTPUT_FILE,
                    timeout=TIMEOUT_SECONDS,
                    max_memory_bytes=MEMORY_LIMIT_BYTES,
                    verbose=VERBOSE,
                )


if __name__ == "__main__":
    collect()
