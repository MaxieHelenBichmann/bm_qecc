"""Collect the complete random-suite data for one or more algorithms.

Usage::

    python3 -m paper.benchmarks.collect_algorithm [--algorithm SELECTOR ...]

``--algorithm`` is the only CLI option. It accepts an exact algorithm name,
shell wildcard, or regular expression and may be repeated; omitting it runs all
configured paper algorithms. Every selected algorithm appends to its own file,
``paper/data/collected/algorithms/<algorithm>.csv``.

Edit ``ALGORITHM_N_RANGES`` and the constants below to change the inclusive
per-algorithm range, master seed, cases per cell, timeout, memory limit, or
verbosity. The generated files are deliberately complete shared measurements:
the A3, A4, A5, and A6 experiment scripts later select only the rows they need.
Positive cases retain the established random-suite construction. Negative
cases reuse A1's invariant-neutral proposals and problem-specific
certification; generation and certification occur before the timed backend
call. The seed derivation matches A1, so selected backends receive the same
deterministic case family.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from benchmarks.experiments.generators_random import NonPEqCodePairGenerator
from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import BenchmarkCase, run_statistics
from benchmarks.thesis import resolve_names
from benchmarks.thesis.thesis_prototypes import (
    ALGORITHMS,
    RandomCaseGenerator,
    measurement_dimensions,
)
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat

ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 42
NUM_SEEDS = 10
TIMEOUT_SECONDS = 5_400.0
CERTIFICATION_TIMEOUT_SECONDS = 600.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
CSS_SAT_MAX_R = 9
CSS_MATROID_MAX_N = 28
STABILIZER_CLIFFORD_GATE_STEPS = 2
VERBOSE = True
OUTPUT_DIRECTORY = ROOT / "paper" / "data" / "collected" / "algorithms"

# Inclusive ranges, intentionally centralized so server runs can be tuned by
# editing one table without adding more command-line configuration.
ALGORITHM_N_RANGES: dict[str, tuple[int, int]] = {
    "pm_stb_bruteforce": (3, 47),
    "pm_stb_classical": (3, 47),
    "pm_stb_graph_iso": (3, 47),
    "pm_stb_sat": (3, 47),
    "pm_css_bruteforce": (3, 47),
    "pm_css_classical": (3, 47),
    "pm_css_graph_iso": (3, 47),
    "pm_css_matroid": (3, 47),
    "pm_css_sat": (3, 47),
    "lc_stb_lse": (3, 47),
    "lc_stb_bruteforce": (3, 47),
    "lc_stb_graph_iso": (3, 47),
    "lc_stb_kls": (3, 47),
    "lc_stb_sat": (3, 47),
}
PAPER_ALGORITHMS = {name: ALGORITHMS[name] for name in ALGORITHM_N_RANGES}
CERTIFIERS: dict[str, Callable[..., bool]] = {
    "pm_stb": are_peq_stab_sat,
    "lc_stb": are_lceq_sat,
}


def _problem_for_algorithm(algorithm_name: str) -> str:
    for problem in ("pm_stb", "pm_css", "lc_stb"):
        if algorithm_name.startswith(f"{problem}_"):
            return problem
    raise ValueError(f"unknown paper algorithm family: {algorithm_name!r}")


def _attempt_seed(problem: str, n: int, k: int, seed: int, attempt: int) -> int:
    population = f"{problem}_negative_matching=False"
    value = f"{population}|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _candidate_pair(problem: str, n: int, k: int, seed: int) -> tuple[Any, Any]:
    if problem == "pm_css":
        rx = seed % (n - k + 1)
        return NonPEqCodePairGenerator.css_codes_independent_candidate(
            n, k, seed, rx=rx
        )
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


def _certified_inequivalent(
    problem: str,
    pair: tuple[Any, Any],
    n: int,
    k: int,
) -> bool:
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
) -> tuple[Any, Any]:
    use_css_fallback = problem == "pm_css" and _css_certifier(n, k) is None
    for attempt in range(max_attempts):
        attempt_seed = _attempt_seed(problem, n, k, seed, attempt)
        pair = (
            NonPEqCodePairGenerator.css_codes_cascaded(n, k, attempt_seed)
            if use_css_fallback
            else _candidate_pair(problem, n, k, attempt_seed)
        )
        if use_css_fallback or _certified_inequivalent(problem, pair, n, k):
            return pair
    raise RuntimeError(
        f"could not generate a certified {problem} negative for [[{n},{k}]], "
        f"seed {seed}"
    )


@dataclass(frozen=True)
class CertifiedRandomCaseGenerator:
    """Use invariant-neutral, exactly certified negatives for paper runtimes."""

    algorithm_name: str
    n: int
    k: int
    positive: bool

    seed_upper_bound = 1_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "certified_negative"
        return f"random_{self.algorithm_name}_{self.n}_{self.k}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "generator": self.__name__,
            "name": None,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
            "density": None,
            "symmetry": None,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        if self.positive:
            case = RandomCaseGenerator(
                self.algorithm_name, self.n, self.k, True
            )(seed)
            inputs = case.inputs
        else:
            problem = _problem_for_algorithm(self.algorithm_name)
            inputs = certified_negative_pair(problem, self.n, self.k, seed)
        return BenchmarkCase(tuple(inputs), self.positive, self.metadata)


def validate_configuration() -> None:
    missing = set(ALGORITHM_N_RANGES) - set(ALGORITHMS)
    if missing:
        raise ValueError(f"unknown algorithms in ALGORITHM_N_RANGES: {sorted(missing)}")
    for name, (nmin, nmax) in ALGORITHM_N_RANGES.items():
        if nmin < 1 or nmax < nmin:
            raise ValueError(f"invalid n range for {name}: {(nmin, nmax)}")
    if NUM_SEEDS <= 0 or TIMEOUT_SECONDS <= 0 or MEMORY_LIMIT_BYTES <= 0:
        raise ValueError("seed count and resource limits must be positive")


def collect(algorithm_names: Sequence[str]) -> None:
    """Append complete positive/negative grid statistics for each algorithm."""
    validate_configuration()
    for algorithm_name in algorithm_names:
        nmin, nmax = ALGORITHM_N_RANGES[algorithm_name]
        output_file = OUTPUT_DIRECTORY / f"{algorithm_name}.csv"
        algorithm = ALGORITHMS[algorithm_name]
        if VERBOSE:
            print(
                f"{algorithm_name}: n={nmin}..{nmax}, {NUM_SEEDS} seeds/cell "
                f"-> {output_file}",
                flush=True,
            )
        for n, k in measurement_dimensions(nmin, nmax):
            if algorithm_name == "lc_stb_lse" and k >= 2:
                continue
            for positive in (True, False):
                if VERBOSE:
                    label = "positive" if positive else "negative"
                    print(f"    [[{n},{k}]] {label}", flush=True)
                run_statistics(
                    algorithm,
                    CertifiedRandomCaseGenerator(algorithm_name, n, k, positive),
                    MASTER_SEED,
                    NUM_SEEDS,
                    output_file,
                    timeout=TIMEOUT_SECONDS,
                    max_memory_bytes=MEMORY_LIMIT_BYTES,
                    verbose=VERBOSE,
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        action="append",
        metavar="SELECTOR",
        help=(
            "Exact algorithm name, shell wildcard, or regex; repeatable. "
            "Omit to run every configured algorithm."
        ),
    )
    args = parser.parse_args(argv)
    if args.algorithm is None:
        args.algorithm = list(ALGORITHM_N_RANGES)
        return args
    try:
        args.algorithm = resolve_names(args.algorithm, PAPER_ALGORITHMS)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    collect(args.algorithm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
