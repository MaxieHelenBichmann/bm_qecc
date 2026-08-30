"""Run prototype algorithms on randomly generated benchmark cases.

Usage::

    python3 -m benchmarks.thesis.thesis_prototypes [OPTIONS]

CLI options:

``--algorithm SELECTOR``
    Select an algorithm by exact name, shell wildcard, or regular expression.
    The option is repeatable. All prototype algorithms run when it is omitted.
``--nmin N`` / ``--nmax N``
    Restrict the inclusive random-code block-length range. By default the full
    thesis parameter grid is used.
``--seed SEED``
    Master seed from which per-case seeds are derived (default: 42).
``--nr-seeds N`` / ``--num-seeds N``
    Number of generated cases per dimension and label (default: 10).
``--output PATH``
    CSV file to append to (default: ``results/prototypes.csv``). A header is
    written automatically when the file does not exist or is empty.
``--timeout SECONDS``
    Optional wall-clock limit for each algorithm call.
``--memory-limit SIZE``
    Optional per-call memory limit. Values such as ``512M``, ``4GiB``, or a
    raw byte count are accepted.
``--verbose``
    Print progress for algorithms, dimensions, labels, and individual seeds.

Each selected ``(n, k)`` cell is benchmarked with positive and negative random
instances where that combination is supported. Raw and structured cases do
not belong to this entry point.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce
from src.algorithms.lc_css.lc_css_cliff_orbit import is_lceq_css_cliff_orbit
from src.algorithms.lc_css.lc_css_kls import is_lceq_css_kls
from src.algorithms.lc_css.lc_css_lc_orbit import is_lceq_css_lc_orbit
from src.algorithms.lc_css.lc_css_sat import is_lceq_css_sat
from src.algorithms.lc_stb.lc_stb_bruteforce import are_lceq_bruteforce
from src.algorithms.lc_stb.lc_stb_graph_iso import are_lceq_graph_iso
from src.algorithms.lc_stb.lc_stb_kls import are_lceq_kls
from src.algorithms.lc_stb.lc_stb_lse import are_lceq_graph_state
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.p_css.p_css_bruteforce import are_peq_css_bruteforce
from src.algorithms.p_css.p_css_classical import are_peq_css_classical
from src.algorithms.p_css.p_css_graph_iso import are_peq_css_graph_iso
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_aut import are_peq_stab_aut
from src.algorithms.p_stb.p_stab_bruteforce import are_peq_stab_bruteforce
from src.algorithms.p_stb.p_stab_classical import are_peq_stab_classical
from src.algorithms.p_stb.p_stab_graph_iso import are_peq_stab_graph_iso
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.stabilizer_code import StabilizerCode

from . import parse_memory_limit, resolve_names, validate_common_args
from ..experiments.generators_random import (
    LCEqCodeGenerator,
    LCEqCodePairGenerator,
    NonLCEqCodeGenerator,
    NonLCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from ..experiments.statistics import BenchmarkCase, Statistic, run_statistics


MEAS_STATS = list(range(3, 26)) + list(range(26, 31, 2)) + list(range(32, 51, 5))
N_STATS = 10
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class DecisionAlgorithm:
    """Adapt Boolean-or-witness implementations to one Boolean decision API."""

    name: str
    function: Callable[..., Any]

    @property
    def __name__(self) -> str:
        return self.name

    def __call__(self, *inputs: Any) -> bool:
        result = self.function(*inputs)
        return result if isinstance(result, bool) else result is not None


_ALGORITHM_FUNCTIONS: dict[str, Callable[..., Any]] = {
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
}
ALGORITHMS: dict[str, DecisionAlgorithm] = {
    name: DecisionAlgorithm(name, function)
    for name, function in _ALGORITHM_FUNCTIONS.items()
}


_LC_CSS_NEGATIVE_EXCLUDED_DIMENSIONS = {
    (3, 0),
    (3, 1),
    (4, 0),
    (8, 0),
    (8, 6),
    (10, 8),
    (12, 10),
}
_LC_CSS_NEGATIVE_EXCLUDED_SEEDS = {
    (4, 1): {85},
    (4, 2): {773, 654, 438, 433, 858, 85, 697, 201, 94},
    (7, 4): {201, 94},
    (9, 6): {89, 697},
}


def supports_lc_css_negative_case(n: int, k: int, seed: int) -> bool:
    """Whether the random generator can certify this negative LC-CSS case."""
    if k == n - 1 or (n, k) in _LC_CSS_NEGATIVE_EXCLUDED_DIMENSIONS:
        return False
    return seed not in _LC_CSS_NEGATIVE_EXCLUDED_SEEDS.get((n, k), set())


def generated_lc_css_code(
    n: int, k: int, *, positive: bool, seed: int
) -> StabilizerCode | None:
    """Load a cached random LC-CSS case when one was precomputed."""
    prefix = "lcc_css" if positive else "non_lcc_css"
    path = DATA_DIR / "lc" / f"{prefix}_{n}_{k}_{seed}.txt"
    if not path.is_file():
        return None
    code = StabilizerCode.from_file(path)
    if (code.n, code.k) != (n, k):
        raise ValueError(
            f"Cached case {path} is [[{code.n},{code.k}]], expected [[{n},{k}]]."
        )
    return code


def measurement_dimensions(
    nmin: int | None = None, nmax: int | None = None
) -> list[tuple[int, int]]:
    """Return the thesis prototype's randomized ``(n, k)`` grid."""
    return [
        (n, k)
        for n in MEAS_STATS
        if (nmin is None or n >= nmin) and (nmax is None or n <= nmax)
        for k in sorted(
            set(range(0, n, 1 if n < 7 else 2 if n < 15 else 4 if n < 30 else 5))
            | {4, 8}
        )
        if k < n
    ]


@dataclass(frozen=True)
class RandomCaseGenerator:
    """Generate one random labeled case for a problem family and dimension."""

    algorithm_name: str
    n: int
    k: int
    positive: bool

    @property
    def seed_upper_bound(self) -> int:
        # Retain the established random-suite seed stream and cache names.
        return 1_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "negative"
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
        n, k = self.n, self.k
        inputs: tuple[Any, ...]
        if self.algorithm_name.startswith("pm_css"):
            inputs = (
                PEqCodePairGenerator.css_codes_basis_changed(n, k, seed)
                if self.positive
                else NonPEqCodePairGenerator.css_codes_cascaded(n, k, seed)
            )
        elif self.algorithm_name.startswith("pm_stb"):
            inputs = (
                PEqCodePairGenerator.stabilizer_codes_basis_changed(n, k, seed)
                if self.positive
                else NonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection(
                    n, k, seed
                )
            )
        elif self.algorithm_name.startswith("lc_stb"):
            inputs = (
                LCEqCodePairGenerator.stabilizer_codes_local_clifford(n, k, seed)
                if self.positive
                else NonLCEqCodePairGenerator.stabilizer_codes_independent(n, k, seed)
            )
        elif self.algorithm_name.startswith("lc_css"):
            if not self.positive and not supports_lc_css_negative_case(n, k, seed):
                raise ValueError("unsupported certified negative LC-CSS case")
            code = generated_lc_css_code(n, k, positive=self.positive, seed=seed)
            if code is None:
                code = (
                    LCEqCodeGenerator.stabilizer_code_local_clifford(n, k, seed)
                    if self.positive
                    else NonLCEqCodeGenerator.stabilizer_code_locally_rank_one(
                        n, k, seed
                    )
                )
            inputs = (code,)
        else:  # pragma: no cover - guarded by the registry
            raise ValueError(f"Unknown algorithm family: {self.algorithm_name}")
        return BenchmarkCase(inputs=tuple(inputs), expected=self.positive, metadata=self.metadata)


def _supports_dimension(algorithm_name: str, n: int, k: int) -> bool:
    if algorithm_name == "lc_stb_lse":
        return k < 2
    if algorithm_name == "lc_css_lc_orbit":
        return k < 2
    return True


def run_suite(
    algorithm_names: Sequence[str],
    *,
    seed: int,
    nr_seeds: int,
    output_file: Path,
    nmin: int | None = None,
    nmax: int | None = None,
    timeout: float | None = None,
    max_memory_bytes: int | None = None,
    verbose: bool = False,
) -> list[Statistic]:
    """Run the selected prototypes over the requested random parameter grid."""
    statistics: list[Statistic] = []
    for algorithm_name in algorithm_names:
        algorithm = ALGORITHMS[algorithm_name]
        if verbose:
            print(f"Running prototype: {algorithm_name}")
        for n, k in measurement_dimensions(nmin, nmax):
            if not _supports_dimension(algorithm_name, n, k):
                continue
            for positive in (True, False):
                if (
                    algorithm_name.startswith("lc_css")
                    and not positive
                    and (n, k) in _LC_CSS_NEGATIVE_EXCLUDED_DIMENSIONS
                ):
                    continue
                generator = RandomCaseGenerator(algorithm_name, n, k, positive)
                if verbose:
                    print(f"    [[{n},{k}]] {'positive' if positive else 'negative'}")
                statistics.append(
                    run_statistics(
                        algorithm,
                        generator,
                        seed,
                        nr_seeds,
                        output_file,
                        timeout=timeout,
                        max_memory_bytes=max_memory_bytes,
                        verbose=verbose,
                    )
                )
    return statistics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        action="append",
        metavar="SELECTOR",
        help="Exact algorithm name, shell wildcard, or regular expression; repeatable.",
    )
    parser.add_argument("--nmin", type=int, help="Minimum n (inclusive).")
    parser.add_argument("--nmax", type=int, help="Maximum n (inclusive).")
    parser.add_argument("--seed", type=int, default=42, help="Master seed.")
    parser.add_argument(
        "--nr-seeds", "--num-seeds", type=int, default=N_STATS, help="Cases per grid cell."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/prototypes.csv"), help="Append-only CSV output."
    )
    parser.add_argument("--timeout", type=float, help="Per-case timeout in seconds.")
    parser.add_argument("--memory-limit", type=parse_memory_limit, help="Per-case memory limit.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    validate_common_args(parser, args)
    try:
        args.algorithm = resolve_names(args.algorithm, ALGORITHMS)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_suite(
        args.algorithm,
        seed=args.seed,
        nr_seeds=args.nr_seeds,
        output_file=args.output,
        nmin=args.nmin,
        nmax=args.nmax,
        timeout=args.timeout,
        max_memory_bytes=args.memory_limit,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
