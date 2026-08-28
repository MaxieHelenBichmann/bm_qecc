"""Run the four hybrid algorithms on structured/named codes.

Usage::

    python3 -m benchmarks.thesis.thesis_hybrids [OPTIONS]

CLI options:

``--algorithm SELECTOR``
    Select a hybrid by exact name, shell wildcard, or regular expression. The
    option is repeatable. All four hybrids run when it is omitted.
``--code NAME``
    Restrict the suite to a named structured code. The option is repeatable;
    omitting it selects every registered named code.
``--nmin N`` / ``--nmax N``
    Restrict the inclusive block-length range of the selected named codes.
``--seed SEED``
    Master seed from which per-case seeds are derived (default: 42).
``--nr-seeds N`` / ``--num-seeds N``
    Number of generated presentations per code and label (default: 10).
``--output PATH``
    CSV file to append to (default: ``results/hybrids.csv``). A header is
    written automatically when the file does not exist or is empty.
``--timeout SECONDS``
    Optional wall-clock limit for each hybrid call.
``--memory-limit SIZE``
    Optional per-call memory limit. Values such as ``512M``, ``4GiB``, or a
    raw byte count are accepted.
``--verbose``
    Print progress for hybrids, named codes, labels, and individual seeds.

Every compatible selected code receives positive and negative presentations.
CSS-only hybrids automatically skip named codes that are not stored as CSS.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.css_code import CSSCode
from src.hybrids.lc_css import is_lceq_css
from src.hybrids.lc_stb import are_lceq
from src.hybrids.p_css import are_peq_css
from src.hybrids.p_stab import are_peq_stab

from . import parse_memory_limit, resolve_names, validate_common_args
from ..experiments.statistics import BenchmarkCase, Statistic, run_statistics
from ..experiments.generators_structured import (
    LCEqCodeGenerator,
    LCEqCodePairGenerator,
    NonLCEqCodeGenerator,
    NonLCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
    load_named_code,
    named_code_names,
)


N_STATS = 10


@dataclass(frozen=True)
class DecisionAlgorithm:
    name: str
    function: Callable[..., Any]

    @property
    def __name__(self) -> str:
        return self.name

    def __call__(self, *inputs: Any) -> bool:
        result = self.function(*inputs)
        return result if isinstance(result, bool) else result is not None


_ALGORITHM_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "pm_css_hybrid": are_peq_css,
    "pm_stb_hybrid": are_peq_stab,
    "lc_stb_hybrid": are_lceq,
    "lc_css_hybrid": is_lceq_css,
}
ALGORITHMS: dict[str, DecisionAlgorithm] = {
    name: DecisionAlgorithm(name, function)
    for name, function in _ALGORITHM_FUNCTIONS.items()
}


@dataclass(frozen=True)
class StructuredCaseGenerator:
    """Generate one labeled presentation of a fixed named code."""

    algorithm_name: str
    code_name: str
    n: int
    k: int
    positive: bool

    @property
    def seed_upper_bound(self) -> int:
        return 1_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "negative"
        return f"structured_{self.algorithm_name}_{self.code_name}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "generator": self.__name__,
            "name": self.code_name,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
            "density": None,
            "symmetry": None,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        name = self.code_name
        inputs: tuple[Any, ...]
        if self.algorithm_name == "pm_css_hybrid":
            inputs = (
                PEqCodePairGenerator.css_codes_basis_changed(name, seed)
                if self.positive
                else NonPEqCodePairGenerator.css_codes_cascaded(name, seed)
            )
        elif self.algorithm_name == "pm_stb_hybrid":
            inputs = (
                PEqCodePairGenerator.stabilizer_codes_basis_changed(name, seed)
                if self.positive
                else NonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection(
                    name, seed
                )
            )
        elif self.algorithm_name == "lc_stb_hybrid":
            inputs = (
                LCEqCodePairGenerator.stabilizer_codes_local_clifford(name, seed)
                if self.positive
                else NonLCEqCodePairGenerator.stabilizer_codes_independent(name, seed)
            )
        elif self.algorithm_name == "lc_css_hybrid":
            code = (
                LCEqCodeGenerator.stabilizer_code_local_clifford(name, seed)
                if self.positive
                else NonLCEqCodeGenerator.stabilizer_code_locally_rank_one(name, seed)
            )
            inputs = (code,)
        else:  # pragma: no cover - guarded by registry
            raise ValueError(f"Unknown hybrid: {self.algorithm_name}")
        return BenchmarkCase(tuple(inputs), self.positive, self.metadata)


def _selected_codes(
    names: Sequence[str] | None, nmin: int | None, nmax: int | None
) -> list[tuple[str, Any]]:
    selected = set(names or named_code_names())
    return [
        (name, code)
        for name in named_code_names()
        if name in selected
        for code in (load_named_code(name),)
        if (nmin is None or code.n >= nmin) and (nmax is None or code.n <= nmax)
    ]


def run_suite(
    algorithm_names: Sequence[str],
    *,
    seed: int,
    nr_seeds: int,
    output_file: Path,
    code_names: Sequence[str] | None = None,
    nmin: int | None = None,
    nmax: int | None = None,
    timeout: float | None = None,
    max_memory_bytes: int | None = None,
    verbose: bool = False,
) -> list[Statistic]:
    """Run the requested hybrids on all compatible selected structured codes."""
    statistics: list[Statistic] = []
    codes = _selected_codes(code_names, nmin, nmax)
    for algorithm_name in algorithm_names:
        if verbose:
            print(f"Running hybrid: {algorithm_name}")
        for code_name, code in codes:
            if algorithm_name in {"pm_css_hybrid", "lc_css_hybrid"} and not isinstance(
                code, CSSCode
            ):
                continue
            for positive in (True, False):
                generator = StructuredCaseGenerator(
                    algorithm_name, code_name, code.n, code.k, positive
                )
                if verbose:
                    print(f"    {code_name} {'positive' if positive else 'negative'}")
                statistics.append(
                    run_statistics(
                        ALGORITHMS[algorithm_name],
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
        help="Hybrid name, shell wildcard, or regex; defaults to all four.",
    )
    parser.add_argument(
        "--code", action="append", choices=named_code_names(), help="Named code; repeatable."
    )
    parser.add_argument("--nmin", type=int, help="Minimum n (inclusive).")
    parser.add_argument("--nmax", type=int, help="Maximum n (inclusive).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nr-seeds", "--num-seeds", type=int, default=N_STATS)
    parser.add_argument("--output", type=Path, default=Path("results/hybrids.csv"))
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--memory-limit", type=parse_memory_limit)
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
        code_names=args.code,
        nmin=args.nmin,
        nmax=args.nmax,
        timeout=args.timeout,
        max_memory_bytes=args.memory_limit,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
