"""Run the fixed thesis invariant benchmark suite.

Usage::

    python3 -m benchmarks.thesis.thesis_invariants [OPTIONS]

CLI options:

``--invariant SELECTOR``
    Select an invariant by exact name, shell wildcard, or regular expression.
    The option is repeatable. The full selected family runs when it is omitted.
``--family {pm,lc,both}``
    Restrict selection to permutation, local-Clifford, or both invariant
    families (default: ``both``).
``--seed SEED``
    Master seed from which per-case seeds are derived (default: 42).
``--nr-seeds N`` / ``--num-seeds N``
    Number of generated cases per suite measurement (default: 5).
``--output PATH``
    CSV file to append to (default: ``results/invariants.csv``). A header is
    written automatically when the file does not exist or is empty.
``--timeout SECONDS``
    Optional wall-clock limit for each invariant call.
``--memory-limit SIZE``
    Optional per-call memory limit. Values such as ``512M``, ``4GiB``, or a
    raw byte count are accepted.
``--verbose``
    Print progress for invariants, dimensions, labels, and individual seeds.

The dimension grids and positive/negative case families are intentionally
fixed in this module so a normal invocation reproduces the thesis suite.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from src.invariants.lc_invariants import (
    preserved_local_weight_distribution,
    preserved_low_degree_local_invariant,
)
from src.invariants.pm_invariants import (
    preserved_linear_dependencies,
    preserved_pauli_weight_enumerator,
    preserved_weight_enumerator,
)

from . import parse_memory_limit, resolve_names, validate_common_args
from ..experiments.generators_random import (
    LCEqCodePairGenerator,
    NonLCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from ..experiments.statistics import BenchmarkCase, Statistic, run_statistics
from ..experiments.generators_structured import (
    NonPEqCodePairGenerator as StructuredNonPEqCodePairGenerator,
)
from ..experiments.generators_structured import (
    PEqCodePairGenerator as StructuredPEqCodePairGenerator,
)
from ..experiments.generators_structured import load_named_code, named_code_names


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

PM_INVARIANTS: dict[str, Callable[..., bool]] = {
    "pm_weight_enumerator": preserved_weight_enumerator,
    "pm_pauli_weight_enumerator": preserved_pauli_weight_enumerator,
    "pm_linear_dependencies": preserved_linear_dependencies,
}
LC_INVARIANTS: dict[str, Callable[..., bool]] = {
    "lc_local_weight_distribution": preserved_local_weight_distribution,
    "lc_local_weight_distribution_s2": partial(
        preserved_local_weight_distribution, max_subset_size=2
    ),
    "lc_local_weight_distribution_s4": partial(
        preserved_local_weight_distribution, max_subset_size=4
    ),
    "lc_low_degree_local_invariant": preserved_low_degree_local_invariant,
    "lc_low_degree_local_invariant_s2": partial(
        preserved_low_degree_local_invariant, max_subset_size=2
    ),
    "lc_low_degree_local_invariant_s4": partial(
        preserved_low_degree_local_invariant, max_subset_size=4
    ),
}
INVARIANTS = {**PM_INVARIANTS, **LC_INVARIANTS}


def invariant_dimensions(pm: bool) -> list[tuple[int, int]]:
    """Return the fixed bounded ``(n, k)`` grid for one invariant family."""
    dimensions: list[tuple[int, int]] = []
    for n in PM_INVARIANT_NS if pm else LC_INVARIANT_NS:
        if n > 50:
            ks: set[int] = set(KNOWN_INVARIANT_KS.get(n, set()))
        elif n < 7:
            ks = set(range(0, n))
        elif n < 15:
            ks = set(range(0, n, 2)) | {n // 2}
        elif n < 30:
            ks = set(range(0, n, 4)) | {n // 2}
        else:
            ks = {0, 1, n // 4, n // 2, 3 * n // 4, n - 1}
        ks |= KNOWN_INVARIANT_KS.get(n, set())
        dimensions.extend((n, k) for k in sorted(ks) if k < n)
    return dimensions


def _named_code_for_dimension(n: int, k: int) -> str | None:
    for name in named_code_names():
        code = load_named_code(name)
        if (code.n, code.k) == (n, k):
            return name
    return None


@dataclass(frozen=True)
class InvariantCaseGenerator:
    invariant_name: str
    n: int
    k: int
    positive: bool
    structured_name: str | None = None

    @property
    def seed_upper_bound(self) -> int:
        return 1_000_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "negative"
        return f"{self.invariant_name}_{self.n}_{self.k}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": self.invariant_name,
            "generator": self.__name__,
            "name": self.structured_name,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
            "density": None,
            "symmetry": None,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        if self.invariant_name.startswith("pm_"):
            if self.structured_name is not None:
                inputs = (
                    StructuredPEqCodePairGenerator.stabilizer_codes_basis_changed(
                        self.structured_name, seed
                    )
                    if self.positive
                    else StructuredNonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection(
                        self.structured_name, seed
                    )
                )
            else:
                inputs = (
                    PEqCodePairGenerator.stabilizer_codes_permuted(self.n, self.k, seed)
                    if self.positive
                    else NonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection(
                        self.n, self.k, seed
                    )
                )
        else:
            inputs = (
                LCEqCodePairGenerator.stabilizer_codes_local_clifford(
                    self.n, self.k, seed
                )
                if self.positive
                else NonLCEqCodePairGenerator.stabilizer_codes_independent(
                    self.n, self.k, seed
                )
            )
        return BenchmarkCase(tuple(inputs), self.positive, self.metadata)


def measurements(
    invariant_names: Sequence[str],
) -> Iterator[tuple[str, Callable[..., bool], InvariantCaseGenerator]]:
    """Yield the complete fixed invariant suite."""
    for invariant_name in invariant_names:
        pm = invariant_name.startswith("pm_")
        for n, k in invariant_dimensions(pm):
            structured_name = _named_code_for_dimension(n, k) if n > 50 else None
            for positive in (True, False):
                yield (
                    invariant_name,
                    INVARIANTS[invariant_name],
                    InvariantCaseGenerator(
                        invariant_name, n, k, positive, structured_name
                    ),
                )


def run_suite(
    invariant_names: Sequence[str],
    *,
    seed: int,
    nr_seeds: int,
    output_file: Path,
    timeout: float | None = None,
    max_memory_bytes: int | None = None,
    verbose: bool = False,
) -> list[Statistic]:
    statistics: list[Statistic] = []
    current_invariant: str | None = None
    for invariant_name, invariant, generator in measurements(invariant_names):
        if verbose and invariant_name != current_invariant:
            print(f"Running invariant: {invariant_name}")
            current_invariant = invariant_name
        if verbose:
            print(
                f"    [[{generator.n},{generator.k}]] "
                f"{'positive' if generator.positive else 'negative'}"
            )
        statistics.append(
            run_statistics(
                invariant,
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
        "--invariant",
        action="append",
        metavar="SELECTOR",
        help="Invariant name, shell wildcard, or regex; defaults to the full suite.",
    )
    parser.add_argument("--family", choices=("pm", "lc", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nr-seeds", "--num-seeds", type=int, default=N_INVARIANT_STATS
    )
    parser.add_argument("--output", type=Path, default=Path("results/invariants.csv"))
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--memory-limit", type=parse_memory_limit)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    validate_common_args(parser, args)
    available = {
        name: function
        for name, function in INVARIANTS.items()
        if args.family == "both" or name.startswith(f"{args.family}_")
    }
    try:
        args.invariant = resolve_names(args.invariant, available)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_suite(
        args.invariant,
        seed=args.seed,
        nr_seeds=args.nr_seeds,
        output_file=args.output,
        timeout=args.timeout,
        max_memory_bytes=args.memory_limit,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
