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
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from benchmarks.thesis import resolve_names
from benchmarks.thesis.thesis_prototypes import ALGORITHMS, run_suite
from paper.benchmarks.utils.config import ALGORITHM_DATA_DIR

MASTER_SEED = 42
NUM_SEEDS = 10
TIMEOUT_SECONDS = 5_400.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
VERBOSE = True
OUTPUT_DIRECTORY = ALGORITHM_DATA_DIR

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
        if VERBOSE:
            print(
                f"{algorithm_name}: n={nmin}..{nmax}, {NUM_SEEDS} seeds/cell "
                f"-> {output_file}",
                flush=True,
            )
        run_suite(
            [algorithm_name],
            seed=MASTER_SEED,
            nr_seeds=NUM_SEEDS,
            output_file=output_file,
            nmin=nmin,
            nmax=nmax,
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
        help="Exact algorithm name, shell wildcard, or regex; repeatable.",
    )
    args = parser.parse_args(argv)
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
