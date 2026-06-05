#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
from collections.abc import Iterable

# python3 data/generate.py 14 3 42
# python3 data/generate.py 20 4 55
# python3 data/generate.py 30 5 654
# python3 data/generate.py 40 10 69
# python3 data/generate.py 50 15 1337
# python3 data/generate.py 60 20 245
# python3 data/generate.py 70 12 7

# python3 data/generate.py 10 4 1337 --css=True
# python3 data/generate.py 11 5 1337 --css=True
# python3 data/generate.py 12 9 4 --css=True
# python3 data/generate.py 13 2 64 --css=True
# python3 data/generate.py 14 6 5 --css=True
# python3 data/generate.py 15 10 65 --css=True
# python3 data/generate.py 16 3 8 --css=True
# python3 data/generate.py 17 7 2 --css=True
# python3 data/generate.py 18 5 67 --css=True
# python3 data/generate.py 19 9 94 --css=True
# python3 data/generate.py 20 12 62 --css=True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_MULTIPLE_PM_CSS_ALGORITHMS = (
    "pm_css_sat",
    "pm_css_matroid",
    "pm_css_graph_iso",
    "pm_css_classical",
    "pm_css_bruteforce",
)


def write_code(code, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = code.stabs_as_pauli_strings()
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} stabilizer generators to {output_path}")


def pair_paths(output_dir: Path, n: int, k: int, seed: int, suffix: str) -> tuple[Path, Path]:
    return (
        output_dir / f"random_css_{suffix}_{n}_{k}_{seed}_1.txt",
        output_dir / f"random_css_{suffix}_{n}_{k}_{seed}_2.txt",
    )


def all_paths_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def write_pair(code1, code2, paths: tuple[Path, Path], *, force: bool) -> bool:
    if not force and all_paths_exist(paths):
        print(f"Skipping existing pair {paths[0].name}, {paths[1].name}")
        return False

    write_code(code1, paths[0])
    write_code(code2, paths[1])
    return True


def generate_run_multiple_css_caches(global_seed: int, output_dir: Path, *, force: bool) -> None:
    """Generate seed-specific negative CSS cache files with n > 17."""
    import numpy as np

    from benchmarks.run import MEAS_STATS, N_STATS, max_n_pm_css
    from benchmarks.utils import random_non_permuted_css_pair

    rng = np.random.default_rng(global_seed)
    seeds = [int(seed) for seed in rng.integers(0, 1000, size=N_STATS)]
    sizes = [
        (n, k)
        for n in MEAS_STATS
        if n > 17
        for k in range(0, n, 1 if n < 7 else 2 if n < 15 else 4)
    ]

    generated = 0
    skipped = 0

    for n, k in sizes:
        needs_negative = any(n <= max_n_pm_css(algorithm, positive=False) for algorithm in RUN_MULTIPLE_PM_CSS_ALGORITHMS)

        for seed in seeds:
            if needs_negative:
                paths = pair_paths(output_dir, n, k, seed, "non_peq")
                if force or not all_paths_exist(paths):
                    code1, code2 = random_non_permuted_css_pair(n, k, seed=seed)
                    generated += int(write_pair(code1, code2, paths, force=force))
                else:
                    skipped += 1

    print(
        f"Done. Generated {generated} negative CSS cache pairs, skipped {skipped} existing pairs "
        f"for global seed {global_seed} in {output_dir}."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate a large code as Paulis text.")
    parser.add_argument("n", type=int, nargs="?", help="n of the code.")
    parser.add_argument("k", type=int, nargs="?", help="k of the code.")
    parser.add_argument("seed", type=int, nargs="?", help="Seed for random number generation.")
    parser.add_argument("output", type=Path, nargs="?", help="Output .txt path.")
    parser.add_argument("--css", type=bool, default=False, help="Whether to generate CSS codes instead of general stabilizer codes.")
    parser.add_argument(
        "--run-multiple-css-caches",
        action="store_true",
        help="Generate negative n > 17 random CSS cache files used by benchmarks/run_multiple.sh.",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=42,
        help="Global benchmark seed for --run-multiple-css-caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for --run-multiple-css-caches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    args = parser.parse_args()

    if args.run_multiple_css_caches:
        generate_run_multiple_css_caches(args.global_seed, args.output_dir, force=args.force)
        return

    if args.n is None or args.k is None or args.seed is None:
        parser.error("n, k, and seed are required unless --run-multiple-css-caches is used.")

    if args.css:
        output = args.output or Path(__file__).resolve().parent / f"random_css_{args.n}_{args.k}.txt"
    else:
        output = args.output or Path(__file__).resolve().parent / f"random_stab_{args.n}_{args.k}.txt"

    from benchmarks.utils import (
        random_non_permuted_stabilizer_pair,
        random_permuted_stabilizer_pair,
        random_non_permuted_css_pair,
        random_permuted_css_pair,
    )

    if args.css:
        code1, code2 = random_permuted_css_pair(args.n, args.k, seed=args.seed)
        code1_non, code2_non = random_non_permuted_css_pair(args.n, args.k, seed=args.seed + 20)
    else:
        code1, code2 = random_permuted_stabilizer_pair(args.n, args.k, seed=args.seed)
        code1_non, code2_non = random_non_permuted_stabilizer_pair(args.n, args.k, seed=args.seed + 20)

    write_code(code1, output.with_name(output.stem + "1_peq.txt"))
    write_code(code1_non, output.with_name(output.stem + "1_non_peq.txt"))
    write_code(code2, output.with_name(output.stem + "2_peq.txt"))
    write_code(code2_non, output.with_name(output.stem + "2_non_peq.txt"))


if __name__ == "__main__":
    main()
