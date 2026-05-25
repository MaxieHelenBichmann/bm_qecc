#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

# python3 data/generate.py 14 3 42
# python3 data/generate.py 20 4 55
# python3 data/generate.py 30 5 654
# python3 data/generate.py 40 10 69
# python3 data/generate.py 50 15 1337
# python3 data/generate.py 60 20 245
# python3 data/generate.py 70 12 7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_code(code, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = code.stabs_as_pauli_strings()
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} stabilizer generators to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a large code as Paulis text.")
    parser.add_argument("n", type=int, help="n of the code.")
    parser.add_argument("k", type=int, help="k of the code.")
    parser.add_argument("seed", type=int, help="Seed for random number generation.")
    parser.add_argument("output", type=Path, nargs="?", help="Output .txt path.")
    parser.add_argument("--css", type=bool, default=False, help="Whether to generate CSS codes instead of general stabilizer codes.")
    args = parser.parse_args()
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
