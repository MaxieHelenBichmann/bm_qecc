#!/usr/bin/env python3
from pathlib import Path
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert a binary .npy matrix to I/X text.")
    parser.add_argument("input", type=Path, help="Path to the .npy file.")
    parser.add_argument("output", type=Path, nargs="?", help="Output .txt path.")
    parser.add_argument("character", type=str, nargs="?", help="Character to use for 1s.")
    args = parser.parse_args()

    matrix = np.load(args.input)
    if matrix.ndim != 2:
        raise SystemExit("Expected a 2D matrix.")
    if not np.isin(matrix, [0, 1]).all():
        raise SystemExit("Expected the matrix to contain only 0 and 1.")

    output = args.output or args.input.with_suffix(".txt")
    lines = ["".join(args.character if int(value) else "I" for value in row) for row in matrix]
    with output.open("a") as file:
        file.write("\n".join(lines) + "\n")
    print(f"Appended to {output}")


if __name__ == "__main__":
    main()
