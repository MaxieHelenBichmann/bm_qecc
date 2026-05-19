"""Automorphism-group based equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from itertools import permutations
from pathlib import Path

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ..core.stabilizer_code import StabilizerCode

def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return mod2.rank(matrix)

_GAP_BEGIN = "__BM_QECC_GAP_AUT_PERMS_BEGIN__"
_GAP_END = "__BM_QECC_GAP_AUT_PERMS_END__"

def _run_gap(script: str) -> str:
    project_gap_root = Path(__file__).resolve().parents[2] / ".gap"
    gap_executable = os.environ.get("GAP_EXECUTABLE", "gap")
    if shutil.which(gap_executable) is None:
        raise RuntimeError(
            f"Could not find GAP executable {gap_executable!r}. "
            "Install GAP and make sure it is on PATH, or set GAP_EXECUTABLE."
        )

    proc = subprocess.run(
        [
            gap_executable,
            "-q",
            "-r",
            "--quitonbreak",
            "--roots",
            f";{project_gap_root}",
        ],
        input=script,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "GAP failed while computing the automorphism group.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    return proc.stdout

def _automorphisms(tableau: np.ndarray, n: int) -> list[tuple[int, ...]]:
    """Use GAP and the package GUAVA to compute the automorphism group of a stabilizer code."""
    def _extract_valid_permutations(aut_group) -> list[tuple[int, ...]]:
        perms = []
        for aut in aut_group:
            perm = [ aut[i] - 1 for i in range(2 * n) ]
            if all(x_i + n == z_i for x_i, z_i in zip(perm[:n], perm[n:])):
                perms.append(tuple(perm[:n]))
        return perms

    if tableau.shape[0] == 0:
        return list(permutations(range(n)))

    script = f"""
if LoadPackage("guava") = fail then
    Print("Could not load GAP package guava.");
    QUIT_GAP(1);
fi;

G := Z(2)^0 * {str(tableau.tolist()).replace(' ', '')};
C := GeneratorMatCode(G, GF(2));
AutC := AutomorphismGroup(C);
perms := List(Elements(AutC), g -> List([1..{2 * n}], i -> i^g));

Print("{_GAP_BEGIN}\\n");
Print(perms);
Print("\\n{_GAP_END}\\n");
QUIT;
"""
    output = _run_gap(script).split(_GAP_BEGIN, 1)[1].split(_GAP_END, 1)[0].strip()
    gap_perms = ast.literal_eval(output)

    return _extract_valid_permutations(gap_perms)

def are_peq_stab_aut(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check permutation equivalence by brute-force search over all elements of S_n, but reducing the search space using automorphisms.

    Can be better than brute-force if the automorphism group of the code is large, but still has factorial worst-case runtime if the automorphism group is trivial.
    """
    def _compose(p, q):
        return tuple(p[q[i]] for i in range(len(q)))

    c2_rank = _rank(c2.symplectic)
    aut_c2 = _automorphisms(c2.symplectic, c2.n)

    remaining_permutations = set(permutations(range(c1.n)))

    while len(remaining_permutations) > 0:
        perm = remaining_permutations.pop()

        perm = np.array(perm)
        perm_symplectic = np.concatenate([perm, perm + c1.n])

        if _rank(c1.symplectic[:, perm_symplectic]) == c2_rank == _rank(np.vstack([c1.symplectic[:, perm_symplectic], c2.symplectic])):
            return True

        # isomorphisms(c1, c2) = { α ∘ φ | α ∈ Aut(c2) } with φ: c1 -> c2
        isomorphisms = { _compose(perm, alpha) for alpha in aut_c2 }
        remaining_permutations -= isomorphisms

    return False
