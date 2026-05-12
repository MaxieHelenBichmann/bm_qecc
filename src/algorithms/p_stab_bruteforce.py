"""Brute-force permutation equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from ldpc.mod2.mod2_numpy import rank

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab_bruteforce(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check permutation equivalence by brute-force search over all elements of S_n.

    Each permutation is checked by permuting the columns of the stabilizer tableau and checking for equality of their according row spaces using the rank.

    Each row space check should be done in O(n^3) time, and there are O(n!) permutations, so the overall runtime is O(n! * n^3) which is obviously not efficient at all.
    """
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return rank(matrix)

    c1_rank = _rank(c1.symplectic)

    for perm in permutations(range(c1.n)):
        perm = np.array(perm)
        perm_symplectic = np.concatenate([perm, perm + c1.n])

        if (c1_rank == _rank(c2.symplectic[:, perm_symplectic]) == _rank(np.vstack([c1.symplectic, c2.symplectic[:, perm_symplectic]]))):
            return True
    
    return False
