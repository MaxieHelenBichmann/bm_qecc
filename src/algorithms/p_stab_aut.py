"""Automorphism-group based equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from pynauty import autgrp
from itertools import permutations

from ..core.stabilizer_code import StabilizerCode

def _automorphisms(tableau: np.ndarray, n: int) -> list[int]:
    # TODO: find a way to make Magma / SageMath work maybe
    return [tuple(range(n))]

def are_peq_stab_aut(c1: StabilizerCode, c2: StabilizerCode) -> bool:    
    """Check permutation equivalence by brute-force search over all elements of S_n, but reducing the search space using automorphisms.

    Can be better than brute-force if the automorphism group of the code is large, but still has factorial worst-case runtime if the automorphism group is trivial.
    """
    def _compose(p, q):
        return tuple(p[q[i]] for i in range(len(q)))

    c2_rank = mod2.rank(c2.symplectic)
    aut_c2 = _automorphisms(c2.symplectic, c2.n)

    remaining_permutations = set(permutations(range(c1.n)))

    while len(remaining_permutations) > 0:
        perm = remaining_permutations.pop()

        perm = np.array(perm)
        perm_symplectic = np.concatenate([perm, perm + c1.n])

        if (mod2.rank(c1.symplectic[:, perm_symplectic]) == c2_rank == mod2.rank(np.vstack([c1.symplectic[:, perm_symplectic], c2.symplectic]))):
            return True
        else:
            # isomorphisms(c1, c2) = { α ∘ φ | α ∈ Aut(c2) } with φ: c1 -> c2
            isomorphisms = { _compose(alpha, perm) for alpha in aut_c2 }
            remaining_permutations -= isomorphisms
    
    return False
