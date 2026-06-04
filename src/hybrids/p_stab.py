"""Best hybrid solution for checking whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import multiprocessing
from collections import Counter
from itertools import permutations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    cheap_invariants = (
        preserved_n,
        preserved_k,
        preserved_rank,
        preserved_number_zero_columns,
        preserved_number_duplicate_columns,
    )

    if not all(invariant(c1, c2) for invariant in cheap_invariants):
        return False
    
    if c1.n <= 5: # TODO: better threshold with benchmarks?
        return _bruteforce(c1, c2)

    
    more_expensive_invariants = (
            preserved_linear_dependencies,
    )

    if not all(invariant(c1, c2) for invariant in more_expensive_invariants):
        return False

    return True

# ----------------------------------------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------------------------------------

def preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k

def preserved_rank(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the rank of the stabilizer tableau is preserved, which is a necessary condition for P-equivalence."""
    return _rank(c1.symplectic) == _rank(c2.symplectic)

def preserved_number_zero_columns(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of zero columns is preserved, which is a necessary condition for P-equivalence."""
    return int(np.count_nonzero(np.all(c1.symplectic == 0, axis=0))) == int(np.count_nonzero(np.all(c2.symplectic == 0, axis=0)))

def preserved_number_duplicate_columns(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of duplicate columns is preserved, which is a necessary condition for P-equivalence."""
    def _duplicate_column(M: np.ndarray) -> list[int]:
        columns = [tuple(M[:, j].tolist()) for j in range(M.shape[1])]
        counts = Counter(columns)
        return sorted(counts.values())

    return _duplicate_column(c1.symplectic) == _duplicate_column(c2.symplectic)

def preserved_linear_dependencies(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the linear dependencies between columns are preserved, which is a necessary condition for P-equivalence."""
    def _linear_dependencies(M: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        n = M.shape[1] // 2
        
        one_columns = [ _rank(np.column_stack([M[:, q], M[:, q + n]])) for q in range(n) ]

        two_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n]])) for i in range(n) for j in range(i + 1, n) ]
        three_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n], M[:, k], M[:, k + n]])) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n) ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))
    
    return _linear_dependencies(c1.symplectic) == _linear_dependencies(c2.symplectic)

# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------

def _bruteforce(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """p_stab_bruteforce.py"""
    c1_rank = _rank(c1.symplectic)

    for perm in permutations(range(c1.n)):
        perm_symplectic = perm + tuple(q + c1.n for q in perm)

        if c1_rank == _rank(np.vstack([c1.symplectic, c2.symplectic[:, perm_symplectic]])):
            return True

    return False


# ----------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------

def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return mod2.rank(matrix)