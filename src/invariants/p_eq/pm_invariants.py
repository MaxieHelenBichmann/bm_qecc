"""Small invariants that are preserved under permutation equivalence, which can be used to quickly rule out P-equivalence in some cases.
"""

from __future__ import annotations
from collections import Counter

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ...core.stabilizer_code import StabilizerCode
from ...core.css_code import CSSCode

def preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n



def preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k



def preserved_rank(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the rank of the stabilizer tableau is preserved, which is a necessary condition for P-equivalence."""
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)
    
    if isinstance(c1, CSSCode) and isinstance(c2, CSSCode):
        return _rank(c1.Hx) == _rank(c2.Hx) and _rank(c1.Hz) == _rank(c2.Hz)
    else:        
        return _rank(c1.symplectic[:, :c1.n]) == _rank(c2.symplectic[:, :c2.n]) and _rank(c1.symplectic[:, c1.n:]) == _rank(c2.symplectic[:, c2.n:])



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



def preserved_weight_enumerator(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the weight enumerator is preserved, which is a necessary condition for P-equivalence."""
    def _row_basis(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.uint8) & 1

        if M.shape[0] == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)

        B = mod2.row_basis(M)

        if hasattr(B, "toarray"):
            B = B.toarray()

        B = np.asarray(B, dtype=np.uint8) & 1

        if B.size == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)

        if B.ndim == 1:
            B = B.reshape(1, -1)

        return B

    def _weight_enumerator(M: np.ndarray) -> list[int]:
        M = _row_basis(M) # remove potentially redundant generators
        n = M.shape[1]
        r = M.shape[0]

        enumerator = [1] + [0] * n

        word = np.zeros(n, dtype=np.uint8)
        previous_gray = 0

        for t in range(1, 1 << r): # issue: O(2^r) -> not cheap invariant
            gray = t ^ (t >> 1)
            changed = gray ^ previous_gray
            row_idx = changed.bit_length() - 1

            word ^= M[row_idx]
            enumerator[int(word.sum())] += 1

            previous_gray = gray

        return enumerator
    
    if isinstance(c1, CSSCode) and isinstance(c2, CSSCode):
        return _weight_enumerator(c1.Hx) == _weight_enumerator(c2.Hx) and _weight_enumerator(c1.Hz) == _weight_enumerator(c2.Hz)
    else:
        return _weight_enumerator(c1.symplectic) == _weight_enumerator(c2.symplectic)

def preserved_pauli_weight_enumerator(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the weight enumerator is preserved, which is a necessary condition for P-equivalence."""
    def _row_basis(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.uint8) & 1

        if M.shape[0] == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)

        B = mod2.row_basis(M)

        if hasattr(B, "toarray"):
            B = B.toarray()

        B = np.asarray(B, dtype=np.uint8) & 1

        if B.size == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)

        if B.ndim == 1:
            B = B.reshape(1, -1)

        return B

    def _pauli_weight_enumerator(M: np.ndarray) -> Counter:
        M = _row_basis(M) # remove potentially redundant generators
        M = np.asarray(M, dtype=np.uint8) & 1

        n = M.shape[1] // 2
        r = M.shape[0]

        X = M[:, :n]
        Z = M[:, n:]

        enumerator = Counter()
        enumerator[(0,0,0)] = 1

        x_word = np.zeros(n, dtype=np.uint8)
        z_word = np.zeros(n, dtype=np.uint8)

        previous_gray = 0

        for t in range(1, 1 << r): # issue: O(2^r) -> not cheap invariant
            gray = t ^ (t >> 1)
            changed = gray ^ previous_gray
            row_idx = changed.bit_length() - 1

            x_word ^= X[row_idx]
            z_word ^= Z[row_idx]

            nx = int(np.count_nonzero(x_word & ~z_word))
            ny = int(np.count_nonzero(x_word & z_word))
            nz = int(np.count_nonzero(~x_word & z_word))

            enumerator[(nx, ny, nz)] += 1

            previous_gray = gray

        return enumerator
    
    
    return _pauli_weight_enumerator(c1.symplectic) == _pauli_weight_enumerator(c2.symplectic)


def preserved_linear_dependencies(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the linear dependencies between columns are preserved, which is a necessary condition for P-equivalence."""
    def _linear_dependencies(M: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        def _rank(matrix: np.ndarray) -> int:
            if matrix.shape[0] == 0:
                return 0
            return mod2.rank(matrix)
        
        n = M.shape[1] // 2
        
        one_columns = [ _rank(np.column_stack([M[:, q], M[:, q + n]])) for q in range(n) ]

        two_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n]])) for i in range(n) for j in range(i + 1, n) ]
        three_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n], M[:, k], M[:, k + n]])) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n) ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))
    
    return _linear_dependencies(c1.symplectic) == _linear_dependencies(c2.symplectic)