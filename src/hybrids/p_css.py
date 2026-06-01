"""Best hybrid solution for checking whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

from collections import Counter, defaultdict

import hashlib
import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ..core.css_code import CSSCode

def are_peq_css(c1: CSSCode, c2: CSSCode) -> bool:
    invariants = (
        preserved_n,
        preserved_k,
        preserved_rank,
        preserved_number_zero_columns,
        preserved_number_duplicate_columns,
        preserved_linear_dependencies,
        preserved_punctured_hull_weight_enumerator,
    )

    if not all(invariant(c1, c2) for invariant in invariants):
        return False
    
    return True

# ----------------------------------------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------------------------------------

def preserved_n(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k

def preserved_rank(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the rank of the stabilizer tableau is preserved, which is a necessary condition for P-equivalence."""
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)
    
    return _rank(c1.Hx) == _rank(c2.Hx) and _rank(c1.Hz) == _rank(c2.Hz)

def preserved_number_zero_columns(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of zero columns is preserved, which is a necessary condition for P-equivalence."""
    return int(np.count_nonzero(np.all(c1.symplectic == 0, axis=0))) == int(np.count_nonzero(np.all(c2.symplectic == 0, axis=0)))

def preserved_number_duplicate_columns(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of duplicate columns is preserved, which is a necessary condition for P-equivalence."""
    def _duplicate_column(M: np.ndarray) -> list[int]:
        columns = [tuple(M[:, j].tolist()) for j in range(M.shape[1])]
        counts = Counter(columns)
        return sorted(counts.values())

    return _duplicate_column(c1.symplectic) == _duplicate_column(c2.symplectic)

def preserved_linear_dependencies(c1: CSSCode, c2: CSSCode) -> bool:
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

def preserved_punctured_hull_weight_enumerator(c1: CSSCode, c2: CSSCode) -> bool:
    """SENDRIER - p_css_classical.py"""
    def _kernel_basis(A: np.ndarray) -> np.ndarray:
        A = (np.asarray(A) & 1).astype(np.uint8)
        K = mod2.nullspace(A)
        if hasattr(K, "toarray"):
            K = K.toarray()
        K = (np.asarray(K) & 1).astype(np.uint8)
        if K.size == 0:
            return np.zeros((0, A.shape[1]), dtype=np.uint8)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        if K.shape[1] != A.shape[1]:
            raise ValueError(
                "Kernel basis must have the same number of columns as the input matrix."
            )
        return K

    def _row_basis(M: np.ndarray) -> np.ndarray:
        M = (np.asarray(M) & 1).astype(np.uint8)
        if M.size == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)
        B = mod2.row_basis(M)
        if hasattr(B, "toarray"):
            B = B.toarray()
        B = (np.asarray(B) & 1).astype(np.uint8)
        if B.size == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        return B
    
    def _generator_matrix_from_parity_check(H: np.ndarray, n: int) -> np.ndarray:
        if H.size == 0 or H.shape[0] == 0:
            return np.eye(n, dtype=np.uint8)
        return _kernel_basis(H)
    
    def _compute_signatures(G1: np.ndarray, G2: np.ndarray) -> list[int]:
        def _weight_enumerator_of_hull_punctured(G: np.ndarray, col_idx: int) -> list[int]:
            Gp = np.delete(G, col_idx, axis=1)
            g_p = Gp.shape[1]

            gram = (Gp @ Gp.T) & 1

            if gram.size == 0:
                hull_basis = np.zeros((0, g_p), dtype=np.uint8)
            elif not gram.any():
                hull_basis = _row_basis(Gp)
            else:
                coeff_basis = _kernel_basis(gram)

                if coeff_basis.shape[0] == 0:
                    hull_basis = np.zeros((0, g_p), dtype=np.uint8)
                else:
                    hull_basis = _row_basis((coeff_basis @ Gp) & 1)

            h = hull_basis.shape[0]
            enumerator = [1] + [0] * g_p

            word = np.zeros(g_p, dtype=np.uint8)
            previous_gray = 0

            for t in range(1, 1 << h):
                gray = t ^ (t >> 1)
                changed = gray ^ previous_gray
                row_idx = changed.bit_length() - 1

                word ^= hull_basis[row_idx]
                enumerator[int(word.sum())] += 1

                previous_gray = gray

            return enumerator

        def _combine_invariants(inv_hx: list[int], inv_hz: list[int]) -> int:
            payload = (
                ",".join(map(str, inv_hx))
                + "|"
                + ",".join(map(str, inv_hz))
            ).encode("ascii")
            return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")

        invariants = []

        for col_idx in range(G1.shape[1]):
            inv1 = _weight_enumerator_of_hull_punctured(G1, col_idx)
            inv2 = _weight_enumerator_of_hull_punctured(G2, col_idx)

            invariants.append(_combine_invariants(inv1, inv2))

        return invariants
    
    def _partition_columns_by_invariants(invariants: list[int]) -> dict[int, list[int]]:
        partition = defaultdict(list)
        for idx, inv in enumerate(invariants):
            partition[inv].append(idx)
        return {k: v for k, v in sorted(partition.items())}


    Gx1 = _generator_matrix_from_parity_check(c1.Hx, c1.n)
    Gz1 = _generator_matrix_from_parity_check(c1.Hz, c1.n)
    Gx2 = _generator_matrix_from_parity_check(c2.Hx, c2.n)
    Gz2 = _generator_matrix_from_parity_check(c2.Hz, c2.n)

    signatures_c1 = _compute_signatures(Gx1, Gz1)
    signatures_c2 = _compute_signatures(Gx2, Gz2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    if partition_c1.keys() != partition_c2.keys():
        return False
    if any(len(partition_c1[k]) != len(partition_c2[k]) for k in partition_c1):
        return False

    for key1, key2 in zip(partition_c1.keys(), partition_c2.keys()):
        if key1 != key2:
            return False
        if len(partition_c1[key1]) != len(partition_c2[key2]):
            return False


# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------
