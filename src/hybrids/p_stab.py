"""Best hybrid solution for checking whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
import z3

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab(c1: StabilizerCode, c2: StabilizerCode) -> None | list[int]:
    """Check whether two stabilizer codes are permutation-equivalent.

    Returns: None if they are not permutation-equivalent, otherwise returns a permutation p with p[i] = j iff c1[i] -> c2[j]
    """
    cheap_invariants = (
        preserved_n,
        preserved_k,
        preserved_d,
        preserved_rank,
        preserved_number_zero_columns,
        preserved_number_duplicate_columns,
    )

    if not all(invariant(c1, c2) for invariant in cheap_invariants):
        return None
    
    if c1.n < 1:
        return list(range(c1.n))
    
    reduced_symplectic_1 = _row_basis(c1.symplectic)
    reduced_symplectic_2 = _row_basis(c2.symplectic)
    
    if c1.n <= 5:
        return _bruteforce(reduced_symplectic_1, reduced_symplectic_2)

    if not preserved_linear_dependencies(reduced_symplectic_1, reduced_symplectic_2):
        return None

    partition1: dict[tuple[int, ...], list[int]] = {
        (0,): list(range(c1.n))
    }
    partition2: dict[tuple[int, ...], list[int]] = {
        (0,): list(range(c2.n))
    }
    if c1.n <= 20:
        result, refined_partition1, refined_partition2 = preserved_punctured_hull_weight_enumerator(reduced_symplectic_1, reduced_symplectic_2)

        if not result:
            return None

        assert refined_partition1 is not None
        assert refined_partition2 is not None
        partition1 = refined_partition1
        partition2 = refined_partition2
        
    return _sat(reduced_symplectic_1, partition1, reduced_symplectic_2, partition2)

# ----------------------------------------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------------------------------------

def preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k

def preserved_d(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the distance is preserved, which is a necessary condition for P-equivalence."""
    return c1.distance == c2.distance
    
def preserved_rank(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the rank of the stabilizer tableau is preserved, which is a necessary condition for P-equivalence."""
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

def preserved_linear_dependencies(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check whether the linear dependencies between columns are preserved, which is a necessary condition for P-equivalence."""
    def _linear_dependencies(M: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        n = M.shape[1] // 2
        
        one_columns = [ _rank(np.column_stack([M[:, q], M[:, q + n]])) for q in range(n) ]

        two_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n]])) for i in range(n) for j in range(i + 1, n) ]
        three_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n], M[:, k], M[:, k + n]])) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n) ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))
    
    return _linear_dependencies(c1) == _linear_dependencies(c2)

def preserved_punctured_hull_weight_enumerator(c1: np.ndarray, c2: np.ndarray) -> tuple[bool, dict[tuple[int, ...], list[int]] | None, dict[tuple[int, ...], list[int]] | None]:
    """SENDRIER - p_stab_classical.py"""
    def _symplectic_to_gf4(symplectic: np.ndarray) -> np.ndarray:
        """
        I -> 0 = 00 = 0
        X -> 1 = 01 = 1
        Z -> w = 10 = 2
        Y -> w_bar = w + 1 = 11 = 3
        """
        n = symplectic.shape[1] // 2
        return symplectic[:, :n] + 2 * symplectic[:, n:]
    
    def _compute_signatures(matrix: np.ndarray) -> list[tuple[int, ...]]:
        """Compute the combined Sendriers invariant of the weight enumerator of the hull of the punctured code of each column of the code.
        """
        def _gf4_column_gram_contributions(M: np.ndarray) -> np.ndarray:
            k, n = M.shape
            contributions = np.zeros((n, k, k), dtype=np.uint8)

            for col in range(n):
                x = M[:, col] & 1
                z = M[:, col] >> 1
                contributions[col, :, :] = (x[:, None] & z[None, :]) ^ (z[:, None] & x[None, :])

            return contributions

        def _gf4_rref(matrix: np.ndarray) -> tuple[int, np.ndarray]:
            matrix = matrix.copy()
            m, n = matrix.shape
            rank = 0
            row = 0

            for bit_col in range(2 * n):
                col = bit_col % n
                bit = bit_col // n

                pivot = None
                for r in range(row, m):
                    if (matrix[r, col] >> bit) & 1:
                        pivot = r
                        break

                if pivot is None:
                    continue

                if pivot != row:
                    matrix[[row, pivot]] = matrix[[pivot, row]]

                for r in range(m):
                    if r != row and ((matrix[r, col] >> bit) & 1):
                        matrix[r, :] ^= matrix[row, :]

                rank += 1
                row += 1
                if row == m:
                    break

            return rank, matrix

        def _gf4_row_basis(M: np.ndarray) -> np.ndarray:
            rank, rref = _gf4_rref(M)
            return rref[:rank, :]

        def _gf2_gf4_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            m, ra = A.shape
            rb, n = B.shape
            if ra != rb:
                raise ValueError("Incompatible shapes for matrix multiplication.")

            C = np.zeros((m, n), dtype=np.uint8)
            for i in range(m):
                rows = np.flatnonzero(A[i])
                if rows.size:
                    C[i, :] = np.bitwise_xor.reduce(B[rows, :], axis=0)

            return C

        def _weight_enumerator_of_hull_punctured(col_idx: int) -> list[int]:
            Mp = np.delete(matrix, col_idx, axis=1)
            m_p = Mp.shape[1]

            if Mp.shape[0] == 0:
                return [1] + [0] * m_p

            # gram is in GF(2) due to the trace inner product that simulates the symplectic product (aka commutation/anti-commutation)
            gram = full_gram ^ column_gram_contributions[col_idx]

            coeff_basis = _kernel_basis(gram.T) # c @ gram = gram.T @ c.T = 0 -> x = c @ Mp with <x, Mp[i]> = 0 for all rows j -> x orthogonal to all rows of Mp -> x in Mp perp

            if coeff_basis.shape[0] == 0:
                hull_basis = np.zeros((0, m_p), dtype=np.uint8)
            else:
                hull_basis = _gf4_row_basis(_gf2_gf4_matmul(coeff_basis, Mp)) # c @ Mp = x -> words in Mp that are orthogonal to all rows of Mp -> hull

            hull_h, hull_n = hull_basis.shape
            enumerator = [1] + [0] * m_p

            word = np.zeros(hull_n, dtype=np.uint8)
            previous_gray = 0

            for t in range(1, 1 << hull_h):
                gray = t ^ (t >> 1)
                changed = gray ^ previous_gray
                row_idx = changed.bit_length() - 1

                # GF(2)-additive
                word ^= hull_basis[row_idx]

                wt = int(np.count_nonzero(word))
                enumerator[wt] += 1

                previous_gray = gray

            return enumerator
        
        column_gram_contributions = _gf4_column_gram_contributions(matrix)
        full_gram = np.bitwise_xor.reduce(column_gram_contributions, axis=0, initial=0)

        invariants = []

        for col_idx in range(matrix.shape[1]):
            inv = tuple(_weight_enumerator_of_hull_punctured(col_idx))
            invariants.append(inv)

        return invariants

    def _partition_columns_by_invariants(invariants: list[tuple[int, ...]]) -> dict[tuple[int, ...], list[int]]:
        partition = defaultdict(list)
        for idx, inv in enumerate(invariants):
            partition[inv].append(idx)
        return {k: sorted(v) for k, v in sorted(partition.items(), key=lambda item: item[0])}
    
    gf4_tableau_c1 = _symplectic_to_gf4(c1)
    gf4_tableau_c2 = _symplectic_to_gf4(c2)

    signatures_c1 = _compute_signatures(gf4_tableau_c1)
    signatures_c2 = _compute_signatures(gf4_tableau_c2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    if partition_c1.keys() != partition_c2.keys():
        return False, None, None
    if any(len(partition_c1[k]) != len(partition_c2[k]) for k in partition_c1):
        return False, None, None

    for key1, key2 in zip(partition_c1.keys(), partition_c2.keys()):
        if key1 != key2:
            return False, None, None
        if len(partition_c1[key1]) != len(partition_c2[key2]):
            return False, None, None
        
    return True, partition_c1, partition_c2


# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------

def _bruteforce(c1: np.ndarray, c2: np.ndarray) -> None | list[int]:
    """p_stab_bruteforce.py"""
    c1_rank = _rank(c1)
    n = c1.shape[1] // 2

    for perm in permutations(range(n)):
        perm_symplectic = perm + tuple(q + n for q in perm)

        if c1_rank == _rank(np.vstack([c1, c2[:, perm_symplectic]])):
            return list(perm)

    return None

def _sat(c1: np.ndarray, partition1: dict[tuple[int, ...], list[int]], c2: np.ndarray, partition2: dict[tuple[int, ...], list[int]]) -> None | list[int]:
    """p_stab_sat.py"""
    def _elementwise_map(normal_bool, variables):
        return z3.And([
            v if bit == 1 else z3.Not(v)
            for bit, v in zip(normal_bool, variables)
        ])

    def _exactly_one(variables):
        return z3.PbEq([(v, 1) for v in variables], 1)

    def _xor_list(variables):
        acc = z3.BoolVal(False)
        for v in variables:
            acc = z3.Xor(acc, v)
        return acc
    
    solver = z3.Solver()

    r, n = c1.shape[0], c1.shape[1] // 2

    # permutations
    aux_tableau = [z3.Bool(f'aux_{row}_{col}') for row in range(r) for col in range(2*n)]

    permutation_variables = {(i,j) :z3.Bool(f'p_{i}_{j}') for sig, col1 in partition1.items() for i in col1 for j in partition2[sig] }

    for i in range(n):
        solver.add(_exactly_one([var for (src, _), var in permutation_variables.items() if src == i]))
    for j in range(n):
        solver.add(_exactly_one([ var for (_, tgt), var in permutation_variables.items() if tgt == j]))

    for (i,j), permutation_variable in permutation_variables.items():
            x_column_original = c1[:, i]
            z_column_original = c1[:, i + n]

            x_column_permuted = [aux_tableau[row * (2*n) + j] for row in range(r)]
            z_column_permuted = [aux_tableau[row * (2*n) + j + n] for row in range(r)]

            solver.add(z3.Implies(permutation_variable, z3.And(_elementwise_map(x_column_original, x_column_permuted), _elementwise_map(z_column_original, z_column_permuted))))

    # row operations
    row_operation_coefficients = [z3.Bool(f'r_{i}_{j}') for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):

            row_contributions = []
            for contribution in range(r):
                if c2[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients[row * r + contribution])

            solver.add(aux_tableau[row * (2*n) + q] == _xor_list(row_contributions))

    if solver.check() != z3.sat:
        return None

    perm = [-1] * n
    model = solver.model()
    for i in range(n):
        perm[i] =  next(
            j
            for (src, j), var in permutation_variables.items()
            if src == i and z3.is_true(
                model.eval(var, model_completion=True)
            )
        )
    return perm


# ----------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------

def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return mod2.rank(matrix)

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
