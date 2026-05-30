"""Classical code equivalence based permutation equivalence checking.

References for this algorithm: 
- A. R. Calderbank, E. M. Rains, P. W. Shor, N. J. A. Sloane: Quantum Error Correction Via Codes Over GF(4)
- Nicolas Sendrier: Finding the Permutation Between Equivalent Linear Codes: The Support Splitting Algorithm
- Thomas Feulner: The Automorphism Groups of Linear Codes and Canonical Representatives of Their Semilinear Isometry Classes
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ...core.stabilizer_code import StabilizerCode

_GF4_MUL_TABLE = (
    (0, 0, 0, 0),
    (0, 1, 2, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
)

_GF4_CONJ_TABLE = (0, 1, 3, 2)

@dataclass(frozen=True, slots=True)
class GF4:
    """
    GF(4) = { 0, 1, w, w_bar } with w^2 = w + 1

    0 = 00
    1 = 01
    w = 10
    w_bar = w + 1 = 11
    
    (not original Calderbank/Rains/Shor/Sloane mapping, but a more convenient/intuitive in my opinion, and also Danielsen/Parker and MacAree/Howard)
    """
    value: int

    def __post_init__(self):
        if self.value not in (0, 1, 2, 3):
            raise ValueError("GF(4) elements must be 0, 1, 2, or 3.")

    def __add__(self, other: GF4) -> GF4:
        return GF4(self.value ^ other.value)

    def __sub__(self, other: GF4) -> GF4:
        return self + other

    def __neg__(self) -> GF4:
        return self

    def __mul__(self, other: GF4) -> GF4:
        return GF4(_GF4_MUL_TABLE[self.value][other.value])

    def __pow__(self, n: int) -> GF4:
        if n < 0:
            return self.inverse() ** (-n)
        result = GF4(1)
        base = self
        for _ in range(n):
            result *= base
        return result

    def inverse(self) -> GF4:
        if self.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse in GF(4).")
        return self ** 2

    def __truediv__(self, other: GF4) -> GF4:
        return self * other.inverse()

    def conjugate(self) -> GF4:
        # conjugation GF(4): x -> x^2
        return GF4(_GF4_CONJ_TABLE[self.value])

    def trace(self) -> int:
        # trace GF(4): Tr(x) = x + x^2
        return (self + self.conjugate()).value

    def is_zero(self) -> bool:
        return self.value == 0

    def __repr__(self) -> str:
        names = {
            0: "0",
            1: "1",
            2: "ω",
            3: "ω̄",
        }
        return names[self.value]

ZERO = GF4(0)
ONE = GF4(1)
W = GF4(2)
W_BAR = GF4(3)

def _symplectic_to_gf4(tableau: np.ndarray) -> np.ndarray:
    # I = (0|0) -> 0, X = (1|0) -> 1, Z = (0|1) -> w, Y = (1|1) -> w_bar
    n = tableau.shape[1] // 2
    gf4_entries = np.array([ZERO, ONE, W, W_BAR], dtype=object)
    values = tableau[:, :n] + 2 * tableau[:, n:]
    return gf4_entries[values]

def _gf4_to_symplectic(tableau: np.ndarray) -> np.ndarray:
    # 0 -> I = (0|0), 1 -> X = (1|0), w -> Z = (0|1), w_bar -> Y = (1|1)
    n = tableau.shape[1]
    r = tableau.shape[0]
    symplectic_matrix = np.empty((r, 2*n), dtype=np.uint8)
    for q in range(n):
        for i in range(r):
            value = tableau[i, q].value
            symplectic_matrix[i, q] = value & 1
            symplectic_matrix[i, q + n] = value >> 1

    return symplectic_matrix

def _gf4_rref(matrix: np.ndarray, to_col: int | None = None) -> tuple[int, np.ndarray, list[int]]:
    # matrix has GF(4) entries, this is NOT the normal RREF of GF(4)-linear codes, but of GF(4)-additive codes!
    def _gf4_bit(x: GF4, bit_col: int, n: int) -> int:
        return x.value >> (bit_col // n) & 1

    if matrix.shape[0] == 0:
        return 0, matrix, []

    matrix = matrix.copy()
    m, n = matrix.shape
    rank = 0
    row = 0
    pivot_columns = []

    if to_col is None:
        to_col = n

    for bit_col in range(2*to_col):
        col = bit_col % to_col

        pivot = None
        for r in range(row, m):
            if _gf4_bit(matrix[r, col], bit_col, to_col):
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            matrix[[row, pivot]] = matrix[[pivot, row]]

        for r in range(m):
            if r != row and _gf4_bit(matrix[r, col], bit_col, to_col):
                for c in range(n):
                    matrix[r, c] += matrix[row, c]

        pivot_columns.append(col)
        rank += 1
        row += 1
        if row == m:
            break
    return rank, matrix, pivot_columns

def _gf4_trace_inner_product(a: np.ndarray, b: np.ndarray) -> GF4:
    acc = 0
    for ai, bi in zip(a, b):
        av = ai.value
        bv = bi.value
        acc ^= ((av & 1) & (bv >> 1)) ^ ((av >> 1) & (bv & 1))
    return ONE if acc else ZERO

def _compute_signatures(generator_matrix: np.ndarray) -> list[tuple[int, ...]]:
    """Compute the combined Sendrier's invariant of the weight enumerator of the hull of the punctured code of each column of the code.
    """
    def _gf2_kernel_basis(A: np.ndarray) -> np.ndarray:
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

    def _gf4_values(G: np.ndarray) -> np.ndarray:
        values = np.empty(G.shape, dtype=np.uint8)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                values[i, j] = G[i, j].value
        return values

    def _gf4_column_gram_contributions(G: np.ndarray) -> np.ndarray:
        k, n = G.shape
        contributions = np.zeros((n, k, k), dtype=np.uint8)

        for col in range(n):
            x = G[:, col] & 1
            z = G[:, col] >> 1
            contributions[col, :, :] = (x[:, None] & z[None, :]) ^ (z[:, None] & x[None, :])

        return contributions

    def _gf4_value_rref(matrix: np.ndarray) -> tuple[int, np.ndarray, list[int]]:
        if matrix.shape[0] == 0:
            return 0, matrix, []

        matrix = matrix.copy()
        m, n = matrix.shape
        rank = 0
        row = 0
        pivot_columns = []

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

            pivot_columns.append(col)
            rank += 1
            row += 1
            if row == m:
                break

        return rank, matrix, pivot_columns

    def _gf4_value_row_basis(M: np.ndarray) -> np.ndarray:
        if M.shape[0] == 0:
            return np.zeros((0, M.shape[1]), dtype=np.uint8)

        rank, rref, _ = _gf4_value_rref(M)
        return rref[:rank, :]

    def _gf2_gf4_value_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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
        Gp = np.delete(G_values, col_idx, axis=1)
        g_p = Gp.shape[1]

        if Gp.shape[0] == 0:
            return [1] + [0] * g_p

        # gram is in GF(2) due to the trace inner product that simulates the symplectic product (aka commutation/anti-commutation)
        gram = full_gram ^ column_gram_contributions[col_idx]

        coeff_basis = _gf2_kernel_basis(gram.T) # c @ gram = gram.T @ c.T = 0 -> x = c @ Gp with <x, Gp[i]> = 0 for all rows j -> x orthogonal to all rows of Gp -> x in Gp perp

        if coeff_basis.shape[0] == 0:
            hull_basis = np.zeros((0, g_p), dtype=np.uint8)
        else:
            hull_basis = _gf4_value_row_basis(_gf2_gf4_value_matmul(coeff_basis, Gp)) # c @ Gp = x -> words in Gp that are orthogonal to all rows of Gp -> hull

        hull_h, hull_n = hull_basis.shape
        enumerator = [1] + [0] * g_p

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

    G_values = _gf4_values(generator_matrix)
    column_gram_contributions = _gf4_column_gram_contributions(G_values)
    full_gram = np.bitwise_xor.reduce(column_gram_contributions, axis=0, initial=0)

    invariants = []

    for col_idx in range(generator_matrix.shape[1]):
        inv = tuple(_weight_enumerator_of_hull_punctured(col_idx))
        invariants.append(inv)

    return invariants

def _partition_columns_by_invariants(invariants: list[tuple[int, ...]]) -> dict[tuple[int, ...], list[int]]:
    partition = defaultdict(list)
    for idx, inv in enumerate(invariants):
        partition[inv].append(idx)
    return {k: sorted(v) for k, v in sorted(partition.items(), key=lambda item: item[0])}

def _compute_canonical_form(matrix: np.ndarray, cells: list[list[int]]) -> np.ndarray:
    """Compute the canonical form of the GF(4) representation of the  code, using Feulner's algorithm, and return the canonical form and the corresponding permutation of the columns. Partition is used for pruning the search tree.
    """
    def _prefix_semicanonical(G: np.ndarray, i: int) -> np.ndarray:
        """Bring the first i columns of G into semi-canonical form only using GF(2) row operations."""
        _, rref_to_i , _ = _gf4_rref(G, to_col=i)
        return rref_to_i

    def _flatten(cells_: list[list[int]]) -> list[int]:
        return [c for cell in cells_ for c in cell]

    def matrix_key(M: np.ndarray) -> tuple[tuple[int, ...], ...]:
        k_m, n_m = M.shape
        symplectic = _gf4_to_symplectic(M)
        return tuple(tuple(symplectic[r, c] for r in range(k_m)) for c in range(2 * n_m))

    def prefix_key(M: np.ndarray, i: int) -> tuple[tuple[int, ...], ...]:
        return matrix_key(M[:, :i]) # lexicographic key of the first i columns of M, because python can compare tuples

    n_g = matrix.shape[1]
    best_matrix: np.ndarray | None = None
    best_full_key: tuple[tuple[int, ...], ...] | None = None
    best_prefix_keys: list[tuple[tuple[int, ...], ...]] | None = None

    cells = [sorted(cell) for cell in cells if cell]

    def _search(prefix: list[int], remaining_cells: list[list[int]], path_keys: list[tuple[tuple[int, ...], ...]]) -> None: # recursive search over the space of permutations
        nonlocal best_matrix, best_full_key, best_prefix_keys
        i = len(prefix)
        trial_perm = prefix + _flatten(remaining_cells)
        M_trial = matrix[:, trial_perm]
        M_semi = _prefix_semicanonical(M_trial, i)
        current_prefix = prefix_key(M_semi, i)

        # Prefix pruning.

        if best_matrix is not None and i > 0:
            best_prefix = best_prefix_keys[i]

            if current_prefix > best_prefix: # prune this branch, since the canonical form must be lexicographically minimal
                return

            if current_prefix < best_prefix: # update the best prefix, since we found a better one on current path
                best_matrix = None
                best_full_key = None
                best_prefix_keys = None

        if i == n_g: # we are at leaf (have a full permutation)
            full_key = matrix_key(M_semi)

            if best_full_key is None or full_key < best_full_key:
                best_full_key = full_key
                best_matrix = M_semi
                best_prefix_keys = path_keys + [current_prefix]
            return

        # Select first nonempty cell
        cell_idx = next(idx for idx, cell in enumerate(remaining_cells) if cell)
        cell = remaining_cells[cell_idx]

        for col in cell:
            new_prefix = prefix + [col]
            new_cells = [list(c) for c in remaining_cells]
            new_cells[cell_idx].remove(col)
            new_cells = [c for c in new_cells if c]
            _search(new_prefix, new_cells, path_keys + [current_prefix])

    _search([], cells, [])

    return best_matrix


def are_peq_stab_classical(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check permutation equivalence mapping a tableau to GF(4) and using algorithms for classical code equivalence. A two-layer approach is used, where the first layer uses Sendrier's Support Splitting Algorithm to partition the columns of the generator matrices into equivalence classes based on the weight enumerator of the hull of the punctured code. 
    The second layer then checks for permutation equivalence by traversing the search tree of possible permutations, and pruning branches based on the canonical form of Feulner's Algorithm.

    ! ATTENTION ! I might map the stabilizer tableau to a classical code over GF(4), but i cannot use all the operations in GF(4) since the original stabilizer code is only GF(2)-additive and i have to keep that property. Thus computing the RREF is NOT the normal RREF of GF(4)-linear codes, similar for the semicanonical form, and as the inner product I use the trace inner product.

    For each code, the following is done:
    1.) Map the stabilizer tableau to a classical code over GF(4).
    2.) Partition the columns of the matrix into equivalence classes according to the weight enumerator of Sendrier.
    3.) Canonicalize the matrices using Feulner's algorithm, and check for equivalence of the canonical forms, pruning the search tree of possible permutations.

    This algorithm should be more efficient than the brute-force algorithm, since it avoids checking all permutations, BUT it is still not efficient in the worst case. And using GF(4) instead of a binary code like in the CSS case make the computations more complex.
    """
    gf4_tableau_c1 = _symplectic_to_gf4(c1.symplectic)
    gf4_tableau_c2 = _symplectic_to_gf4(c2.symplectic)

    # Sendrier
    signatures_c1 = _compute_signatures(gf4_tableau_c1)
    signatures_c2 = _compute_signatures(gf4_tableau_c2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    for key1, key2 in zip(partition_c1.keys(), partition_c2.keys()):
        if key1 != key2:
            return False
        if len(partition_c1[key1]) != len(partition_c2[key2]):
            return False

    # Feulner
    canon_c1 = _compute_canonical_form(gf4_tableau_c1, list(partition_c1.values()))
    canon_c2 = _compute_canonical_form(gf4_tableau_c2, list(partition_c2.values()))

    return np.array_equal(canon_c1, canon_c2)
