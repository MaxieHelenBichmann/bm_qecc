"""Classical code equivalence based permutation equivalence checking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from itertools import product
import ldpc.mod2.mod2_numpy as mod2

from ..core.stabilizer_code import StabilizerCode

@dataclass(frozen=True)
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
        table = {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,

            (1, 0): 0,
            (1, 1): 1,
            (1, 2): 2,
            (1, 3): 3,

            (2, 0): 0,
            (2, 1): 2,
            (2, 2): 3,  # w * w = w^2 = w + 1 = w_bar
            (2, 3): 1,  # w * w_bar = w * (w + 1) = w^2 + w = (w + 1) + w = 1

            (3, 0): 0,
            (3, 1): 3,
            (3, 2): 1,  # w_bar * w = (w + 1) * w = w^2 + w = (w + 1) + w = 1
            (3, 3): 2,  # w_bar * w_bar = (w + 1) * (w + 1) = w^2 + 1 = (w + 1) + 1 = w
        }
        return GF4(table[(self.value, other.value)])

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
        return self ** 2

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
    # I -> 0, X -> 1, Z -> w, Y -> w_bar
    n = tableau.shape[1] // 2
    r = tableau.shape[0]
    gf4_matrix = np.zeros([[ZERO for _ in range(n)] for _ in range(r)], dtype=object)
    for q in range(n):
        for i in range(r):
            x = tableau[i, q]
            z = tableau[i, q + n]
            if x == 0 and z == 0:
                gf4_matrix[i, q] = ZERO
            elif x == 1 and z == 0:
                gf4_matrix[i, q] = ONE
            elif x == 0 and z == 1:
                gf4_matrix[i, q] = W
            elif x == 1 and z == 1:
                gf4_matrix[i, q] = W_BAR

    return gf4_matrix

def _gf4_rank_rref(matrix: np.ndarray) -> tuple[int, np.ndarray]:
    # matrix has GF(4) entries
    if matrix.shape[0] == 0:
        return 0, matrix
    matrix = matrix.copy()
    m, n = matrix.shape
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if not matrix[r, col].is_zero():
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            matrix[[row, pivot]] = matrix[[pivot, row]]

        pivot_value = matrix[row, col]
        matrix[row, :] = [x / pivot_value for x in matrix[row, :]]

        for r in range(m):
            if r != row and not matrix[r, col].is_zero():
                factor = matrix[r, col]

                matrix[r, :] = [
                    matrix[r, c] - factor * matrix[row, c]
                    for c in range(n)
                ]
        rank += 1
        row += 1
        if row == m:
            break
    return rank, matrix

def _gf4_rank(matrix: np.ndarray) -> int:
    return _gf4_rank_rref(matrix)[0]

def _gf4_rref(matrix: np.ndarray) -> np.ndarray:
    return _gf4_rank_rref(matrix)[1]

def _gf4_kernel_basis(A: np.ndarray, n: int) -> list[np.ndarray]:
    # basis of the set of vectors x such that A @ x = 0, with operations done in GF(4)
    if A.size == 0 or A.shape[0] == 0:
        return [np.array([ZERO] * i + [ONE] + [ZERO] * (n - i - 1), dtype=object) for i in range(n)]
    rref = _gf4_rref(A)
    m, n = rref.shape
    pivot_cols = []
    for r in range(m):
        for c in range(n):
            if not rref[r, c].is_zero():
                pivot_cols.append(c)
                break
    free_cols = [c for c in range(n) if c not in pivot_cols]

    basis = []
    for free_col in free_cols:
        vec = np.array([ZERO for _ in range(n)], dtype=object)
        vec[free_col] = ONE
        for i, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = -rref[i, free_col]
        basis.append(vec)

    return basis

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

def _gf4_row_basis(M: np.ndarray) -> list[np.ndarray]:
    # basis (base vectors as columns) of the row space of M, with operations done in GF(4)
    if M.shape[0] == 0:
        return [np.array([ZERO] * M.shape[1], dtype=object)]
    rref = _gf4_rref(M)
    pivot_rows = []
    for i in range(rref.shape[0]):
        if not all(rref[i, c].is_zero() for c in range(rref.shape[1])):
            pivot_rows.append(i)
    return [rref[i, :] for i in pivot_rows]

def _gf4_trace_inner_product(a: np.ndarray, b: np.ndarray) -> GF4:
    sum = ZERO
    for i in range(len(a)):
        sum += a[i].conjugate() * b[i] + a[i] * b[i].conjugate()
    return sum

def _gf4_gram(G: np.ndarray) -> np.ndarray:
    # gram matrix of G and G in GF(4) but stabilizer semantics
    k = G.shape[0]

    gram = np.zeros((k, k), dtype=np.uint8)

    for i in range(k):
        for j in range(k):
            gram[i, j] = _gf4_trace_inner_product(G[i], G[j]).value

    return gram

def _gf4_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # gram matrix of A and B in GF(4)
    m, ra = A.shape
    rb, n = B.shape
    if ra != rb:
        raise ValueError("Incompatible shapes for matrix multiplication.")
    
    C = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            s = ZERO
            for t in range(ra):
                s += A[i, t] * B[t, j]
            C[i, j] = s
    return C

def _gf2_gf4_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # gram matrix of A in GF(2) and B in GF(4)
    m, ra = A.shape
    rb, n = B.shape
    if ra != rb:
        raise ValueError("Incompatible shapes for matrix multiplication.")
    
    C = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            s = ZERO
            for t in range(ra):
                s += GF4(A[i, t]) * B[t, j]
            C[i, j] = s
    return C

def _compute_signatures(generator_matrix: np.ndarray) -> list[int]:
    """Compute the combined Sendrier's invariant of the weight enumerator of the hull of the punctured code of each column of the code.
    """
    def _weight_enumerator_of_hull_punctured(G: np.ndarray, col_idx: int) -> list[int]:
        Gp = np.delete(G, col_idx, axis=1)
        g_p = Gp.shape[1]

        if Gp.shape[0] == 0:
            return [1] + [0] * g_p

        # gram is in GF(2) due to the trace inner product that simulates the symplectic product (aka commutation/anti-commutation)
        gram = _gf4_gram(Gp)

        coeff_basis = _gf2_kernel_basis(gram.T) # c @ gram = gram.T @ c.T = 0 -> x = c @ Gp with <x, Gp[i]> = 0 for all rows j -> x orthogonal to all rows of Gp -> x in Gp perp
            
        if coeff_basis.shape[0] == 0:
            hull_basis = np.zeros((0, g_p), dtype=object)
        else:
            hull_basis = np.array(_gf4_row_basis(_gf2_gf4_matmul(coeff_basis, Gp)), dtype=object) # c @ Gp = x -> words in Gp that are orthogonal to all rows of Gp -> hull

        hull_h, hull_n = hull_basis.shape
        enumerator = [1] + [0] * g_p

        word = np.array([ZERO for _ in range(hull_n)], dtype=object)
        previous_gray = 0

        for t in range(1, 1 << hull_h):
            gray = t ^ (t >> 1)
            changed = gray ^ previous_gray
            row_idx = changed.bit_length() - 1

            # GF(2)-additive
            for j in range(hull_n):
                word[j] += hull_basis[row_idx, j]

            wt = sum(not x.is_zero() for x in word)
            enumerator[wt] += 1
            
            previous_gray = gray

        return enumerator
    
    def _hash_enumerator(enum: list[int]) -> int:
        payload = (",".join(map(str, enum))).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")

    invariants = []

    for col_idx in range(generator_matrix.shape[1]):
        inv = _hash_enumerator(_weight_enumerator_of_hull_punctured(generator_matrix, col_idx))
        invariants.append(inv)

    return invariants

def _partition_columns_by_invariants(invariants: list[int]) -> dict[int, list[int]]:
    partition = {}
    for idx, inv in enumerate(invariants):
        if inv not in partition:
            partition[inv] = []
        partition[inv].append(idx)
    return {k: sorted(v) for k, v in sorted(partition.items(), key=lambda item: item[0])}

def _compute_canonical_form(G: np.ndarray, cells: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    """Compute the canonical form of Hx of the CSS code, using Feulner's algorithm, and return the canonical form and the corresponding permutation of the columns. Partition is used for pruning the search tree.
    """
    def _prefix_semicanonical(G: np.ndarray, i: int) -> np.ndarray:
        """Bring the first i columns of G into semi-canonical form only using row operations."""
        M = np.array(G, dtype=np.int8, copy=True) & 1
        k_m, n_m = M.shape
        if i == 0:
            return M
        pivot_row = 0

        for c in range(i):
            if _gf4_rank(M[:, : c + 1]) > _gf4_rank(M[:, :c]): # independent column, bring it to semi-canonical form (dependent case not important for binary matrices)
                if pivot_row >= k_m:
                    continue

                pivot_candidates = np.flatnonzero(M[pivot_row:, c])

                if pivot_candidates.size == 0:
                    continue

                pivot = pivot_row + int(pivot_candidates[0])

                # swap row with potential pivot up
                if pivot != pivot_row:
                    M[[pivot_row, pivot], :] = M[[pivot, pivot_row], :]

                # make pivot column a unit vector
                for r in range(k_m):
                    if r != pivot_row and M[r, c]:
                        M[r, :] ^= M[pivot_row, :]

                pivot_row += 1

        return M

    def _flatten(cells_: list[list[int]]) -> list[int]:
        return [c for cell in cells_ for c in cell]

    def matrix_key(M: np.ndarray) -> tuple[tuple[int, ...], ...]:
        M = np.asarray(M, dtype=np.int8) & 1
        k_m, n_m = M.shape
        return tuple(tuple(int(M[r, c]) for r in range(k_m)) for c in range(n_m))

    def prefix_key(M: np.ndarray, i: int) -> tuple[tuple[int, ...], ...]:
        return matrix_key(M[:, :i]) # lexicographic key of the first i columns of M, because python can compare tuples


    G = np.array(G, dtype=np.int8, copy=True) & 1
    k_g, n_g = G.shape
    best_matrix: np.ndarray | None = None
    best_full_key: tuple[tuple[int, ...], ...] | None = None
    best_perms: list[list[list[int]]] = []

    def _search(prefix: list[int], remaining_cells: list[list[int]]) -> None: # recursive search over the space of permutations
        nonlocal best_matrix, best_full_key, best_perms
        i = len(prefix)
        trial_perm = prefix + _flatten(remaining_cells)
        M_trial = G[:, trial_perm]
        M_semi = _prefix_semicanonical(M_trial, i)

        # Prefix pruning.

        if best_matrix is not None and i > 0:
            current_prefix = prefix_key(M_semi, i)
            best_prefix = prefix_key(best_matrix, i)

            if current_prefix > best_prefix: # prune this branch, since the canonical form must be lexicographically minimal
                return 

            if current_prefix < best_prefix: # update the best prefix, since we found a better one on current path
                best_matrix = None 
                best_full_key = None
                best_perms = []

        if i == n_g: # we are at leaf (have a full permutation)
            full_key = matrix_key(M_semi)

            if best_full_key is None or full_key < best_full_key:
                best_full_key = full_key
                best_matrix = M_semi
                best_perms = [prefix.copy()]

            elif full_key == best_full_key:
                best_perms.append(prefix.copy())
            return

        # Select first nonempty cell
        cell_idx = next(idx for idx, cell in enumerate(remaining_cells) if cell)
        cell = remaining_cells[cell_idx]

        for col in sorted(cell):
            new_prefix = prefix + [col]
            new_cells = [list(c) for c in remaining_cells]
            new_cells[cell_idx].remove(col)
            new_cells = [c for c in new_cells if c]
            _search(new_prefix, new_cells)

    _search([], cells)

    return best_matrix, best_perms


def are_peq_stab_classical(c1: StabilizerCode, c2: StabilizerCode) -> bool:  
    """Check permutation equivalence mapping a tableau to GF(4) and using algorithms for classical code equivalence. A two-layer approach is used, where the first layer uses Sendrier's Support Splitting Algorithm to partition the columns of the generator matrices into equivalence classes based on the weight enumerator of the hull of the punctured code. 
    The second layer then checks for permutation equivalence by traversing the search tree of possible permutations, and pruning branches based on the canonical form of Feulner's Algorithm.
    
    For each code, the following is done:
    1.) Map the stabilizer tableau to a classical code over GF(4).
    2.) Partition the columns of the generator matrix into equivalence classes according to the weight enumerator of Sendrier (we risk a complex computation O(2^k) for both Hx and Hz to have a better chance of a fine-grained partition)
    3.) Canonicalize the generator matrices using Feulner's algorithm, and check for equivalence of the canonical forms, pruning the search tree of possible permutations. 

    This algorithm should be more efficient than the brute-force algorithm, since it avoids checking all permutations, BUT it is still not efficient in the worst case. And using GF(4) instead of a binary code like in the CSS case make the computations more complex.
    """  
    gf4_tableau_c1 = _symplectic_to_gf4(c1.symplectic)
    gf4_tableau_c2 = _symplectic_to_gf4(c2.symplectic)

    # Sendrier
    G1 = np.array(_gf4_kernel_basis(gf4_tableau_c1, c1.n), dtype=object) 
    G2 = np.array(_gf4_kernel_basis(gf4_tableau_c2, c2.n), dtype=object) 

    signatures_c1 = _compute_signatures(G1)
    signatures_c2 = _compute_signatures(G2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    for key1, key2 in zip(partition_c1.keys(), partition_c2.keys()):
        if key1 != key2:
            return False
        if len(partition_c1[key1]) != len(partition_c2[key2]):
            return False

    # Feulner
    canon_c1 = _compute_canonical_form(G1, list(partition_c1.values()))
    canon_c2 = _compute_canonical_form(G2, list(partition_c2.values()))

    return np.array_equal(canon_c1, canon_c2) 