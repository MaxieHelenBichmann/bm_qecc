"""Classical code equivalence based permutation equivalence checking.

References for this algorithm: 
- Nicolas Sendrier: Finding the Permutation Between Equivalent Linear Codes: The Support Splitting Algorithm
- Thomas Feulner: The Automorphism Groups of Linear Codes and Canonical Representatives of Their Semilinear Isometry Classes
"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterator

import hashlib

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ...core.css_code import CSSCode

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

def _compute_signatures(G1: np.ndarray, G2: np.ndarray) -> list[int]:
    """Compute the combined Sendrier's invariant of the weight enumerator of the hull of the punctured code of each column of the CSS code.
    """
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

def _compute_canonical_form(G: np.ndarray, cells: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    """Compute the canonical form of Hx of the CSS code, using Feulner's algorithm, and return the canonical form and the corresponding permutation of the columns. Partition is used for pruning the search tree.
    """
    def _prefix_semicanonical(G: np.ndarray, i: int) -> np.ndarray:
        """Bring the first i columns of G into semi-canonical form only using row operations."""
        M = np.array(G, dtype=np.int8) & 1
        k_m = M.shape[0]
        if i == 0:
            return M
        pivot_row = 0

        for c in range(i):
            if _rank(M[:, : c + 1]) > _rank(M[:, :c]): # independent column, bring it to semi-canonical form (dependent case not important for binary matrices)
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
        return tuple(map(tuple, M.T.tolist())) # lexicographic key of the rows of M, because python can compare tuples

    def prefix_key(M: np.ndarray, i: int) -> tuple[tuple[int, ...], ...]:
        return matrix_key(M[:, :i]) # lexicographic key of the first i columns of M, because python can compare tuples


    G = np.array(G, dtype=np.int8, copy=True) & 1
    n_g = G.shape[1]
    best_matrix: np.ndarray | None = None
    best_full_key: tuple[tuple[int, ...], ...] | None = None
    best_perms: list[list[int]] = []

    cells = [sorted(cell) for cell in cells if cell]

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
                best_perms.append(prefix)
            return

        # Select first nonempty cell
        cell_idx = next(idx for idx, cell in enumerate(remaining_cells) if cell)
        cell = remaining_cells[cell_idx]

        for col in cell:
            new_prefix = prefix + [col]
            new_cells = [list(c) for c in remaining_cells]
            new_cells[cell_idx].remove(col)
            new_cells = [c for c in new_cells if c]
            _search(new_prefix, new_cells)

    _search([], cells)

    return best_matrix, best_perms

def _iter_permutations(canon1: np.ndarray, canon2: np.ndarray, can_to_g1: list[list[int]], can_to_g2: list[list[int]]) -> Iterator[tuple[int, ...]]:
    def _inverse_perm(p):
        inv = [None] * len(p)
        for i, x in enumerate(p):
            inv[x] = i
        return tuple(inv)

    def _compose(p, q):
        return tuple(p[i] for i in q)

    if not np.array_equal(np.asarray(canon1, dtype=np.int8) & 1, np.asarray(canon2, dtype=np.int8) & 1):
        return

    # find all permutations that map g1 to g2
    g2_to_can =  [_inverse_perm(p2) for p2 in can_to_g2]

    for p1 in can_to_g1:
        p1 = tuple(p1)
        for p2 in g2_to_can:
            yield _compose(p1, p2)

def are_peq_css_classical(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence using algorithms for classical code equivalence. A two-layer approach is used, where the first layer uses Sendrier's Support Splitting Algorithm to partition the columns of the generator matrices into equivalence classes based on the weight enumerator of the hull of the punctured code. 
    The second layer then checks for permutation equivalence by traversing the search tree of possible permutations, and pruning branches based on the canonical form of Feulner's Algorithm.
    
    For each code, the following is done:
    1.) Partition the columns of Hx and Hz into equivalence classes according to the weight enumerator of Sendrier (we risk a complex computation O(2^k) for both Hx and Hz to have a better chance of a fine-grained partition).
    2.) Canonicalize the generator matrices of Gx1 anf Gx2 using Feulner's algorithm, and check for equivalence of the canonical forms, pruning the search tree of possible permutations. 
    3.) Check if the found permutation is valid for both Hx and Hz.

    This algorithm should be more efficient than the brute-force algorithm, since it avoids checking all permutations, BUT it is still not efficient in the worst case.
    """
    if c1.n != c2.n:
        return False

    hx_rank = _rank(c1.Hx)
    hz_rank = _rank(c1.Hz)

    if hx_rank != _rank(c2.Hx) or hz_rank != _rank(c2.Hz):
        return False

    # Sendrier

    def _generator_matrix_from_parity_check(H: np.ndarray, n: int) -> np.ndarray:
        if H.size == 0 or H.shape[0] == 0:
            return np.eye(n, dtype=np.uint8)
        return _kernel_basis(H)

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

    # Feulner
    canon_c1, perm1 = _compute_canonical_form(Gx1, list(partition_c1.values()))
    canon_c2, perm2 = _compute_canonical_form(Gx2, list(partition_c2.values()))

    for perm in _iter_permutations(canon_c1, canon_c2, perm1, perm2):
        if hx_rank == _rank(np.vstack([c2.Hx, c1.Hx[:, perm]])) and hz_rank == _rank(np.vstack([c2.Hz, c1.Hz[:, perm]])):
            return True

    return False
