"""Graph-isomorphism based permutation equivalence checking.

References for this algorithm: 
- Nicolas Sendrier: Finding the Permutation Between Equivalent Linear Codes: The Support Splitting Algorithm
"""

from __future__ import annotations

import hashlib
from collections import deque, defaultdict

import numpy as np

import ldpc.mod2.mod2_numpy as mod2
from pynauty import Graph, certificate, canon_label, autgrp

from ...core.css_code import CSSCode

def _compose(p, q):
    return tuple(p[i] for i in q)

def _compute_invariant_a(code: CSSCode) -> list[int]:
    """Compute combined invariant of (non)zero columns of Hx and Hz for each column of the CSS code.

    This is a very simple and cheap invariant, but cannot distinguish a lot of columns.
    """
    hx_nonzero = np.any(code.Hx != 0, axis=0).astype(int)
    hz_nonzero = np.any(code.Hz != 0, axis=0).astype(int)
    return (hx_nonzero + 2 * hz_nonzero).tolist()

def _compute_invariant_b(code: CSSCode) -> list[int]:
    """Compute the combined Sendrier's invariant of the weight enumerator of the hull of the punctured code of each column of Hx and Hz of the CSS code.
    
    This is a more expensive invariant, as the hull can be large, but it is also more powerful, as it can distinguish more columns.
    """
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

    def _weight_enumerator_of_hull_punctured(G: np.ndarray, col_idx: int) -> list[int]:
        Gp = np.delete(G, col_idx, axis=1) & 1
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
    Gx = _generator_matrix_from_parity_check(code.Hx, code.n)
    Gz = _generator_matrix_from_parity_check(code.Hz, code.n)

    for col_idx in range(code.n):
        inv_hx = _weight_enumerator_of_hull_punctured(Gx, col_idx)
        inv_hz = _weight_enumerator_of_hull_punctured(Gz, col_idx)

        invariants.append(_combine_invariants(inv_hx, inv_hz))

    return invariants

def _graph_from_invariants(n: int, invariants: list[list[int]]) -> Graph:

    adj = defaultdict(list)
    offset = n
    coloring = []

    for invariant in invariants:
        inv_value_to_index: dict[int, int] = {}
        for i , value in enumerate(invariant):
            idx = inv_value_to_index.setdefault(value, len(inv_value_to_index))

            adj[offset + idx].append(i)
            adj[i].append(offset + idx)

        coloring.append(set(range(offset, offset + len(inv_value_to_index))))
        offset += len(inv_value_to_index)

    return Graph(
        number_of_vertices= offset,
        directed=False,
        adjacency_dict=adj,
        vertex_coloring=[set(range(n))] + coloring
    )

def _extract_aut_and_permutation(g1: Graph, g2: Graph, n: int) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    def _inverse_perm(p):
        inv = [None] * len(p)
        for i, x in enumerate(p):
            inv[x] = i
        return inv

    if certificate(g1) != certificate(g2):
        return [], ()

    # get one isomorphism from g1 to g2 using the canonical labeling
    can_to_g1 = canon_label(g1)
    can_to_g2 = canon_label(g2)
    g1_to_can = _inverse_perm(can_to_g1)

    phi = _compose(can_to_g2, g1_to_can)

    # get the full automorphism group of g2
    generators, _, _, _, _ = autgrp(g2)
    gens = [tuple(gen) for gen in generators] + [tuple(_inverse_perm(gen)) for gen in generators]

    identity = tuple(range(len(phi)))
    aut_g2 = {identity}
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        for gen in gens:
            nxt = _compose(gen, current)
            if nxt not in aut_g2:
                aut_g2.add(nxt)
                queue.append(nxt)

    return aut_g2, phi

def are_peq_css_graph_iso(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence by checking for isomorphism of the associated graphs constructed from the codes using some invariants.
    
    For each code, the following is done:
    1.) Compute some invariants of the columns of the parity-check matrices Hx and Hz.
    2.) Construct a graph G = (V, E) from the invariants, where the vertices are the set of columns (color 1) and the sets of all values each invariant takes colored according to the invariants (color 2...l). There are edges between column-vertices and invariant-value-vertices if the column has that invariant value, and there are no edges between vertices of the same color.
    3.) Check for isomorphism of the resulting graphs for the two codes and extract candidate permutations between the column-vertices.
    4.) Check the candidate permutations for actual permutation equivalence of the codes.

    This algorithm should be more efficient than the brute-force algorithm, since the number of valid permutations (under the given invariants) that have to be checked is typically much smaller than the full number of permutations. BUT the number of valid permutations can still be factorial in the worst case, in the case of highly symmetric codes or poor invariants, and graph isomorphism is not known to be in P. Issues with good invariants is that they can be costly to compute.
    """
    # TODO: add more invariants
    invariants_c1 = [_compute_invariant_a(c1), _compute_invariant_b(c1)]
    invariants_c2 = [_compute_invariant_a(c2), _compute_invariant_b(c2)]

    # TODO: use the fact that invariants are the actual same for permutation, aka use one color for each invariant value (and not only invariant-patterns/partitions), but keep the coloring consistent between the two graphs
    graph_c1 = _graph_from_invariants(c1.n, invariants_c1)
    graph_c2 = _graph_from_invariants(c2.n, invariants_c2)

    aut_g2, phi = _extract_aut_and_permutation(graph_c1, graph_c2, c1.n)

    def _rank(A: np.ndarray) -> int:
        if A.shape[0] == 0 or A.shape[1] == 0:
            return 0
        return mod2.rank(A)
    
    hx_rank = _rank(c1.Hx)
    hz_rank = _rank(c1.Hz)

    if hx_rank != _rank(c2.Hx) or hz_rank != _rank(c2.Hz):
        return False
    
    seen = set()

    for aut in aut_g2:
        # apply the automorphisms of g2 to phi to get all isomorphisms from g1 to g2
        # isomorphisms(g1, g2) = { α ∘ φ | α ∈ Aut(g2) } with φ: g1 -> g2
        full_perm = _compose(aut, phi)

        # extract the permutations of only the qubit vertices from the isomorphisms (permutations map columns of c2 to columns of c1)
        qubit_perm = tuple(full_perm[i] for i in range(c1.n))

        if qubit_perm in seen:
            continue

        seen.add(qubit_perm)

        if hx_rank and hx_rank != mod2.rank(np.vstack([c1.Hx, c2.Hx[:, qubit_perm]])):
            continue
        if hz_rank and hz_rank != mod2.rank(np.vstack([c1.Hz, c2.Hz[:, qubit_perm]])):
            continue
        return True

    return False
