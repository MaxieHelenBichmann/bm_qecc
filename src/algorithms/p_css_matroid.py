"""Matroid-isomorphism based permutation equivalence checking."""

from __future__ import annotations

from ..core.css_code import CSSCode

import numpy as np
import numpy.typing as npt
from collections import defaultdict

import ldpc.mod2.mod2_numpy as mod2
from pynauty import Graph, certificate

def _circuits_binary_matroid(A: npt.NDArray[np.int8]) -> list[tuple[int, ...]]:
    """
    Circuits of the binary matroid whose ground set is the columns of A.

    In the case of a binary matrix A as matroid, the circuits are the minimal non-empty supports of vectors in the kernel of A.

    Returns tuples of column indices.
    """

    A = (np.asarray(A) & 1).astype(np.uint8)
    K = mod2.nullspace(A)
 
    if hasattr(K, "toarray"):
        K = K.toarray()

    K = (np.asarray(K) & 1).astype(np.uint8)

    if K.size == 0:
        return []

    if K.shape[1] != A.shape[1]:
        raise ValueError("Kernel basis must have the same number of columns as the input matrix.")

    k, n_cols = K.shape
    candidates: list[set[int]] = []

    # all nonzero combinations of kernel basis rows
    for mask in range(1, 1 << k):
        x = np.zeros(n_cols, dtype=np.uint8)

        for i in range(k):
            if (mask >> i) & 1:
                x ^= K[i]

        support = set(np.flatnonzero(x))
        if support:
            candidates.append(support)

    # inclusion-minimal supports
    candidates.sort(key=len)
    circuits: list[set[int]] = []

    for support in candidates:
        if not any(c <= support for c in circuits):
            circuits.append(support)

    return [tuple(sorted(c)) for c in circuits]

def _graph_from_circuits(n: int, circuits_hx: list[tuple[int, ...]], circuits_hz: list[tuple[int, ...]]) -> Graph:

    adj = defaultdict(list)

    def _add_edges_from_circuits(circuits: list[tuple[int, ...]], offset: int) -> None:
        for i, circuit in enumerate(circuits):
            for q in circuit:
                adj[q].append(offset + i)
                adj[offset + i].append(q)

    hx_offset = n
    hz_offset = n + len(circuits_hx)

    _add_edges_from_circuits(circuits_hx, hx_offset)
    _add_edges_from_circuits(circuits_hz, hz_offset)

    return Graph(
        number_of_vertices=n + len(circuits_hx) + len(circuits_hz),
        directed=False,
        adjacency_dict=adj,
        vertex_coloring=[
            set(range(n)),
            set(range(hx_offset, hx_offset + len(circuits_hx))),
            set(range(hz_offset, hz_offset + len(circuits_hz)))
        ]
    )

def are_peq_css_matroid(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence by checking for isomorphism of the associated pair of binary matroids.
    
    For each code, the following is done:
    1.) Construct a binary matroid M = (E, I) from the Hx and Hz of the CSS code, by treating the columns of Hx and Hz as the ground elements E and using linear dependence of the columns for I. 
    2.) Extract the circuits of the matroids, which are the minimal non-empty dependent sets of columns of Hx and Hz.
    3.) Construct a graph G = (V, K) from the circuits, where V represents the ground set (color 1), circuits of Hx (color 2) and circuits of Hz (color 3). The circuits are connected to the ground elements they contain, and there are no connections between circuits.

    4.) Check for isomorphism of the resulting graphs for the two codes via canonical forms.

    This algorithm should be more efficient than the brute-force algorithm, since the number of circuits is typically much smaller than the number of permutations, BUT it is still not efficient in general since the number of circuits can be exponential in the size of the codes, and graph isomorphism is not known to be in P.
    """
    circuits_c1_hx = _circuits_binary_matroid(c1.Hx)
    circuits_c1_hz = _circuits_binary_matroid(c1.Hz)

    graph_c1 = _graph_from_circuits(c1.n, circuits_c1_hx, circuits_c1_hz)

    circuits_c2_hx = _circuits_binary_matroid(c2.Hx)
    circuits_c2_hz = _circuits_binary_matroid(c2.Hz)

    graph_c2 = _graph_from_circuits(c2.n, circuits_c2_hx, circuits_c2_hz)

    return certificate(graph_c1) == certificate(graph_c2)