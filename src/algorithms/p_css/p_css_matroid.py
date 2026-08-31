"""Matroid-isomorphism based permutation equivalence checking."""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import numpy.typing as npt

import ldpc.mod2.mod2_numpy as mod2
from pynauty import Graph, certificate

from ...core.css_code import CSSCode


def _circuits_binary_matroid(A: npt.NDArray[np.int8]) -> list[int]:
    """
    Circuits of the binary matroid whose ground set is the columns of A.
    In the case of a binary matrix A as matroid, the circuits are the minimal non-empty supports of vectors in the kernel of A.
    Returns bitmasks of column indices.
    """
    def _row_support_as_mask(row: npt.NDArray[np.uint8]) -> int:
        support = 0
        for col in np.flatnonzero(row):
            support |= 1 << int(col)
        return support
    
    # find all linear dependencies of the columns of A by finding all vectors in the kernel of A, bc Ax=0 and x interpreted as a coefficient vector stating whether a certain column is included in the linear dependence or not

    A = (np.asarray(A) & 1).astype(np.uint8, copy=False)
    K = mod2.nullspace(A)

    if hasattr(K, "toarray"):
        K = K.toarray()

    K = (np.asarray(K) & 1).astype(np.uint8, copy=False)

    if K.size == 0:
        return []

    if K.shape[1] != A.shape[1]:
        raise ValueError("Kernel basis must have the same number of columns as the input matrix.")

    k, _ = K.shape
    row_supports = [_row_support_as_mask(row) for row in K]
    circuits_by_size: list[list[int]] = [[] for _ in range(A.shape[1] + 1)]

    # All nonzero combinations of kernel basis rows. Gray-code order changes
    # one row at a time, avoiding a fresh vector and k row checks per mask

    # Streaming Filter, aka dont materialize all candidate supports and sort by size first, but keep track of the current minimal supports and only check against those

    # Keep only the current inclusion-minimal supports instead of materializing
    # all candidate supports by size first. Size buckets avoid scanning circuits
    # that cannot be subsets or supersets of the current support.
    support = 0
    previous_gray = 0
    for mask in range(1, 1 << k):
        gray = mask ^ (mask >> 1)
        changed = gray ^ previous_gray
        support ^= row_supports[changed.bit_length() - 1]
        previous_gray = gray

        if not support:
            continue

        support_size = support.bit_count()

        if any(
            (circuit & support) == circuit
            for size in range(1, support_size + 1)
            for circuit in circuits_by_size[size]
        ):
            continue

        for size in range(support_size + 1, len(circuits_by_size)):
            if not circuits_by_size[size]:
                continue
            circuits_by_size[size] = [
                circuit
                for circuit in circuits_by_size[size]
                if (support & circuit) != support
            ]

        circuits_by_size[support_size].append(support)

    return [
        circuit
        for circuits in circuits_by_size
        for circuit in sorted(circuits)
    ]


def _graph_from_circuits(n: int, circuits_hx: list[int], circuits_hz: list[int]) -> Graph:

    adj = defaultdict(list)

    def _iter_mask_bits(mask: int):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def _add_edges_from_circuits(circuits: list[int], offset: int) -> None:
        for i, circuit in enumerate(circuits):
            circuit_vertex = offset + i
            for q in _iter_mask_bits(circuit):
                adj[q].append(circuit_vertex)
                adj[circuit_vertex].append(q)

    n_hx = len(circuits_hx)
    n_hz = len(circuits_hz)

    hx_offset = n
    hz_offset = n + n_hx
    hx_anchor = n + n_hx + n_hz
    hz_anchor = hx_anchor + 1

    _add_edges_from_circuits(circuits_hx, hx_offset)
    _add_edges_from_circuits(circuits_hz, hz_offset)

    return Graph(
        number_of_vertices=n + n_hx + n_hz + 2,
        directed=False,
        adjacency_dict=adj,
        vertex_coloring=[
            set(range(n)),
            set(range(hx_offset, hx_offset + n_hx)) | {hx_anchor},
            set(range(hz_offset, hz_offset + n_hz)) | {hz_anchor},
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

    len_circuits_c1_hx = len(circuits_c1_hx)
    len_circuits_c1_hz = len(circuits_c1_hz)

    del circuits_c1_hx
    del circuits_c1_hz

    cert_c1 = certificate(graph_c1)

    del graph_c1

    circuits_c2_hx = _circuits_binary_matroid(c2.Hx)
    if len_circuits_c1_hx != len(circuits_c2_hx):
        return False
    
    circuits_c2_hz = _circuits_binary_matroid(c2.Hz)
    if len_circuits_c1_hz != len(circuits_c2_hz):
        return False
    
    graph_c2 = _graph_from_circuits(c2.n, circuits_c2_hx, circuits_c2_hz)

    del circuits_c2_hx
    del circuits_c2_hz

    return cert_c1 == certificate(graph_c2)
