"""Graph-isomorphism based permutation equivalence checking.

References for this algorithm:
- Andrew Cross, Drew Vandeth: Small Binary Stabilizer Subsystem Codes
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from pynauty import Graph, certificate

from ...core.stabilizer_code import StabilizerCode

def _graph_from_code(code: StabilizerCode) -> Graph:
    r = code.n - code.k
    adj_dict = defaultdict(list)

    for mask in range(0, 1 << r):
        group_element_vertex = 3 * code.n + mask
        x = np.zeros(2*code.n, dtype=np.int8)

        for i in range(r):
            if (mask >> i) & 1:
                x ^= code.symplectic[i]

        x_part = x[:code.n]
        z_part = x[code.n:]

        for q in range(code.n):
            if x_part[q] == 1 and z_part[q] == 0: # X contribution
                adj_dict[3 * q].append(group_element_vertex)
                adj_dict[group_element_vertex].append(3 * q)

            elif x_part[q] == 0 and z_part[q] == 1: # Z contribution
                adj_dict[3 * q + 1].append(group_element_vertex)
                adj_dict[group_element_vertex].append(3 * q + 1)

            elif x_part[q] == 1 and z_part[q] == 1: # Y contribution
                adj_dict[3 * q + 2].append(group_element_vertex)
                adj_dict[group_element_vertex].append(3 * q + 2)

    return Graph(number_of_vertices=code.n * 3 + 2 ** r,
                 directed=False,
                 vertex_coloring=[set(range(3 * code.n)), set(range(3 * code.n, 3 * code.n + 2 ** r))],
                 adjacency_dict=adj_dict)

def are_lceq_graph_iso(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check Local-Clifford equivalence by reducing to graph isomorphism.

    For each code, the following is done:
    1.) Convert the stabilizer code into a colored graph G = (V, E) enumerating all elements in the stabilizer group, thus:
    V = {x_i, y_i, z_i | i = 1, ..., n} union { S_i | S_i ∈ S }
    E = { {S_i, a_j} ) | S_i has a on qubit j with a in {x, j, z} } union { {x_i, y_i}, {x_i, z_i}, {y_i, z_i} | i = 1, ..., n }

    2.) Check if the resulting graphs are isomorphic.

    By creating a node for each element of the stabilizer group, this being independent from the generator basis, and by splitting the X, Y and Z contributions with the possibility of rotating the Paulis, we create a graph of an exponential size, which is not efficient.
    """
    graph_1 = _graph_from_code(c1)
    graph_2 = _graph_from_code(c2)

    return certificate(graph_1) == certificate(graph_2)
