"""Graph-isomorphism based permutation equivalence checking."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from pynauty import Graph, certificate

from ..core.stabilizer_code import StabilizerCode

def _graph_from_code(code: StabilizerCode) -> Graph:
    r = code.n - code.k
    adj_dict = defaultdict(list)

    # ISSUE: pynauty does no support colored edges, so we need to encode the edge colors into the vertex colors by splitting edges and introducing auxiliary vertices

    z_edges = set()
    x_edges = set()

    stabilizer_group_size = 2 ** r
    edge_id = code.n + stabilizer_group_size

    for mask in range(0, 1 << r):
        x = np.zeros(2*code.n, dtype=np.int8)

        for i in range(r):
            if (mask >> i) & 1:
                x ^= code.symplectic[i]

        x_part = x[:code.n]
        z_part = x[code.n:]

        for idx in np.flatnonzero(x_part):
            # add new x edge
            x_edges.add(edge_id)

            # edge: qubit --- x-edge
            adj_dict[idx].append(edge_id)
            adj_dict[edge_id].append(idx)

            # edge: x-edge --- stabilizer group element
            adj_dict[edge_id].append(code.n + stabilizer_group_size)
            adj_dict[code.n + stabilizer_group_size].append(edge_id)
            edge_id += 1

        for idx in np.flatnonzero(z_part):
            # add new z edge
            z_edges.add(edge_id)

            # edge: qubit --- z-edge
            adj_dict[idx].append(edge_id)
            adj_dict[edge_id].append(idx)

            # edge: z-edge --- stabilizer group element
            adj_dict[edge_id].append(code.n + stabilizer_group_size)
            adj_dict[code.n + stabilizer_group_size].append(edge_id)
            edge_id += 1


    return Graph(number_of_vertices=code.n + stabilizer_group_size + len(z_edges) + len(x_edges),
                 directed=False,
                 vertex_coloring=[set(range(code.n))] + [set(range(code.n, code.n + stabilizer_group_size))] + [z_edges] + [x_edges],
                 adjacency_dict=adj_dict)

def are_peq_stab_graph_iso(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check permutation equivalence by reducing to graph isomorphism.

    For each code, the following is done:
    1.) Convert the stabilizer code into a colored graph G = (V, E) enumerating all elements in the stabilizer group, thus:
    V = {1,...,n} union { S_i | S_i ∈ S } with color(S_i) 
    E = { (j, S_i, red) | S_i has X on qubit j } union { (j, S_i, green) | S_i has Z on qubit j }

    2.) Check if the resulting graphs are isomorphic.

    By creating a node for each element of the stabilizer group, this being independent from the generator basis, and by splitting the X and Z edges, we create a graph of an exponential size, which is not efficient.
    """
    graph_1 = _graph_from_code(c1)
    graph_2 = _graph_from_code(c2)

    return certificate(graph_1) == certificate(graph_2)
