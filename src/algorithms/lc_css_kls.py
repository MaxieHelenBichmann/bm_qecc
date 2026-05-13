"""KLS normal form for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
import pyzx as zx
from ..core.stabilizer_code import StabilizerCode

class ZXGraph:
    def __init__(self):
        # each node is represented as its phase-factor of pi, input nodes before output nodes
        self.n = 0
        self.k = 0
        self.vertices = []

        self.edges = set()

    def add_edge(self, idx1: float, idx2: float):
        self.edges.add({ idx1, idx2 })

    def add_input_vertex(self, u: float):
        self.vertices.append(u)
        self.k += 1
        self.n += 1

    def add_output_vertex(self, u: float):
        self.vertices.append(u)
        self.n += 1

    def get_input_vertices(self) -> list[float]:
        return self.vertices[:self.k]
    
    def get_output_vertices(self) -> list[float]:
        return self.vertices[self.k:]
    
    def get_input_output_adjacency(self) -> np.ndarray:
        adj = np.zeros((self.k, self.n), dtype=bool)
        for edge in self.edges:
            u, v = edge
            if u < self.k and v >= self.k:
                adj[u, v - self.k] = True
                adj[v - self.k, u] = True
            elif v < self.k and u >= self.k:
                adj[v, u - self.k] = True
                adj[u - self.k, v] = True
        return adj
    
    def get_full_adjacency(self) -> np.ndarray:
        adj = np.zeros((self.n + self.k, self.n + self.k), dtype=bool)
        for edge in self.edges:
            u, v = edge
            i = self.vertices.index(u)
            j = self.vertices.index(v)
            adj[i, j] = True
            adj[j, i] = True
        return adj
    
    def local_complementation(self, i: int):
        neighbors = [v for v in self.vertices if {i, v} in self.edges]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u = neighbors[i]
                v = neighbors[j]
                if {u, v} in self.edges:
                    self.edges.remove({u, v})
                else:
                    self.edges.add({u, v})

def _code_to_encoder_circuit(code) -> zx.Circuit:
    def _delete_first_row_and_qubit(tab: np.ndarray) -> np.ndarray:
        n = tab.shape[1] // 2
        return np.delete(np.delete(np.delete(tab, 0, axis=0), 0, axis=0) , n, axis=0)

    tableau = np.asarray(code.symplectic.copy(), dtype=np.uint8) & 1
    n = code.n
    k = code.k
    original_qubits = list(range(n))

    # Elimination gates are recorded in forward elimination order.
    elimination_gates: list[tuple[str, tuple[int, ...]]] = []
    while tableau.shape[0] > 0:
        cur_n = tableau.shape[1] // 2
        x0 = tableau[0, :cur_n]
        z0 = tableau[0, cur_n:]
        support = np.flatnonzero(x0 | z0)
        if len(support) == 0:
            raise ValueError("Encountered an identity stabilizer row.")

        # 1.) turn every non-identity Pauli in row 0 into Z
        # I = (0|0) --I---> I = (0|0)
        # X = (0|1) --H---> Z = (1|0)
        # Y = (1|1) --HS--> Z = (1|0)
        # Z = (0|1) --I---> Z = (1|0)
        for q in range(cur_n):
            x_q = bool(tableau[0, q])
            z_q = bool(tableau[0, cur_n + q])
            if x_q and z_q:
                # Y -> X under S
                tableau[:, q + n] ^= tableau[:, q]
                tableau[:, [q, q + n]] = tableau[:, [q + n, q]]
                elimination_gates.append(("S", (original_qubits[q],)))
                elimination_gates.append(("H", (original_qubits[q],)))
            elif x_q and not z_q:
                # X -> Z under H
                tableau[:, [q, q + n]] = tableau[:, [q + n, q]]
                elimination_gates.append(("H", (original_qubits[q],)))

        # 2.) make the first qubit Z CNOT(0 -> pivot)
        z_support = np.flatnonzero(tableau[0, cur_n:])

        if len(z_support) == 0:
            raise RuntimeError("Failed to convert stabilizer row to Z support.")
        pivot = int(z_support[0])

        if pivot != 0:
            tableau[:, pivot] ^= tableau[:, 0]
            tableau[:, cur_n + 0] ^= tableau[:, cur_n + pivot]

        # 3.) clear all other Zs in row 0 using CNOT(q -> 0)
        # control: (x_c|z_c) --CNOT--> (  x_c  |z_c^z_t)
        # target : (x_t|z_t) --CNOT--> (x_t^x_c|  z_t  )
        for q in range(1, cur_n):
            if tableau[:, cur_n + q] == 1:
                tableau[:, 0] ^= tableau[:, q]
                tableau[:, cur_n + q] ^= tableau[:, cur_n + 0]

                elimination_gates.append(
                    ("CNOT", (original_qubits[q], original_qubits[0]))
                )

        if (
            np.count_nonzero(tableau[0, :cur_n]) != 0
            or np.count_nonzero(tableau[0, cur_n:]) != 1
            or tableau[0, cur_n + pivot] != 1
        ):
            raise RuntimeError("Failed to isolate a stabilizer as a single Z.")

        # 4.) clear pivot column
        for r in range(1, tableau.shape[0]):
            if tableau[r, cur_n + pivot]:
                tableau[r] ^= tableau[0]
            if tableau[r, pivot]:
                raise RuntimeError(
                    "A remaining row has X on the pivot qubit."
                )

        # 5.) remove stabilizer and qubit
        tableau = _delete_first_row_and_qubit(tableau)
        del original_qubits[pivot]

    # encoder = inverse elimination Cliffords
    circuit = zx.Circuit(n)
    circuit.initialize_qubits([True] * (n-k) + [False] * k)
    for name, qubits in reversed(elimination_gates):
        if name == "H":
            circuit.add_gate("HAD", qubits[0])
        elif name == "S":
            # S† = Z phase 3π/2
            circuit.add_gate("ZPhase", qubits[0], phase=3 / 2)
        elif name == "CNOT":
            circuit.add_gate("CNOT", qubits[0], qubits[1])
        
    return circuit 

def _code_to_graph(code) -> ZXGraph:
    # 1.) code -> encoder circuit
    circuit = _code_to_encoder_circuit(code)

    # 2.) Encoder circuit -> zx diagram
    zx_diagram = circuit.to_graph()

    # 3.) zx diagram -> graph state
    zx_diagram.normalize() 

    # 4.) graph state -> ZXGraph
    # TODO
    graph = ZXGraph()

    print(zx_diagram)

def _hk_normal_form(graph: ZXGraph) -> ZXGraph:
    # TODO
    return graph

def _kls_normal_form(graph_hk: ZXGraph) -> ZXGraph:
    # 1.) set input vertices to have no phase
    for q in range(graph_hk.k):
        graph_hk.vertices[q] = 0

    # 2.) bring the IO-adjacency matrix into RREF
    adj = graph_hk.get_input_output_adjacency()
    gates = []
    for i in range(graph_hk.k):
        pivot_candidates = np.flatnonzero(adj[i, :])
        if len(pivot_candidates) == 0:
            continue
        pivot = int(pivot_candidates[0])
        for r in range(0, graph_hk.k):
            if r != i and adj[r, pivot]:
                adj[r] ^= adj[i]
                gates.append(("CNOT", (i, r)))

    # 3.) simplify the graph with the added gates
    zx_diagram = zx.Graph()
    for p in graph_hk.vertices:
        print(p)

    for u, v in graph_hk.edges:
        print(u, v)

    zx_diagram.normalize()

    # 4.) graph state -> ZXGraph
    # TODO
    graph = ZXGraph()

    return graph

def _is_bipartite(graph: ZXGraph) -> bool:
    adj = graph.get_full_adjacency()
    n = adj.shape[0]
    colors = np.full(n, -1, dtype=np.int8)

    for start in range(n):
        if colors[start] != -1:
            continue

        colors[start] = 0
        frontier = np.array([start], dtype=int)

        while frontier.size:
            current_color = colors[frontier[0]]
            next_color = 1 - current_color

            neighbors = np.flatnonzero(adj[frontier].any(axis=0))

            if np.any(colors[neighbors] == current_color):
                return False

            new = neighbors[colors[neighbors] == -1]
            colors[new] = next_color
            frontier = new

    return True


def is_lceq_css_kls(code: StabilizerCode) -> bool:
    graph = _code_to_graph(code)
    graph_hk = _hk_normal_form(graph)
    graph_kls = _kls_normal_form(graph_hk)
    return _is_bipartite(graph_kls)
