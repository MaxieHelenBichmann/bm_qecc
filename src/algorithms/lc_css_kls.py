"""KLS normal form for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
import pyzx as zx
from fractions import Fraction

from ..core.stabilizer_code import StabilizerCode

class ZXGraph:
    def __init__(self):
        self.n = 0
        self.k = 0
        # each node is represented as its local clifford decoration, and identified with the index in the array, input nodes before output nodes
        # I -> 0, S -> 1, Z -> 2, S† -> 3,  H -> 4, HS -> 5, HZ -> 6, HS† -> 7
        self.vertices = []
        self.edges : set[tuple[int, int]]= set() # (u,v) with u < v
    
    def get_input_output_adjacency(self) -> np.ndarray:
        adj = np.zeros((self.k, self.n), dtype=bool)
        for edge in self.edges:
            u, v = edge
            if u < self.k and v >= self.k:
                adj[u, v - self.k] = True
                adj[v - self.k, u] = True
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
    
    def neighbors(self, v: int) -> list[int]:
        return [u for u in self.vertices if (u, v) in self.edges or (v, u) in self.edges]

    def toggle_edge(self, u: int, v: int):
        min_u_v = min(u, v)
        max_u_v = max(u, v)
        if (min_u_v, max_u_v) in self.edges:
            self.edges.remove((min_u_v, max_u_v))
        else:
            self.edges.add((min_u_v, max_u_v))
    
    def graph_local_complementation(self, i: int):
        neighbors = self.neighbors(i)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                self.toggle_edge(neighbors[i], neighbors[j])

    def graph_pivot_edge(self, u: int, v: int):
        min_u_v = min(u, v)
        max_u_v = max(u, v)
        if (min_u_v, max_u_v) in self.edges:
           self.graph_local_complementation(u)
           self.graph_local_complementation(v)
           self.graph_local_complementation(u)

    @classmethod
    def decoration_to_word(d : int) -> list[str]:
        if d == 0:
            return []
        elif d == 1:
            return ["S"]
        elif d == 2:
            return ["S", "S"]
        elif d == 3:
            return ["S", "S", "S"]
        elif d == 4:
            return ["H"]
        elif d == 5:
            return ["H", "S"]
        elif d == 6:
            return ["H", "S", "S"]
        elif d == 7:
            return ["H", "S", "S", "S"]
        else:
            raise ValueError(f"Invalid local Clifford decoration {d}.")
        
    @classmethod
    def word_to_decoration(s: list[str]) -> int:
        if s == []:
            return 0
        elif s == ["S"]:
            return 1
        elif s == ["S", "S"]:
            return 2
        elif s == ["S", "S", "S"]:
            return 3
        elif s == ["H"]:
            return 4
        elif s == ["H", "S"]:
            return 5
        elif s == ["H", "S", "S"]:
            return 6
        elif s == ["H", "S", "S", "S"]:
            return 7

    @classmethod
    def from_pyzx_diagram(diagram: zx.Graph) -> ZXGraph:
        def _pyzx_to_local_clifford(phase: Fraction, edge_type: zx.EdgeType) -> int:
            if edge_type == zx.EdgeType.SIMPLE:
                if (phase is None or phase == 0):
                    return 0 # I
                elif phase == Fraction(1, 2):
                    return 1 # S
                elif phase == Fraction(1, 1):
                    return 2 # Z
                elif phase == Fraction(3, 2):
                    return 3 # S†
                else:
                    raise ValueError(f"Unexpected vertex phase {phase} in ZX diagram.")
            elif edge_type == zx.EdgeType.HADAMARD:
                if (phase is None or phase == 0):
                    return 4 # H
                elif phase == Fraction(1, 2):
                    return 5 # SH
                elif phase == Fraction(1, 1):
                    return 6 # ZH
                elif phase == Fraction(3, 2):
                    return 7 # S†H
                else:
                    raise ValueError(f"Unexpected vertex phase {phase} in ZX diagram.")
            
        graph = ZXGraph()
        pyzx_input_boundaries = sorted([ v for v in diagram.inputs()], key=lambda v: v.id)
        pyzx_output_boundaries = sorted([ v for v in diagram.outputs()], key=lambda v: v.id)
        pyzx_vertices = []

        for input_boundary in pyzx_input_boundaries:
            n = diagram.neighbors(input_boundary)
            if len(n) != 1:
                raise ValueError("Expected ZX diagram in encoder form, but found an input vertex with degree != 1.")
            graph.vertices.append(_pyzx_to_local_clifford(diagram.phase(n[0]), diagram.edge_type(input_boundary, n[0])))
            pyzx_vertices.append(n[0])

        for output_boundary in pyzx_output_boundaries:
            n = diagram.neighbors(output_boundary)
            if len(n) != 1:
                raise ValueError("Expected ZX diagram in encoder form, but found an output vertex with degree != 1.")
            graph.vertices.append(_pyzx_to_local_clifford(diagram.phase(n[0]), diagram.edge_type(output_boundary, n[0])))
            pyzx_vertices.append(n[0])

        for edge in diagram.edges():
            u, v = diagram.edge_st(edge)
            if u in pyzx_vertices and v in pyzx_vertices:
                idx_u = pyzx_vertices.index(u)
                idx_v = pyzx_vertices.index(v)
                graph.add_edge(graph.vertices[min(idx_u, idx_v)], graph.vertices[max(idx_u, idx_v)])
        

        return graph
    
    def to_pyzx_diagram(self, cnots: list[tuple[int, int]] = []) -> zx.BaseGraph:
        def _local_clifford_to_pyzx(lc_decoration: int) -> tuple[Fraction, zx.EdgeType]:
            if lc_decoration == 0:
                return (Fraction(0), zx.EdgeType.SIMPLE) # I
            elif lc_decoration == 1:
                return (Fraction(1, 2), zx.EdgeType.SIMPLE) # S
            elif lc_decoration == 2:
                return (Fraction(1, 1), zx.EdgeType.SIMPLE) # Z
            elif lc_decoration == 3:
                return (Fraction(3, 2), zx.EdgeType.SIMPLE) # S†
            elif lc_decoration == 4:
                return (Fraction(0), zx.EdgeType.HADAMARD) # H
            elif lc_decoration == 5:
                return (Fraction(1, 2), zx.EdgeType.HADAMARD) # SH
            elif lc_decoration == 6:
                return (Fraction(1, 1), zx.EdgeType.HADAMARD) # ZH
            elif lc_decoration == 7:
                return (Fraction(3, 2), zx.EdgeType.HADAMARD) # S†H
            
        def _add_cnots(diagram, inputs):
            for control, target in cnots:
                new_x = diagram.add_vertex(zx.VertexType.X) # place on target

                input_target = inputs[target]
                neighbor = diagram.neighbors(input_target)[0]

                e = diagram.edge(input_target, neighbor)
                e_type = diagram.edge_type(input_target, neighbor)
                diagram.remove_edge(e)
                diagram.add_edge(new_x, neighbor, edge_type=e_type)
                diagram.add_edge(new_x, input_target, edge_type=zx.EdgeType.SIMPLE)


                new_z = diagram.add_vertex(zx.VertexType.Z) # place on control
                
                input_control = inputs[control]
                neighbor = diagram.neighbors(input_control)[0]
                
                e = diagram.edge(input_control, neighbor)
                e_type = diagram.edge_type(input_control, neighbor)
                diagram.remove_edge(e)
                diagram.add_edge(new_z, neighbor, edge_type=e_type)
                diagram.add_edge(new_z, input_control, edge_type=zx.EdgeType.SIMPLE)

                diagram.add_edge(new_z, new_x, edge_type=zx.EdgeType.SIMPLE)
            
        diagram = zx.Graph()
        vertex_map = {}
        input_boundaries = []
        output_boundaries = []
        for i, deco in enumerate(self.vertices):
            if i == self.k:
                _add_cnots(diagram, input_boundaries)
            boundary = diagram.add_vertex(zx.VertexType.BOUNDARY)
            if i < self.k:      
                input_boundaries.append(boundary)
            else:
                output_boundaries.append(boundary)

            phase , edge_type = _local_clifford_to_pyzx(deco)
            v = diagram.add_vertex(zx.VertexType.Z, phase=phase)
            vertex_map[i] = v
            diagram.add_edge(boundary, v, edge_type=edge_type)
        
        diagram.set_inputs(input_boundaries)
        diagram.set_outputs(output_boundaries)

        for edge in self.edges:
            u, v = edge
            diagram.add_edge(vertex_map[u], vertex_map[v], edge_type=zx.EdgeType.HADAMARD)

        return diagram
    
    def is_bipartite(self) -> bool:
        n = self.n + self.k
        colors = np.full(n, -1, dtype=np.int8)

        neighbors = { v: set() for v in self.vertices }
        for u, v in self.edges:
            neighbors[u].add(v)
            neighbors[v].add(u)

        for start in range(n):
            if colors[start] != -1:
                continue

            colors[start] = 0
            frontier = np.array([start], dtype=int)

            while frontier.size:
                current_color = colors[frontier[0]]
                next_color = 1 - current_color

                neighbors = set()
                for node in frontier:
                    neighbors |= neighbors[node]
                neighbors = np.array(list(neighbors), dtype=int)

                if np.any(colors[neighbors] == current_color):
                    return False

                new = list(neighbors)[colors[neighbors] == -1]
                colors[new] = next_color
                frontier = new

        return True

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

    # 3.) zx diagram -> graph like state
    zx.to_graph_like(zx_diagram)
    zx_diagram.normalize() 

    if not zx.is_graph_like(zx_diagram, strict=True):
        raise ValueError("Expected the ZX diagram to be graph-like after normalization, but it was not.")

    # 4.) graph state -> ZXGraph
    graph = ZXGraph.from_pyzx_diagram(zx_diagram)

    return graph

def _hk_normal_form(graph: ZXGraph) -> ZXGraph:
    """Requirements HK normal form:
    1.) graph like state (only Z nodes, one Z node per boundary, no inner Z edges, only H-edges between Z nodes)
    2.) Requirements on LC decorations:
        - only decorations I , S , H allowed
        - if vertex has a H decoration, then it cannot have a neighbor with a smaller index
    """ 
    def _h_slide_down(g: ZXGraph, upper: int, lower: int) -> ZXGraph:
        # (Eq. 13) H_upper |G> = H_lower Z_upper Z_lower prod_{p in A, q in B} CZ_{p,q} |G>
        # with A = N(upper) union {upper}, B = N(lower) union {lower}
        def _add_z_decoration(v: int):
            if g.vertices[v] in (0, 4): # I -> Z, H -> HZ
                g.vertices[v] += 2
            elif g.vertices[v] in (1, 5): # S -> S†, SH -> S†H
                g.vertices[v] += 2
            elif g.vertices[v] in (2, 6): # Z -> I, ZH -> H
                g.vertices[v] -= 2
            elif g.vertices[v] in (3, 7): # S† -> S, S†H -> SH
                g.vertices[v] -= 2

        A = set(g.neighbors(upper)) | {upper}
        B = set(g.neighbors(lower)) | {lower}

        g.vertices[upper] -= 4 # remove H decoration
        g.vertices[lower] += 4 # add H decoration

        g.vertices[upper] = _add_z_decoration(g.vertices[upper])
        g.vertices[lower] = _add_z_decoration(g.vertices[lower])

        for p in A:
            for q in B:
                if p == q:
                    g.vertices[p] = _add_z_decoration(g.vertices[p])
                else:
                    g.toggle_edge(p, q)

        return g
    
    def _reduce_trailing_HS(g: ZXGraph, words: list[list[str]], i: int):
        # (Eq. 12) H_i S_i |G> = S_i^3 prod_{p in N(i)} Z_p prod_{p,q in N(i)} CS_{p,q} |G>
        words[i] = words[i][:-2]

        words[i] += ["S", "S", "S"]

        for j in g.neighbors(i):
            words[j] += ["S", "S"]

        for p in g.neighbors(i):
            for q in g.neighbors(i):
                # TODO
                pass

    def _reduce_solo_trailing_SH(g: ZXGraph, words: list[list[str]], i: int):
        # (Eq. 11) S_i H_i |G> = H_i prod_{p,q in N(i)} CS_{p,q} |G>
        words[i] = words[i][:-2] + ["H"]

        for p in g.neighbors(i):
            for q in g.neighbors(i):
                # TODO
                pass

    def _reduce_shared_trailing_SH(g: ZXGraph, words: list[list[str]], i: int, j: int):
        # (Eq. 11) H_i H_j|G> = Z_i Z_j prod_{p in A, q in B} CZ_{p,q} |G>
        # with A = N(i) union {i}, B = N(j) union {j}
        def _add_z_decoration(v: int) -> list[str]:
            if v[-2:] == ["S", "S"]:
                return v[-2:]
            return v + ["S", "S"]
            

        A = set(g.neighbors(i)) | {i}
        B = set(g.neighbors(j)) | {j}

        words[i] = words[i][:-1]
        words[j] = words[j][:-1]

        words[i] = _add_z_decoration(words[i])
        words[j] = _add_z_decoration(words[j])

        for p in A:
            for q in B:
                if p == q:
                    words[p] = _add_z_decoration(words[p])
                else:
                    g.toggle_edge(p, q)

        return g
    
    def _reduce_word(word: list[str]) -> list[str]:
        changed = True
        while changed:
            new_word = []
            changed = False
            for i in range(len(word)):
                if i < len(word) - 1 and ((word[i] == "H" and word[i + 1] == "H") or (word[i] == "Z" and word[i + 1] == "Z")):
                    changed = True
                elif i < len(word) - 1 and (word[i] == "S" and word[i + 1] == "S"):
                    new_word += ["Z"]
                    changed = True
                else:
                    new_word.append(word[i])
            word = new_word
        
        return word
    
    n = graph.n + graph.k

    # 1.) reduce LC decorations to either I, S, or H
    words = [ZXGraph.decoration_to_word(d) for d in graph.vertices] # during step 1 this is the ONLY valid state of the decorations, not the graph.vertices array
    changed = True
    while changed:
        changed = False 

        for i in range(len(n)):
            words[i] = _reduce_word(words[i])
            if len(words[i]) < 2:
                continue
            if words[i][-2:] == ["H", "S"]:
                _reduce_trailing_HS(graph, words, i)
                changed = True
                break
            if words[i][-2:] == ["S", "H"]:
                h_neighbors = [ j for j in graph.neighbors(i) if len(words[j]) >= 2 and words[j][-1:] == ["H"] ]
                if len(h_neighbors) == 0:
                    _reduce_solo_trailing_SH(graph, words, i)
                else:
                    _reduce_shared_trailing_SH(graph, words, i, h_neighbors[0])
                changed = True
                break
    graph.vertices = [ZXGraph.word_to_decoration(_reduce_word(w)) for w in words] # should only be I, S, or H decorations left I hope

    # 2.) slides H down
    for x in reversed(range(n)):
        while graph.vertices[x] in (5, 7): # x has a Hadamard component
            low_neighbors = [v for v in graph.vertices if (v, x) in graph.edges]
            if len(low_neighbors) == 0:
                break

            y = min(low_neighbors)
            graph = _h_slide_down(graph, x, y)
    
    # 3.) cleanup decorations

    return graph

def _kls_normal_form(graph_hk: ZXGraph) -> ZXGraph:
    """Additional requirements KLS normal form:
    1.) input nodes have no non-I LC decoration, and no edges between input nodes
    2.) the input-output adjacency matrix is in RREF
    3.) the pivot columns of the input-output adjacency matrix have no non-I LC decorations, and no edges between pivot vertices
    """
    # 1.) set input vertices to have no decoration and no edges
    for q in range(graph_hk.k):
        graph_hk.vertices[q] = 0
        for i in range(q + 1, graph_hk.k):
            if {q, i} in graph_hk.edges:
                graph_hk.edges.remove({q, i})

    # 2.) bring the IO-adjacency matrix into RREF
    adj = graph_hk.get_input_output_adjacency()
    cnots_gates = []
    for i in range(graph_hk.k):
        pivot_candidates = np.flatnonzero(adj[i, :])
        if len(pivot_candidates) == 0:
            continue
        pivot = int(pivot_candidates[0])
        for r in range(0, graph_hk.k):
            if r != i and adj[r, pivot]:
                adj[r] ^= adj[i]
                cnots_gates.append((i, r))

    # 3.) simplify the graph with the added CNOTS
    zx_diagram = graph_hk.to_pyzx_diagram(cnots=cnots_gates)

    zx.to_graph_like(zx_diagram)
    zx_diagram.normalize() 

    # 4.) graph state -> ZXGraph
    graph = ZXGraph.from_pyzx_diagram(zx_diagram)
    return graph

def is_lceq_css_kls(code: StabilizerCode) -> bool:
    """Check if a stabilizer code is LC-equivalent to a CSS code by converting it into KLS normal form and checking if the resulting graph state is bipartite.

    1.) Convert the stabilizer code into a clifford encoder circuit.
    2.) Convert the encoder circuit into a ZX diagram.
    3.) Simplify the ZX diagram into a graph-like state, aka a graph with local Clifford decorations on the vertices.
    4.) Convert the graph-like state into HK normal form, while treating the input vertices as output vertices.
    5.) Convert the HK normal form into KLS normal form, treating the input vertices as inputs again.

    The conversion from stabilizer code into a graph state can be done in O(n^3) time an the efficient algorithm for equivalence checking of graph states runs in O(n^4) time, so the overall runtime of this algorithm should be O(n^3) which is very efficient.
    """
    graph = _code_to_graph(code)
    graph_hk = _hk_normal_form(graph)
    graph_kls = _kls_normal_form(graph_hk)
    return graph_kls.is_bipartite()
