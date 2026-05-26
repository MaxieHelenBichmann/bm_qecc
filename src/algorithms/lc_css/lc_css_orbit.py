"""LC Orbit traversal for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ...core.stabilizer_code import StabilizerCode

@dataclass(frozen=True)
class RedStabGraph:

    def __init__(self) -> None:
        self.n = 0
        self.k = 0

        # each node identified with the index in the array, input nodes after output nodes
        # each node is represented by a tuple with flags for the property of the node:
        # (self-loop, solid, minus-sign)
        self.vertices : list[tuple[bool, bool, bool]] = []
        self.edges : set[tuple[int, int]]= set() # (u,v) with u < v

    def toggle_edge(self, u:int , v:int) -> None:
        min_u_v = min(u, v)
        max_u_v = max(u, v)
        if (min_u_v, max_u_v) in self.edges:
            self.edges.remove((min_u_v, max_u_v))
        else:
            self.edges.add((min_u_v, max_u_v))

    def local_complementation(self, v:int) -> None:
        neighbors = self.neighbors(v)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                p = neighbors[i]
                q = neighbors[j]
                self.toggle_edge(p, q)

    def pivot_edge(self, u: int, v:int) -> None:
        self.local_complementation(u)
        self.local_complementation(v)
        self.local_complementation(u)

    def advance_loop(self, u: int) -> None:
        self.vertices[u] = (self.vertices[u][1] ^ True, self.vertices[u][1], self.vertices[u][2] ^ self.vertices[u][1])

    def flip_fill(self, u:int) -> None:
        self.vertices[u] = (self.vertices[u][0], self.vertices[u][1] ^ True, self.vertices[u][2])
    
    def flip_sign(self, u:int) -> None:
        self.vertices[u] = (self.vertices[u][0], self.vertices[u][1], self.vertices[u][2] ^ True)

    def neighbors(self, u: int) -> list[int]:
        return [v for v in range(self.n + self.k) if (u, v) in self.edges or (v, u) in self.edges]
    
    def input_neighbors(self, u: int) -> list[int]:
            return [v for v in range(self.n, self.n + self.k) if (u, v) in self.edges or (v, u) in self.edges]
    
    def copy(self) -> RedStabGraph:
        new_graph = RedStabGraph()
        new_graph.n = self.n
        new_graph.k = self.k
        new_graph.vertices = [ (l, h, s) for l, h, s in self.vertices ]
        new_graph.edges = self.edges.copy()
        return new_graph
    
    def adj_matrix(self) -> np.ndarray:
        nr = self.n + self.k
        adj = np.zeros((nr, nr), dtype=np.uint8)
        for u, v in self.edges:
            adj[u][v] = 1
            adj[v][u] = 1

    def canon_key(self) -> tuple[tuple[tuple[int,...]], tuple[tuple[bool, bool, bool]]]:
        return tuple(map(tuple, self.adj_matrix().tolist())), tuple(self.vertices)
    
    def apply_h(self, u:int) -> RedStabGraph:
        result = self.copy()
        l, s, m = result.vertices[u]
        non_solid_n = [a for a in result.neighbors(u) if not result.vertices[a][1]]
        if s and not l and not non_solid_n:  # T(i)
            result.flip_fill(u)
            return result
        if s and l and not non_solid_n: # T(ii)
            for n in result.neighbors(u):
                result.advance_loop(n)
                if not m:
                    result.flip_sign(n)

            result.local_complementation(u)
            result.flip_sign(u)
        if s and not l and non_solid_n: # T(iii)
            for n in non_solid_n:
                result.flip_fill(n)
                result.pivot_edge(n, u)
            # TODO
        if s and l and non_solid_n: # T(iv)
            for n in non_solid_n:
                result.local_complementation(u)
                result.local_complementation(n)

                result.vertices[u] = (False, result.vertices[u][0], result.vertices[u][0])

                for cur_n in result.neighbors(u):
                    result.advance_loop(cur_n)

                result.flip_fill(n)

                # TODO

        if not s: # T(v)
            result.flip_fill(u)
            return result

    def apply_s(self, u:int) -> RedStabGraph:
        result = self.copy()
        l, s, m = result.vertices[u]

        if s: # T(vi)
            result.advance_loop(u)
            return result
        else: # T(vii)
            for n in result.neighbors(u):
                result.advance_loop(n)
                if m:
                    result.flip_sign(n)

            result.local_complementation(u)

    def apply_z(self, u:int) -> RedStabGraph:
        result = self.copy()
        l, s, m = result.vertices[u]

        if s: # T5
            result.flip_sign(u)
            return result
        else:
            for n in result.neighbors(u): # T6
                result.flip_sign(n)

            if l:
                result.flip_sign(u)
         

    def apply_cz(self, u:int, v:int) -> RedStabGraph:
        pass

    def equivalent_graphs(self) -> list[RedStabGraph]:
        pass

    @staticmethod
    def from_adj_matrix(adj:np.ndarray, k : int, solid: int) -> RedStabGraph:
        graph = RedStabGraph()
        nr = adj.shape[1]
        graph.n = nr - k
        graph.k = k
        for i in range(nr):
            for j in range(i + 1, nr):
                if adj[i, j]:
                    graph.edges.add((i, j))

        graph.vertices = []
        for v in range(nr):
            node = (adj[v, v] & 1, v < solid, False)
            graph.vertices.append(node)
        return graph
    
    def is_bipartite(self) -> bool:
        n = self.n + self.k
        colors = np.full(n, -1, dtype=np.int8)

        neighbors_list : list[set] = [ set() for _ in range(n) ]
        for u, v in self.edges:
            (neighbors_list[u]).add(v)
            (neighbors_list[v]).add(u)

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
                    neighbors |= neighbors_list[node]
                neighbors = np.array(list(neighbors), dtype=int)

                if np.any(colors[neighbors] == current_color):
                    return False

                new = neighbors[colors[neighbors] == -1]
                colors[new] = next_color
                frontier = new

        return True


def _stab_code_to_stab_state(code: StabilizerCode) -> np.ndarray:
    """Convert a stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    Return only stabilizer tableau of the resulting stabilizer state.
    
    S = [S_x | S_z] ; Lx = [Lx_x | Lx_z] ; Lz = [Lz_x | Lz_z]

    S_choi = [S_x  | 0 | S_z  | 0]
             [Lx_x | I | Lx_z | 0]
             [Lz_x | 0 | Lz_z | I]
    """
    if code.k == 0:
        return code.symplectic

    n = code.n
    r = n - code.k
    k = code.k

    stab_x = code.symplectic[:, :n]
    stab_z = code.symplectic[:, n:]

    log_x_x = code.x_logicals.tableau.matrix[:, :n]
    log_x_z = code.x_logicals.tableau.matrix[:, n:]

    log_z_x = code.z_logicals.tableau.matrix[:, :n]
    log_z_z = code.z_logicals.tableau.matrix[:, n:]

    stabilizer_part = np.hstack([stab_x,np.zeros((r, k), dtype=np.int8), stab_z, np.zeros((r, k), dtype=np.int8)])
    logical_x_part = np.hstack([log_x_x,np.eye(k, dtype=np.int8),log_x_z,np.zeros((k, k), dtype=np.int8)])
    logical_z_part = np.hstack([log_z_x,np.zeros((k, k), dtype=np.int8),log_z_z,np.eye(k, dtype=np.int8)])

    return np.vstack([stabilizer_part, logical_x_part, logical_z_part]).astype(np.int8)

def _stab_state_to_graph_state(tableau: np.ndarray, n: int) -> RedStabGraph:
    """Convert a stabilizer state into a graph state under local Clifford operations.
    Returns the reduced stabilizer-state"""
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)

    def _bring_into_canon_form(t: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int]]]:
        pass

    def _extract_adjacency_matrix(tableau: np.ndarray, swaps: list[tuple[int, int]]) -> np.ndarray:
        """Extract the adjacency matrix from the stabilizer state."""
        pass

    s, canon, swaps = _bring_into_canon_form(tableau)
    gamma = _extract_adjacency_matrix(canon, swaps)

    if not np.array_equal(gamma, gamma.T):
        raise ValueError("Extracted adjacency matrix is not symmetric, something went wrong.")

    return RedStabGraph.from_adj_matrix(gamma, n, s)


def _traverse_lc_orbit(graph: RedStabGraph) -> bool:
    nr = graph.n + graph.k
    n = graph.n
    k = graph.k
    seen = set()
    queue = deque([graph.copy()])

    while queue:
        current_graph = queue.popleft()
        equivalent_graphs = current_graph.equivalent_graphs()
        for g in equivalent_graphs:
            if g.is_bipartite():
                return True

        for q in range(nr):
            new_graph_h = current_graph.apply_h(q)
            key_h = new_graph_h.canon_key()

            if key_h not in seen:
                queue.append(new_graph_h)
                seen.add(key_h)

            new_graph_s = current_graph.apply_s(q)
            key_s = new_graph_s.canon_key()

            if key_s not in seen:
                queue.append(new_graph_s)
                seen.add(key_s)

            new_graph_z = current_graph.apply_z(q)
            key_z = new_graph_z.canon_key()

            if key_z not in seen:
                queue.append(new_graph_z)
                seen.add(key_z)

        for q in range(n, n+k):
            for p in range(q+1, n+k):
                new_graph_cz = current_graph.apply_cz(q, p)
                key_cz = new_graph_cz.canon_key()

                if key_cz not in seen:
                    queue.append(new_graph_cz)
                    seen.add(key_cz) 

    return False


def is_lceq_css_orbit(code: StabilizerCode) -> bool:
    """Check if a stabilizer code is LC-equivalent to a CSS code by traversing the LC orbit of the corresponding graph state.

    1.) Convert the stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    2.) Convert the stabilizer state into a graph state under local Clifford operations.
    3.) Traverse the LC orbit of the graph state and check if any graph in the orbit is bipartite.


    The conversion from stabilizer code into a graph state can be done in O(n^3) time. The LC orbit can be large in general, this is not an efficient algorithm, but might be better than brute-force checking of all local Clifford operations, as duplicate graphs in the orbit can be identified and skipped ?.
    """
    stab_state = _stab_code_to_stab_state(code)
    graph_state = _stab_state_to_graph_state(stab_state, code.n + code.k)
    return _traverse_lc_orbit(graph_state)
