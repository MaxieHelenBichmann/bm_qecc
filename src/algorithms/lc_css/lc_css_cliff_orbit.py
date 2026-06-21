"""LC Orbit traversal for checking whether a stabilizer code is LC-equivalent to a CSS code.

References for this algorithm: 
- Matthew B. Elliott, Bryan Eastin, Carlton M. Caves: Graphical description of the action of Cliﬀord operators on stabilizer states
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ...core.stabilizer_code import StabilizerCode

GraphKey = tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[bool, bool, bool], ...],
]

@dataclass
class RedStabGraph:

    def __init__(self) -> None:
        self.n = 0
        self.k = 0

        # each node identified with the index in the array, input nodes after output nodes
        # each node is represented by a tuple with flags for the property of the node:
        # (self-loop, solid, minus-sign)
        self.vertices : list[tuple[bool, bool, bool]] = []
        self.edges : set[tuple[int, int]]= set() # (u,v) with u < v
        self.adj : list[set[int]] = [] # optimization
        self._adj_edge_count = 0

    def _rebuild_adjacency(self) -> None:
        nr = self.n + self.k
        self.adj = [set() for _ in range(nr)]
        for u, v in self.edges:
            if 0 <= u < nr and 0 <= v < nr and u != v:
                self.adj[u].add(v)
                self.adj[v].add(u)
        self._adj_edge_count = len(self.edges)

    def _ensure_adjacency(self) -> None:
        nr = self.n + self.k
        if len(self.adj) != nr or self._adj_edge_count != len(self.edges):
            self._rebuild_adjacency()

    def toggle_edge(self, u:int , v:int) -> None:
        if u == v:
            raise ValueError("Self-loops are stored on vertices, not in the edge set.")

        min_u_v = min(u, v)
        max_u_v = max(u, v)
        edge = (min_u_v, max_u_v)
        self._ensure_adjacency()
        if edge in self.edges:
            self.edges.remove(edge)
            self.adj[u].remove(v)
            self.adj[v].remove(u)
        else:
            self.edges.add(edge)
            self.adj[u].add(v)
            self.adj[v].add(u)
        self._adj_edge_count = len(self.edges)

    def local_complementation(self, v:int) -> None:
        self._ensure_adjacency()
        neighbors = list(self.adj[v])
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
        self.vertices[u] = (self.vertices[u][0] ^ True, self.vertices[u][1], self.vertices[u][2] ^ self.vertices[u][0])

    def flip_fill(self, u:int) -> None:
        self.vertices[u] = (self.vertices[u][0], self.vertices[u][1] ^ True, self.vertices[u][2])
    
    def flip_sign(self, u:int) -> None:
        self.vertices[u] = (self.vertices[u][0], self.vertices[u][1], self.vertices[u][2] ^ True)

    def neighbors(self, u: int) -> list[int]:
        self._ensure_adjacency()
        return list(self.adj[u])
    
    def input_neighbors(self, u: int) -> list[int]:
        self._ensure_adjacency()
        return [v for v in self.adj[u] if v >= self.n]
    
    def copy(self) -> RedStabGraph:
        self._ensure_adjacency()
        new_graph = RedStabGraph()
        new_graph.n = self.n
        new_graph.k = self.k
        new_graph.vertices = self.vertices.copy()
        new_graph.edges = self.edges.copy()
        new_graph.adj = [neighbors.copy() for neighbors in self.adj]
        new_graph._adj_edge_count = self._adj_edge_count
        return new_graph
    
    def adj_matrix(self) -> np.ndarray:
        nr = self.n + self.k
        adj = np.zeros((nr, nr), dtype=np.uint8)
        for u, v in self.edges:
            adj[u][v] = 1
            adj[v][u] = 1

        return adj

    def canon_key(self) -> GraphKey:
        return tuple(sorted(self.edges)), tuple(self.vertices)
    
    def is_valid(self) -> bool:
        nr = self.n + self.k
        self._ensure_adjacency()
        for u, v in self.edges:
            if not (0 <= u < v < nr):
                return False

            if v not in self.adj[u] or u not in self.adj[v]:
                return False

        for u in range(self.n + self.k):
            if self.vertices[u][1]:
                continue

            if self.vertices[u][0]: # hollow nodes never have loops
                return False
            
            for v in range(u + 1, self.n + self.k): # hollow nodes have no edges between them
                if self.vertices[v][1]:
                    continue
                if (u, v) in self.edges:
                    return False
        return True
    
    def apply_h(self, u:int) -> RedStabGraph:
        result = self.copy()
        lp, s, m = result.vertices[u]
        non_solid_n = [a for a in result.neighbors(u) if not result.vertices[a][1]]

        if s and not lp and not non_solid_n:  # T(i)
            result.flip_fill(u)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
        
        if s and lp and not non_solid_n: # T(ii)
            result.local_complementation(u)

            for n in result.neighbors(u):
                result.advance_loop(n)
                if not m:
                    result.flip_sign(n)

            result.flip_sign(u)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")

            return result
        
        if s and not lp and non_solid_n: # T(iii)
            n = non_solid_n[0]

            result.flip_fill(n)
            result.pivot_edge(n, u)
            for nn in result.adj[n] & result.adj[u]:
                result.flip_sign(nn)

            if m:
                result.flip_sign(u)
                for cur_n in result.neighbors(u):
                    result.flip_sign(cur_n)

            if result.vertices[n][2]:
                result.flip_sign(n)
                for cur_n in result.neighbors(n):
                    result.flip_sign(cur_n)
            
            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
   
        if s and lp and non_solid_n: # T(iv)
            n = non_solid_n[0]

            init_m = result.vertices[n][2]
            both = result.adj[n] & result.adj[u]

            result.local_complementation(u)
            result.local_complementation(n)

            result.vertices[u] = (False, result.vertices[u][1], result.vertices[u][2])

            for cur_n in result.neighbors(u):
                result.advance_loop(cur_n)

            result.flip_fill(n)

            for b in both:
                result.flip_sign(b)
            
            if m:
                result.flip_sign(u)
                for cur_n in result.neighbors(u):
                    result.flip_sign(cur_n)

            if init_m:
                for cur_n in result.neighbors(n):
                    result.flip_sign(cur_n)
                
            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result

        if not s: # T(v)
            result.flip_fill(u)
            return result
        
        return result

    def apply_s(self, u:int) -> RedStabGraph:
        result = self.copy()
        _, s, m = result.vertices[u]

        if s: # T(vi)
            result.advance_loop(u)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
        
        else: # T(vii)
            for n in result.neighbors(u):
                result.advance_loop(n)
                if m:
                    result.flip_sign(n)

            result.local_complementation(u)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            return result

    def apply_z(self, u:int) -> RedStabGraph:
        result = self.copy()
        lp, s, _ = result.vertices[u]

        if s: # T5
            result.flip_sign(u)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
        
        else:
            for n in result.neighbors(u): # T6
                result.flip_sign(n)

            if lp:
                result.flip_sign(u)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
         

    def apply_cz(self, u:int, v:int) -> RedStabGraph:
        result = self.copy()
        l_u, s_u, m_u = self.vertices[u]
        l_v, s_v, m_v = self.vertices[v]

        init_connected = (min(u, v), max(u, v)) in result.edges

        if s_u and s_v: # T(viii)
            result.toggle_edge(u, v)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
        
            return result
        
        if s_u and not s_v or s_v and not s_u: # T(ix)
            hollow = v if s_u else u
            solid = u if s_u else v

            hollow_m = m_v if s_u else m_u

            for n in result.neighbors(hollow):
                if n == solid:
                    continue
                result.toggle_edge(solid, n)

            if init_connected and not hollow_m or not init_connected and hollow_m:
                result.flip_sign(solid)
                for n in result.neighbors(solid):
                    result.flip_sign(n)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
        
        if not s_u and not s_v: # T(x)
            both = result.adj[v] & result.adj[u]
            n_u = result.neighbors(u)
            n_v = result.neighbors(v)

            result.pivot_edge(u, v)

            for b in both:
                result.flip_sign(b)

            if m_u:
                for n in n_v:
                    result.flip_sign(n)
            if m_v:
                for n in n_u:
                    result.flip_sign(n)

            if not result.is_valid():
                raise ValueError("Resulting graph is not reduced, something went wrong.")
            
            return result
        
        return result


    def equivalent_graphs(self, seen: set[GraphKey]) -> bool:
        key = self.canon_key()
        seen.add(key)
        queue = deque([self.copy()])

        while queue:
            current_graph = queue.popleft()

            if current_graph.is_bipartite():
                return True

            for u, v in current_graph.edges:
                _, s_u, _ = current_graph.vertices[u]
                _, s_v, _ = current_graph.vertices[v]

                if s_u and not s_v or s_v and not s_u: 
                    hollow = v if s_u else u
                    solid = u if s_u else v

                    if current_graph.vertices[solid][0]: # E(i)
                        new_graph = current_graph.copy()

                        init_solid = new_graph.vertices[solid][2]
                        init_hollow = new_graph.vertices[hollow][2]
                        both = new_graph.adj[v] & new_graph.adj[u]

                        new_graph.local_complementation(solid)
                        new_graph.local_complementation(hollow)

                        new_graph.vertices[solid] = (False, new_graph.vertices[solid][1], new_graph.vertices[solid][2])

                        for n in new_graph.neighbors(solid):
                            new_graph.advance_loop(n)

                        new_graph.flip_fill(hollow)
                        new_graph.flip_fill(solid)

                        for b in both:
                            new_graph.flip_sign(b)
                        
                        if init_solid:
                            new_graph.flip_sign(solid)
                            for cur_n in new_graph.neighbors(solid):
                                new_graph.flip_sign(cur_n)

                        if init_hollow:
                            for cur_n in new_graph.neighbors(hollow):
                                new_graph.flip_sign(cur_n)

                        key = new_graph.canon_key()

                        if key not in seen:
                            queue.append(new_graph)
                            seen.add(key)

                    else: # E(ii)
                        new_graph = current_graph.copy()
                        init_u = new_graph.vertices[u][2]
                        init_v = new_graph.vertices[v][2]

                        both = new_graph.adj[v] & new_graph.adj[u]

                        new_graph.pivot_edge(u, v)

                        new_graph.flip_fill(u)
                        new_graph.flip_fill(v)

                        for b in both:
                            new_graph.flip_sign(b)

                        if init_u:
                            for cur_n in new_graph.neighbors(u):
                                new_graph.flip_sign(cur_n)

                        if init_v:
                            for cur_n in new_graph.neighbors(v):
                                new_graph.flip_sign(cur_n)

                        key = new_graph.canon_key()

                        if key not in seen:
                            queue.append(new_graph)
                            seen.add(key)


        return False

    @staticmethod
    def from_adj_matrix(adj:np.ndarray, k : int, solid: int | set[int]) -> RedStabGraph:
        graph = RedStabGraph()
        nr = adj.shape[1]
        graph.n = nr - k
        graph.k = k
        graph.adj = [set() for _ in range(nr)]
        solid_vertices = set(range(solid)) if isinstance(solid, int) else solid
        for i in range(nr):
            for j in range(i + 1, nr):
                if adj[i, j]:
                    graph.edges.add((i, j))
                    graph.adj[i].add(j)
                    graph.adj[j].add(i)
        graph._adj_edge_count = len(graph.edges)

        graph.vertices = []
        for v in range(nr):
            node = (bool(adj[v, v] & 1), v in solid_vertices, False)
            graph.vertices.append(node)
        return graph
    
    def is_bipartite(self) -> bool:
        n = self.n + self.k
        self._ensure_adjacency()
        colors = [-1] * n

        for start in range(n):
            if colors[start] != -1:
                continue

            colors[start] = 0
            frontier = deque([start])

            while frontier:
                node = frontier.popleft()
                current_color = colors[node]
                next_color = 1 - current_color

                for neighbor in self.adj[node]:
                    if colors[neighbor] == current_color:
                        return False
                    if colors[neighbor] == -1:
                        colors[neighbor] = next_color
                        frontier.append(neighbor)

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
        return code.symplectic.copy()

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

def _stab_state_to_graph_state(tableau: np.ndarray, n: int, k: int) -> RedStabGraph:
    """Convert a stabilizer state into a graph state under local Clifford operations.
    Returns the reduced stabilizer-state"""
    def _bring_into_canon_form() -> tuple[int, list[tuple[int, int]], set[int]]:
        swaps = []
        qubit_order = list(range(n))

        # row-reduce X block
        row = 0
        for col_target in range(n):
            pivot = None
            for c in range(col_target, n):
                for rr in range(row, n):
                    if tableau[rr, c] == 1:
                        pivot = (rr, c)
                        break
                if pivot is not None:
                    break

            if pivot is None:
                break

            pivot_row, pivot_col = pivot
            if pivot_col != col_target:
                tableau[:, [pivot_col, col_target]] = tableau[:, [col_target, pivot_col]]
                tableau[:, [pivot_col + n, col_target + n]] = tableau[:, [col_target + n, pivot_col + n]]
                qubit_order[pivot_col], qubit_order[col_target] = qubit_order[col_target], qubit_order[pivot_col]
                swaps.append((pivot_col, col_target))

            if pivot_row != row:
                tableau[[pivot_row, row], :] = tableau[[row, pivot_row], :]

            for rr in range(n):
                if rr != row and tableau[rr, col_target] == 1:
                    tableau[rr, :] = (tableau[rr, :] + tableau[row, :]) % 2
            row += 1

        # X = [ I A ]
        #     [ 0 0 ]

        r = row

        # row-reduce Z block
        lower_row = row

        for col_target in range(r, n):
            pivot = None
            for c in range(col_target, n):
                for rr in range(lower_row, n):
                    if tableau[rr, n + c] == 1:
                        pivot = (rr, c)
                        break
                if pivot is not None:
                    break

            if pivot is None:
                raise ValueError("Unexpected rank deficiency in Z block, something went wrong.")

            pivot_row, pivot_col = pivot
            if pivot_col != col_target:
                tableau[:, [pivot_col, col_target]] = tableau[:, [col_target, pivot_col]]
                tableau[:, [pivot_col + n, col_target + n]] = tableau[:, [col_target + n, pivot_col + n]]
                qubit_order[pivot_col], qubit_order[col_target] = qubit_order[col_target], qubit_order[pivot_col]
                swaps.append((pivot_col, col_target))
            
            if pivot_row != lower_row:
                tableau[[pivot_row, lower_row], :] = tableau[[lower_row, pivot_row], :]

            for rr in range(n):
                if rr != lower_row and tableau[rr, n + col_target] == 1:
                    tableau[rr, :] = (tableau[rr, :] + tableau[lower_row, :]) % 2
            lower_row += 1

        # clear upper corner of Z block
        for top in range(r):
            for c in range(r, n):
                if tableau[top, n + c] == 1:
                    lower = r + (c - r)
                    tableau[top, :] = (tableau[top, :] + tableau[lower, :]) % 2

        return r, swaps, set(qubit_order[:r])

    def _extract_adjacency_matrix(swaps: list[tuple[int, int]], s: int) -> np.ndarray:
        X = tableau[:, :n]
        Z = tableau[:, n:]

        A = X[:s, s:n]
        B = Z[:s, :s]

        gamma = np.zeros((n, n), dtype=np.uint8)
        gamma[:s, :s] = B
        gamma[:s, s:n] = A
        gamma[s:n, :s] = A.T

        for col1, col2 in reversed(swaps):
            gamma[:, [col1, col2]] = gamma[:, [col2, col1]]
            gamma[[col1, col2], :] = gamma[[col2, col1], :]

        return gamma

    s, swaps, solid_vertices = _bring_into_canon_form()
    gamma = _extract_adjacency_matrix(swaps, s)

    if not np.array_equal(gamma, gamma.T):
        raise ValueError("Extracted adjacency matrix is not symmetric, something went wrong.")

    return RedStabGraph.from_adj_matrix(gamma, k, solid_vertices)


def _traverse_cliff_orbit(graph: RedStabGraph) -> bool:
    nr = graph.n + graph.k
    n = graph.n
    k = graph.k
    seen : set[GraphKey] = set()
    queue = deque([graph.copy()])

    while queue:
        current_graph = queue.popleft()

        if current_graph.equivalent_graphs(seen):
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


def is_lceq_css_cliff_orbit(code: StabilizerCode) -> bool:
    """Check if a stabilizer code is LC-equivalent to a CSS code by traversing the LC orbit of the corresponding graph state.

    1.) Convert the stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    2.) Convert the stabilizer state into a graph state under local Clifford operations.
    3.) Traverse the LC orbit of the graph state on the physical qubits, as well as the general clifford orbit for the input/reference qubits, and check if any graph in the orbit is bipartite.


    The conversion from stabilizer code into a graph state can be done in O(n^3) time. The LC orbit can be large in general, and the general Clifford orbit on the input/reference qubits adds an additional factor, so the algorithm is even worse than the brute-force checking of all local Cliffords.
    """
    stab_state = _stab_code_to_stab_state(code)
    graph_state = _stab_state_to_graph_state(stab_state, code.n + code.k, code.k)
    return _traverse_cliff_orbit(graph_state)
