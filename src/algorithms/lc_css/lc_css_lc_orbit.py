"""LC Orbit traversal for checking whether a stabilizer code with k < 2 is LC-equivalent to a CSS code.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ...core.stabilizer_code import StabilizerCode

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

def _stab_state_to_graph_state(tableau: np.ndarray) -> np.ndarray:
    """Convert a stabilizer state into a graph state under local Clifford operations.
    Returns the adjacency matrix of the graph state."""
    n = tableau.shape[1] // 2

    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)

    def _make_X_invertible(t: np.ndarray) -> np.ndarray:
        old_x_rank = _rank(t[:, :n])
        while old_x_rank < n:
            improved = False

            for q in range(n):
                if old_x_rank == n:
                    break

                best_rank = old_x_rank
                best_choice = (None, None)

                x_col = t[:, q].copy()
                z_col = t[:, q + n].copy()

                for new_x, new_z in [ (x_col, z_col), (z_col, x_col), ((x_col + z_col) % 2, x_col) ]:
                    t[:, q] = new_x
                    new_x_rank = _rank(t[:, :n])
                    if new_x_rank > best_rank:
                        best_rank = new_x_rank
                        best_choice = (new_x, new_z)

                if best_choice[0] is not None:
                    t[:, q] = best_choice[0]
                    t[:, q + n] = best_choice[1]
                    old_x_rank = best_rank
                    improved = True
                else:
                    t[:, q] = x_col
                    t[:, q + n] = z_col

            if not improved:
                break

        return t

    def _extract_adjacency_matrix(tableau: np.ndarray) -> np.ndarray:
        """Extract the adjacency matrix from the stabilizer state."""
        def _rref_no_column_swaps(matrix: np.ndarray) -> np.ndarray:
            matrix = matrix.copy()
            n_rows, n_cols = matrix.shape
            pivot_row = 0
            for col in range(n_cols):
                if pivot_row >= n_rows:
                    break

                pivot_candidates = np.flatnonzero(matrix[pivot_row:, col])

                if pivot_candidates.size == 0:
                    continue

                pivot = pivot_row + int(pivot_candidates[0])

                if pivot != pivot_row:
                    matrix[[pivot_row, pivot], :] = matrix[[pivot, pivot_row], :]

                for r in range(n_rows):
                    if r != pivot_row and matrix[r, col]:
                        matrix[r, :] ^= matrix[pivot_row, :]
                pivot_row += 1

            return matrix

        rre = _rref_no_column_swaps(tableau)

        if _rank(rre[:, :n]) != n:
            raise ValueError("X part of the tableau is not full rank, something went wrong.")

        return rre[:, n:]

    def _remove_diagonal(tableau: np.ndarray) -> None:
        """Basically apply S gate on all qubits to remove self-loops in the graph state."""
        np.fill_diagonal(tableau, 0)

    state = _make_X_invertible(tableau)
    gamma = _extract_adjacency_matrix(state)
    _remove_diagonal(gamma)

    if not np.array_equal(gamma, gamma.T):
        raise ValueError("Extracted adjacency matrix is not symmetric, something went wrong.")

    return gamma

def _traverse_lc_orbit(graph: np.ndarray) -> bool:
    def _lc(graph: np.ndarray, q: int) -> np.ndarray:
        new_graph = graph.copy()
        neighbors = np.flatnonzero(graph[q])

        for idx in range(len(neighbors)):
            neighbor = neighbors[idx]
            for other_neighbor in neighbors[idx + 1:]:
                new_graph[neighbor, other_neighbor] ^= 1
                new_graph[other_neighbor, neighbor] ^= 1

        np.fill_diagonal(new_graph, 0)
        return new_graph

    def _canonical_key(graph: np.ndarray) -> bytes:
        return np.asarray(graph, dtype=np.uint8).tobytes()

    def _is_bipartite(graph: np.ndarray) -> bool:
        graph = np.asarray(graph, dtype=bool)
        n = graph.shape[0]
        colors = np.full(n, -1, dtype=np.int8)

        for start in range(n):
            if colors[start] != -1:
                continue

            colors[start] = 0
            frontier = np.array([start], dtype=int)

            while frontier.size:
                current_color = colors[frontier[0]]
                next_color = 1 - current_color

                neighbors = np.flatnonzero(graph[frontier].any(axis=0))

                if np.any(colors[neighbors] == current_color):
                    return False

                new = neighbors[colors[neighbors] == -1]
                colors[new] = next_color
                frontier = new

        return True

    n = graph.shape[0]
    seen = set()
    queue = deque([graph.copy()])

    while queue:
        current_graph = queue.popleft()
        if _is_bipartite(current_graph):
            return True

        for q in range(n):
            new_graph = _lc(current_graph, q)
            key = _canonical_key(new_graph)

            if key not in seen:
                queue.append(new_graph)
                seen.add(key)

    return False


def is_lceq_css_lc_orbit(code: StabilizerCode) -> bool:
    """Check if a stabilizer code with k < 2 is LC-equivalent to a CSS code by traversing the LC orbit of the corresponding graph state.

    1.) Convert the stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    2.) Convert the stabilizer state into a graph state under local Clifford operations.
    3.) Traverse the LC orbit of the graph state and check if any graph in the orbit is bipartite.


    This approach is only valid for stabilizer codes with k < 2, as the Choi-Jamiolkowski isomorphism fixes a certain logical basis and thus hides the freedom of choice of the logical operators (there can be arbitrary Cliffords on the input qubits, which can be entangling - thus not recognized - for more than one input qubit). For codes with k >= 2, this method will most likely lead to false negatives.
    It will only work for k >= 2 if it is guaranteed that the logical operators of the input code are already "matching" the logical basis of the potential CSS Code in the orbit, which is a very strong restriction and thus not generally applicable. 
    """
    stab_state = _stab_code_to_stab_state(code)
    graph_state = _stab_state_to_graph_state(stab_state)
    return _traverse_lc_orbit(graph_state)