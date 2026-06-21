"""Best hybrid solution for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import multiprocessing
from collections import deque
from itertools import product

from typing import TYPE_CHECKING

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

from ..core.stabilizer_code import StabilizerCode


def is_lceq_css(code: StabilizerCode) -> bool:
    if code.n <= 5:
        return _bruteforce(code)
    
    if code.n < 1:
        return True
    
    reduced_symplectic = _row_basis(code.symplectic)
    
    if code.k < 2:
        return _lc_orbit(code, reduced_symplectic)
    
    return False # TODO: k >= 2

# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------
LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")

def _bruteforce(code: StabilizerCode) -> bool:
    """lc_css_bruteforce.py"""
    n = code.n
    r = _rank(code.symplectic)

    def apply_lc(tableau: npt.NDArray[np.int8], lc: str, qubit: int) -> npt.NDArray[np.int8]:
        if lc == "I":
            pass
        elif lc  == "H":
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "S":
            tableau[:, qubit + n] ^= tableau[:, qubit]
        elif lc == "HS":
            tableau[:, qubit + n] ^=  tableau[:, qubit]
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "SH":
            tableau[:, qubit] ^= tableau[:, qubit + n]
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "HSH":
            tableau[:, qubit] ^= tableau[:, qubit + n]
        return tableau

    for action in product(LOCAL_CLIFFORDS, repeat=n):
        lc_tableau = code.symplectic.copy()

        for qubit, lc in enumerate(action):
            lc_tableau = apply_lc(lc_tableau, lc, qubit)

        if _rank(lc_tableau[:, :n]) + _rank(lc_tableau[:, n:]) == r:
            return True

    return False

def _lc_orbit(code: StabilizerCode, reduced_symplectic: np.ndarray) -> bool:
    """lc_css_lc_orbit.py"""

    def _stab_code_to_stab_state(code: StabilizerCode, reduced_symplectic: np.ndarray) -> np.ndarray:
        """Convert a stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
        Return only stabilizer tableau of the resulting stabilizer state.
        
        S = [S_x | S_z] ; Lx = [Lx_x | Lx_z] ; Lz = [Lz_x | Lz_z]

        S_choi = [S_x  | 0 | S_z  | 0]
                [Lx_x | I | Lx_z | 0]
                [Lz_x | 0 | Lz_z | I]
        """
        if code.k == 0:
            return reduced_symplectic

        n = code.n
        r = reduced_symplectic.shape[0]
        k = code.k

        stab_x = reduced_symplectic[:, :n]
        stab_z = reduced_symplectic[:, n:]

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
            def _rref_no_column_swaps(matrix: np.ndarray) -> tuple[np.ndarray, int]:
                n_rows, n_cols = matrix.shape
                pivot_row = 0
                for col in range(n_cols // 2):
                    if pivot_row >= n_rows:
                        break

                    tail = matrix[pivot_row:, col]
                    pivot_offset = int(np.argmax(tail))

                    if not tail[pivot_offset]:
                        continue

                    pivot = pivot_row + pivot_offset

                    if pivot != pivot_row:
                        matrix[[pivot_row, pivot], :] = matrix[[pivot, pivot_row], :]

                    for r in range(n_rows):
                        if r != pivot_row and matrix[r, col]:
                            matrix[r, :] ^= matrix[pivot_row, :]
                    pivot_row += 1

                return matrix, pivot_row

            rre, rank_x = _rref_no_column_swaps(tableau)

            if rank_x != n:
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
        def _lc(graph: np.ndarray, q: int) -> np.ndarray | None:
            neighbors = np.flatnonzero(graph[q])

            if neighbors.size < 2:
                return None
            
            new_graph = graph.copy()

            new_graph[np.ix_(neighbors, neighbors)] ^= 1
            new_graph[neighbors, neighbors] = 0
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
        start = graph.copy()
        seen = {_canonical_key(start)}
        queue = deque([start])

        while queue:
            current_graph = queue.popleft()
            if _is_bipartite(current_graph):
                return True

            for q in range(n):
                new_graph = _lc(current_graph, q)
                
                if new_graph is None:
                    continue

                key = _canonical_key(new_graph)

                if key not in seen:
                    queue.append(new_graph)
                    seen.add(key)

        return False
    
    stab_state = _stab_code_to_stab_state(code, reduced_symplectic)
    graph_state = _stab_state_to_graph_state(stab_state)
    return _traverse_lc_orbit(graph_state)


# ----------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------

def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return mod2.rank(matrix)

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