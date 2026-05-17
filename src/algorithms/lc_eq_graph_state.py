"""Graph-state machinery for local-Clifford equivalence checking."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
from itertools import product

from ..core.stabilizer_code import StabilizerCode

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

def _stab_state_to_graph_state(tableau: np.ndarray, n: int) -> np.ndarray:
    """Convert a stabilizer state into a graph state under local Clifford operations.
    Returns the adjacency matrix of the graph state."""
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

                # swap row with potential pivot up
                if pivot != pivot_row:
                    matrix[[pivot_row, pivot], :] = matrix[[pivot, pivot_row], :]

                # make pivot column a unit vector
                for r in range(n_rows):
                    if r != pivot_row and matrix[r, col]:
                        matrix[r, :] ^= matrix[pivot_row, :]
                pivot_row += 1

            return matrix

        rre = _rref_no_column_swaps(tableau)

        if _rank(rre[:, :n]) != n:
            raise ValueError("X part of the tableau is not full rank, something went wrong.")

        return rre[:, n:]

    def _remove_diagonal(tableau: np.ndarray) -> np.ndarray:
        """Basically apply S gate on all qubits to remove self-loops in the graph state."""
        np.fill_diagonal(tableau, 0)
        return tableau
    
    state = _make_X_invertible(tableau)
    gamma = _extract_adjacency_matrix(state)
    gamma = _remove_diagonal(gamma)

    if not np.array_equal(gamma, gamma.T):
        raise ValueError("Extracted adjacency matrix is not symmetric, something went wrong.")

    return gamma

def _lc_equiv_graph_states(g1: np.ndarray, g2: np.ndarray, n : int) -> bool:
    """Check if two graph states are equivalent under local complementations using an efficient algorithm that considers a linear system of equations."""

    def _build_lse():
        """Build the matrix A for the following LSE
        ( sum_{i=0}^{n-1} g1[i,j] * g2[i,k] * c_i ) + g1[j,k] * a_k + g2[j,k] * d_j + delta[j,k] * b_j = 0 
        with n^2 equations for j,k = 0...n-1 and the following 4n unknowns:
            [a_0,...,a_{n-1},
            b_0,...,b_{n-1},
            c_0,...,c_{n-1},
            d_0,...,d_{n-1}]
        """
        A = np.zeros((n * n, 4 * n), dtype=np.uint8)
        def a_idx(i): return i
        def b_idx(i): return n + i
        def c_idx(i): return 2 * n + i
        def d_idx(i): return 3 * n + i

        row = 0
        for j in range(n):
            for k in range(n):
                # sum_{i=0}^{n-1} g1[i,j] * g2[i,k] * c_i
                A[row, 2 * n:3 * n] = g1[j, :] & g2[:, k]
                # g1[j, k] * a_k
                A[row, a_idx(k)] ^= g1[j, k]
                # g2[j, k] * d_j
                A[row, d_idx(j)] ^= g2[j, k]
                # delta[j, k] * b_j
                if j == k:
                    A[row, b_idx(j)] ^= 1
                row += 1
        return A

    def _satisfy_constraints(x : np.ndarray) -> bool:
        """Check that the solution x of the LSE also satisfies the following constraints on the unknowns for i = 0...n-1:
        a_i d_i + b_i c_i = 1
        """
        x = np.asarray(x, dtype=np.uint8) % 2
        a = x[0:n]
        b = x[n:2*n]
        c = x[2*n:3*n]
        d = x[3*n:4*n]
        dets = (a & d) ^ (b & c)
        return np.all(dets == 1)

    def _kernel_basis(A: np.ndarray) -> np.ndarray:
        A = (np.asarray(A) & 1).astype(np.uint8)
        K = mod2.nullspace(A)
        if hasattr(K, "toarray"):
            K = K.toarray()
        K = (np.asarray(K) & 1).astype(np.uint8)
        if K.size == 0:
            return np.zeros((0, A.shape[1]), dtype=np.uint8)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        if K.shape[1] != A.shape[1]:
            raise ValueError(
                "Kernel basis must have the same number of columns as the input matrix."
            )
        return K


    A = _build_lse()
    V = _kernel_basis(A)

    dim = V.shape[0]

    if dim == 0: # trivial nullspace
        return False
    
    # TODO: false assumption of the paper of Van den Nest, as Bouchet (lemma Van de Nest uses) requires the graph to be connected for the proof to work (which we do not guarantee, thus code broken) -> not polynomial anymore
    # if dim > 4:
    #    for i in range(dim):
    #        for j in range(i, dim):
    #            x = V[i] ^ V[j]
    #            if _satisfy_constraints(x):
    #                return True
 
    for coeffs in product([0, 1], repeat=dim):
        x = np.zeros(4 * n, dtype=np.uint8)
        for bit, basis_vec in zip(coeffs, V):
            if bit:
                x ^= basis_vec
            if _satisfy_constraints(x):
                return True

    return False



def are_lceq_graph_state(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check Local-Clifford equivalence by comparing graph states with an efficient algorithm of Van den Nest, Dehaene, De Moor.

    For both codes, we can compute a graph state representative of their local-Clifford equivalence class:
    1.) Convert the stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    2.) Convert the stabilizer state into a graph state under local Clifford operations.

    3.) Check if the resulting graph states are equal under local complementations, using an efficient algorithm.


    The conversion from stabilizer code into a graph state can be done in O(n^3) time an the efficient algorithm for equivalence checking of graph states runs in O(n^4) time, so the overall runtime of this algorithm should be O(n^3) which is very efficient.
    """
    stab_state_1 = _stab_code_to_stab_state(c1)
    stab_state_2 = _stab_code_to_stab_state(c2)
    
    graph_state_1 = _stab_state_to_graph_state(stab_state_1, c1.n + c1.k)
    graph_state_2 = _stab_state_to_graph_state(stab_state_2, c2.n + c2.k)

    return _lc_equiv_graph_states(graph_state_1, graph_state_2, c1.n + c1.k)
