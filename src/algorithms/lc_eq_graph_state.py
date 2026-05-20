"""Graph-state machinery for local-Clifford equivalence checking."""

from __future__ import annotations
from itertools import product

import numpy as np
import pyzx as zx
import ldpc.mod2.mod2_numpy as mod2

from ..core.stabilizer_code import StabilizerCode

def _code_to_encoder_circuit(code) -> zx.Circuit:
    def _delete_first_row_and_qubit(tab: np.ndarray) -> np.ndarray:
        n = tab.shape[1] // 2
        return np.delete(np.delete(np.delete(tab, 0, axis=0), 0, axis=1) , n-1, axis=1)

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
                tableau[:, q + cur_n] ^= tableau[:, q]
                tableau[:, [q, q + cur_n]] = tableau[:, [q + cur_n, q]]
                elimination_gates.append(("S", (original_qubits[q],)))
                elimination_gates.append(("H", (original_qubits[q],)))
            elif x_q and not z_q:
                # X -> Z under H
                tableau[:, [q, q + cur_n]] = tableau[:, [q + cur_n, q]]
                elimination_gates.append(("H", (original_qubits[q],)))

        # 2.) make the first qubit Z (pivot) CNOT(0 -> pivot)
        z_support = np.flatnonzero(tableau[0, cur_n:])

        if len(z_support) == 0:
            raise RuntimeError("Failed to convert stabilizer row to Z support.")
        pivot = int(z_support[0])

        if pivot != 0:
            tableau[:, pivot] ^= tableau[:, 0]
            tableau[:, cur_n + 0] ^= tableau[:, cur_n + pivot]
            elimination_gates.append(("CNOT", (original_qubits[0], original_qubits[pivot])))

        # 3.) clear all other Zs in row 0 using CNOT(q -> 0)
        # control: (x_c|z_c) --CNOT--> (  x_c  |z_c^z_t)
        # target : (x_t|z_t) --CNOT--> (x_t^x_c|  z_t  )
        for q in range(1, cur_n):
            if tableau[0, cur_n + q] == 1:
                tableau[:, 0] ^= tableau[:, q]
                tableau[:, cur_n + q] ^= tableau[:, cur_n + 0]

                elimination_gates.append(
                    ("CNOT", (original_qubits[q], original_qubits[0]))
                )

        if (
            np.count_nonzero(tableau[0, :cur_n]) != 0
            or np.count_nonzero(tableau[0, cur_n:]) != 1
            or tableau[0, cur_n] != 1
        ):
            raise RuntimeError("Failed to isolate a stabilizer as a single Z.")

        # 4.) clear pivot column
        for r in range(1, tableau.shape[0]):
            if tableau[r, cur_n]:
                tableau[r] ^= tableau[0]
            if tableau[r, 0]:
                raise RuntimeError(
                    "A remaining row has X on the pivot qubit."
                )

        # 5.) remove stabilizer and qubit
        tableau = _delete_first_row_and_qubit(tableau)
        del original_qubits[0]

    # encoder = inverse elimination Cliffords
    circuit = zx.Circuit(n + k)
    circuit.initialize_qubits([True] * (n+k))
    # already prep with choi in mind
    for j in range(k):
        ref = j
        inp = k + (n-k) + j
        circuit.add_gate("HAD", ref)
        circuit.add_gate("CNOT", ref, inp)

    for name, qubits in reversed(elimination_gates):
        if name == "H":
            circuit.add_gate("HAD", qubits[0] + k)
        elif name == "S":
            # S† = Z phase 3π/2
            circuit.add_gate("ZPhase", qubits[0] + k, phase=3 / 2)
        elif name == "CNOT":
            circuit.add_gate("CNOT", qubits[0] + k, qubits[1] + k)

    return circuit

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

def _code_to_graph(code) -> np.ndarray:
    """Convert the stabilizer code into a LC-equivalent graph state with local Clifford decorations on the vertices.

    1.) Convert the code into an encoder circuit using Gaussian elimination on the tableau.
    2.) Apply the Choi-Jamiołkowski isomorphism on the circuit, by applying the Bell-state |Φ⁺⟩ on the inputs and reference output qubits, which is the same as "bending the wires" in th ZX-calculus.
    3.) Apply the resulting circuit to the initial state |0⟩^(n+k) (stabilized by the tableau [0 | I]) to get the state tableau.
    4.) Convert the state tableau into a graph state under local Clifford operations (making X invertible, bringing the tableau into the form [I | A] and extracting the adjacency matrix A)
    """
    # 1.) Code -> Encoder Circuit
    circuit = _code_to_encoder_circuit(code)

    # 2.) Encoder Circuit -> State Tableau
    n = code.n + code.k
    initial_state = np.hstack([np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)])
    for gate in circuit.gates:
        if gate.name == "HAD":
            initial_state[:, [gate.target, n + gate.target]] = initial_state[:, [n + gate.target, gate.target]]
        elif gate.name == "ZPhase":
            if gate.phase == 1 / 2:
                initial_state[:, n + gate.target] ^= initial_state[:, gate.target]
            elif gate.phase == 1:
                pass # Z has no effect on tableau
            elif gate.phase == 3 / 2:
                initial_state[:, n + gate.target] ^= initial_state[:, gate.target]
                # S† = SZ, but Z has no effect on tableau, so only apply S part to tableau
            else:
                raise ValueError(f"Unexpected Z phase {gate.phase} in encoder circuit.")
        elif gate.name == "CNOT":
            initial_state[:, gate.target] ^= initial_state[:, gate.control]
            initial_state[:, n + gate.control] ^= initial_state[:, n + gate.target]
        else:
            raise ValueError(f"Unexpected gate {gate.name} in encoder circuit.")

    # 3.) state tableau -> graph state
    graph = _stab_state_to_graph_state(initial_state)

    return graph


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
        def a_idx(i):
            return i
        def b_idx(i):
            return n + i
        def d_idx(i):
            return 3 * n + i

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

    # TODO: false assumption of the paper of Van den Nest, as Bouchet (lemma 1, in Van de Nest) requires the graph to be connected for the proof to work (which we do not guarantee, thus code broken) -> not polynomial anymore, but maybe preprocess graphs according to connected components (and their size) and run algorithm on each one separately
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
    1.) Convert the stabilizer code into a graph state under local Clifford operations.
    2.) Check if the resulting graph states are equal under local complementations, using an efficient algorithm.


    The efficient algorithm for equivalence checking of graph states runs in O(n^4) time (TODO: if graph connected!), so the overall runtime of this algorithm should be O(n^4) which is very efficient.
    """
    graph_1 = _code_to_graph(c1)
    graph_2 = _code_to_graph(c2)

    return _lc_equiv_graph_states(graph_1, graph_2, c1.n + c1.k)
