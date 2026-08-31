"""Best hybrid solution for checking whether two stabilizer codes are LC-equivalent, including diagnostic information."""

from __future__ import annotations

import z3
from collections import deque
from itertools import product, combinations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from collections import defaultdict
from pynauty import Graph, certificate

from src.core.stabilizer_code import StabilizerCode

def are_lceq(c1: StabilizerCode, c2: StabilizerCode) -> tuple[bool, str]:
    """Check whether two stabilizer codes are local-clifford-equivalent.

    Returns: A tuple of (is_equivalent, diagnostic_info) where is_equivalent is a boolean indicating whether the codes are equivalent, and diagnostic_info is a string providing information about the equivalence check.
    """
    cheap_invariants = (
        preserved_n,
        preserved_k,
    )

    if not all(invariant(c1, c2) for invariant in cheap_invariants):
        return False, "CI"
    
    if c1.n < 1:
        return True, ""
    
    reduced_symplectic_1 = _row_basis(c1.symplectic)
    reduced_symplectic_2 = _row_basis(c2.symplectic)

    if c1.k < 2:
        return _lse(c1, c2, reduced_symplectic_1, reduced_symplectic_2)

    if c1.n <= 30:
        if not preserved_low_degree_local_invariant(reduced_symplectic_1, reduced_symplectic_2):
            return False, "EI"
    
    return _sat(reduced_symplectic_1, reduced_symplectic_2)

# ----------------------------------------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------------------------------------
def preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.k == c2.k

def preserved_low_degree_local_invariant(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check whether the r = 2 local invariant is preserved, which is a necessary condition for LC-equivalence.

    for each A ⊆ {1, ..., n}: 
        d(A) = dim({s in S | supp(s) ⊆ A})
            =!= 
        d'(A) = dim({s in S' | supp(s) ⊆ A})
    
    Reference for this invariant: 
    - Maarten Van den Nest, Bart De Moor: Local Invariants of Stabilizer Codes
    """
    print("EI")
    n = c1.shape[1] // 2
    stabilizer_rank = c1.shape[0]

    def _supp_subcode_dim(code: np.ndarray, subset: tuple[int, ...]) -> int:
        """
        d(A) = dim({s in S | supp(s) ⊆ A}) = dim({y in F_2 | supp(yG) ⊆ A}) 
             = rank(G) - rank(G|_(A^c)})

        -> supp(yG) ⊆ A means "outside of A (aka all qubit not in A), there can only be identity", aka all columns outside of A must be zero 
        -> y (G|_(A^c)) = 0 for the restricted matrix outside if A 
        -> {y in F_2 | supp(yG) ⊆ A} = kernel of G|_(A^c) -> dim ker = n - rank
        """
        G = np.asarray(code, dtype=np.uint8) & 1

        A = set(subset)
        outside = [i for i in range(n) if i not in A]

        cols = outside + [i + n for i in outside]

        if not cols:
            return stabilizer_rank
        
        restricted = G[:, cols]
        return stabilizer_rank - _rank(restricted)
    
    max_subset_size = 2
    for a in range(max_subset_size + 1):
        for subset in combinations(range(n), a):
            if _supp_subcode_dim(c1, subset) != _supp_subcode_dim(c2, subset):
                return False

    return True


# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------
LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")

def _lse(c1: StabilizerCode, c2: StabilizerCode, reduced_symplectic_1: np.ndarray, reduced_symplectic_2: np.ndarray) -> tuple[bool, str]:
    """lc_stb_lse.py"""
    print("LSE")

    def _stab_code_to_stab_state(code: StabilizerCode, reduced_symplectic: np.ndarray) -> np.ndarray:
        """Convert a stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
        Return only stabilizer tableau of the resulting stabilizer state.
        
        S = [S_x | S_z] ; Lx = [Lx_x | Lx_z] ; Lz = [Lz_x | Lz_z]

        S_choi = [S_x  | 0 | S_z  | 0]
                [Lx_x | I | Lx_z | 0]
                [Lz_x | 0 | Lz_z | I]
        """
        if code.k == 0:
            return reduced_symplectic.copy()

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
            for i in range(tableau.shape[0]):
                if tableau[i,i] == 1:
                    tableau[i,i] = 0

        state = _make_X_invertible(tableau)
        gamma = _extract_adjacency_matrix(state)
        _remove_diagonal(gamma)

        if not np.array_equal(gamma, gamma.T):
            raise ValueError("Extracted adjacency matrix is not symmetric, something went wrong.")

        return gamma

    def _extract_connected_components(g: np.ndarray) -> list[list[int]]:
        n = g.shape[0]
        connected_components : list[list[int]] = []
        seen : set[int] = set()

        while len(seen) < n:
            start = next(i for i in range(n) if i not in seen)
            comp = []

            queue = deque([start])
            seen.add(start)

            while queue:
                cur : int = queue.popleft()
                comp.append(cur)

                for neighbor in g[cur, :].nonzero()[0]:
                    neighbor = int(neighbor)

                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)

            connected_components.append(sorted(comp))

        return connected_components

    def _lc_equiv_connected(g1: np.ndarray, g2: np.ndarray, n : int) -> bool:
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

        A = _build_lse()
        V = _kernel_basis(A)

        dim = V.shape[0]

        if dim == 0: # trivial nullspace
            return False

        if dim > 4:
            for i in range(dim):
                for j in range(i, dim):
                    x = V[i] ^ V[j]
                    if _satisfy_constraints(x):
                        return True
        else:
            for coeffs in product([0, 1], repeat=dim):

                x = np.zeros(4 * n, dtype=np.uint8)
                for bit, basis_vec in zip(coeffs, V):
                    if bit:
                        x ^= basis_vec

                if _satisfy_constraints(x):
                    return True

        return False

    def _lc_equiv_graph_states(graph_1: np.ndarray, graph_2: np.ndarray) -> bool:
        connected_components_g1 = sorted(tuple(comp) for comp in _extract_connected_components(graph_1))
        connected_components_g2 = sorted(tuple(comp) for comp in _extract_connected_components(graph_2))

        if connected_components_g1 != connected_components_g2:
            return False

        for comp in connected_components_g1:
            comp_idx = list(comp)

            if not _lc_equiv_connected(graph_1[np.ix_(comp_idx, comp_idx)],graph_2[np.ix_(comp_idx, comp_idx)],len(comp_idx)):
                return False

        return True

    stab_state1 = _stab_code_to_stab_state(c1, reduced_symplectic_1)
    stab_state2 = _stab_code_to_stab_state(c2, reduced_symplectic_2)

    graph_state1 = _stab_state_to_graph_state(stab_state1)
    graph_state2 = _stab_state_to_graph_state(stab_state2)

    return _lc_equiv_graph_states(graph_state1, graph_state2), "LSE"



def _sat(reduced_symplectic_1: np.ndarray, reduced_symplectic_2: np.ndarray) -> tuple[bool, str]:
    """lc_stb_sat.py"""
    print("SAT")
    def _elementwise_map(normal_bool, variables):
        return z3.And([
            v if bit == 1 else z3.Not(v)
            for bit, v in zip(normal_bool, variables)
        ])

    def _exactly_one(variables):
        return z3.PbEq([(v, 1) for v in variables], 1)

    def _xor_list(variables):
        acc = z3.BoolVal(False)
        for v in variables:
            acc = z3.Xor(acc, v)
        return acc

    solver = z3.Solver()

    n = reduced_symplectic_1.shape[1] // 2
    r = reduced_symplectic_1.shape[0]

    # local cliffords
    aux_tableau = [z3.Bool(f'aux_{row}_{col}') for row in range(r) for col in range(2*n)]

    local_clifford_variables = [
        {
            operation: z3.Bool(f"lc_{qubit}_{operation}")
            for operation in LOCAL_CLIFFORDS
        }
        for qubit in range(n)
    ]

    for qubit_variables in local_clifford_variables:
        solver.add(_exactly_one(qubit_variables.values()))

    for i in range(n):
        x_column_original = reduced_symplectic_1[:, i]
        z_column_original = reduced_symplectic_1[:, i + n]
        x_z_column_original = (x_column_original + z_column_original) % 2

        x_column_aux = [aux_tableau[row * (2*n) + i] for row in range(r)]
        z_column_aux = [aux_tableau[row * (2*n) + i + n] for row in range(r)]

        # I : (x, z) -> (x, z)
        solver.add(z3.Implies(local_clifford_variables[i]["I"], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

        # H : (x, z) -> (z, x)
        solver.add(z3.Implies(local_clifford_variables[i]["H"], z3.And(_elementwise_map(z_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux))))

        # S : (x, z) -> (x, x + z)
        solver.add(z3.Implies(local_clifford_variables[i]["S"], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(x_z_column_original, z_column_aux))))

        # HS : (x, z) -> (x + z, x)
        solver.add(z3.Implies(local_clifford_variables[i]["HS"], z3.And(_elementwise_map(x_z_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux))))

        # SH : (x, z) -> (z, x + z)
        solver.add(z3.Implies(local_clifford_variables[i]["SH"], z3.And(_elementwise_map(z_column_original, x_column_aux), _elementwise_map(x_z_column_original, z_column_aux))))

        # HSH : (x, z) -> (x + z, z)
        solver.add(z3.Implies(local_clifford_variables[i]["HSH"], z3.And(_elementwise_map(x_z_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

    # row operations
    row_operation_coefficients = [z3.Bool(f'r_{i}_{j}') for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):

            row_contributions = []
            for contribution in range(r):
                if reduced_symplectic_2[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients[row * r + contribution])

            solver.add(aux_tableau[row * (2*n) + q] == _xor_list(row_contributions))

    return solver.check() == z3.sat, "SAT"


def _graph_iso(reduced_symplectic_1: np.ndarray, reduced_symplectic_2: np.ndarray) -> tuple[bool, str]:
    """lc_stb_graph_iso.py"""
    print("GI")
    def _graph_from_code(reduced_symplectic: np.ndarray) -> Graph:
        n = reduced_symplectic.shape[1] // 2
        r = reduced_symplectic.shape[0]

        adj_dict = defaultdict(list)

        for mask in range(0, 1 << r):
            group_element_vertex = 3 * n + mask
            x = np.zeros(2*n, dtype=np.int8)

            for i in range(r):
                if (mask >> i) & 1:
                    x ^= reduced_symplectic[i]

            x_part = x[:n]
            z_part = x[n:]

            for q in range(n):
                if x_part[q] == 1 and z_part[q] == 0: # X contribution
                    adj_dict[3 * q].append(group_element_vertex)
                    adj_dict[group_element_vertex].append(3 * q)

                elif x_part[q] == 0 and z_part[q] == 1: # Z contribution
                    adj_dict[3 * q + 1].append(group_element_vertex)
                    adj_dict[group_element_vertex].append(3 * q + 1)

                elif x_part[q] == 1 and z_part[q] == 1: # Y contribution
                    adj_dict[3 * q + 2].append(group_element_vertex)
                    adj_dict[group_element_vertex].append(3 * q + 2)

        pauli_vertex_colors = [set(range(3 * q, 3 * q + 3)) for q in range(n)]
        stabilizer_group_vertices = set(range(3 * n, 3 * n + 2 ** r))

        return Graph(number_of_vertices=n * 3 + 2 ** r,
                    directed=False,
                    vertex_coloring=[*pauli_vertex_colors, stabilizer_group_vertices],
                    adjacency_dict=adj_dict)
    
    graph_1 = _graph_from_code(reduced_symplectic_1)
    cert1 = certificate(graph_1)
    del graph_1

    graph_2 = _graph_from_code(reduced_symplectic_2)
    cert2 = certificate(graph_2)
    del graph_2

    return cert1 == cert2, "GI"


# ----------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------

def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return mod2.rank(matrix)

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
