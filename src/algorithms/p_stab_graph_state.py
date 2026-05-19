"""Graph-state machinery for permutation equivalence checking."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from collections import deque

from pynauty import Graph, certificate, canon_label, autgrp

from ..core.stabilizer_code import StabilizerCode

def _stab_code_to_stab_state(code: StabilizerCode) -> np.ndarray:
    """Convert a stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    Return only stabilizer tableau of the resulting stabilizer state.
    
    S = [S_x | S_z] ; Lx = [Lx_x | Lx_z] ; Lz = [Lz_x | Lz_z]

    S_choi = [S_x  | 0 | S_z  | 0]
             [Lx_x | I | Lx_z | 0]
             [Lz_x | 0 | Lz_z | I]
    """
    # TODO: with this approach, the resulting state is dependent on the choice of logical operators, aka it die NOT solve the issue: is tableu 1 P-equivalent to tableu 2? and also NOT: is code 1 (generated solely by a given stabilizer tableau) P-equivalent to code 2 (generated solely by a different stabilizer tableau)? because the generation of logical operators is NOT unique
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

def _traverse_lc_orbit(graph1: np.ndarray, graph2: np.ndarray, n_code: int, k_code: int) -> set[tuple[int]]:
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

    n = graph1.shape[0]
    seen = set()
    queue = deque([graph1.copy()])

    permutations = set()

    while queue:
        current_graph = queue.popleft()

        permutations.update(_extract_qubit_permutations(current_graph, graph2, n_code, k_code))

        for q in range(n):
            new_graph = _lc(current_graph, q)
            key = _canonical_key(new_graph)

            if key not in seen:
                queue.append(new_graph)
                seen.add(key)

    return permutations

def _extract_qubit_permutations(adj1: np.ndarray, adj2: np.ndarray, n: int, k: int) -> list[tuple[int]]:
    def _inverse_perm(p):
        inv = [None] * len(p)
        for i, x in enumerate(p):
            inv[x] = i
        return inv

    def _compose(p, q):
        return tuple(p[q[i]] for i in range(len(q)))

    g1 = Graph(
        number_of_vertices=n+k,
        directed=False,
        adjacency_dict={ i: list(np.flatnonzero(adj1[i])) for i in range(n) },
        vertex_coloring=[set(range(n)), set(range(n, n+k))],
    )
    g2 = Graph(
        number_of_vertices=n+k,
        directed=False,
        adjacency_dict={ i: list(np.flatnonzero(adj2[i])) for i in range(n) },
        vertex_coloring=[set(range(n)), set(range(n, n+k))],
    )

    if certificate(g1) != certificate(g2):
        return []

    # get one isomorphism from g1 to g2 using the canonical labeling
    can_to_g1 = canon_label(g1)
    can_to_g2 = canon_label(g2)
    g1_to_can = _inverse_perm(can_to_g1)

    phi = _compose(can_to_g2, g1_to_can)

    # get the full automorphism group of g2
    generators, _, _, _, _ = autgrp(g2)
    gens = [tuple(gen) for gen in generators] + [tuple(_inverse_perm(gen)) for gen in generators]

    aut_g2 = {tuple(range(len(phi)))}
    queue = deque([tuple(range(len(phi)))])
    while queue:
        current = queue.popleft()
        for gen in gens:
            nxt = _compose(gen, current)
            if nxt not in aut_g2:
                aut_g2.add(nxt)
                queue.append(nxt)

    # apply the automorphisms of g2 to phi to get all isomorphisms from g1 to g2
    # isomorphisms(g1, g2) = { α ∘ φ | α ∈ Aut(g2) } with φ: g1 -> g2
    isomorphisms = [_compose(alpha, phi) for alpha in aut_g2]

    # extract the permutations of only the qubit vertices from the isomorphisms
    qubit_permutations = { tuple(isomorphism[i] for i in range(n)) for isomorphism in isomorphisms }

    return list(qubit_permutations)


def are_peq_stab_graph_state(c1: StabilizerCode, c2: StabilizerCode) -> bool:    
    """Check permutation equivalence by going over LC-equivalence, comparing graph states for isomorphism.

    For both codes, we can compute a graph state representative of their local-Clifford equivalence class:
    1.) Convert the stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism.
    2.) Convert the stabilizer state into a graph state under local Clifford operations.

    3.) Traversing the LC orbit of one graph and checking for graph isomorphism with the other representative.
    4.) Check with the found isomorphism, if the corresponding permutation of the qubits is indeed a permutation also without local clifford operations.


    The detour via graph states and local-clifford equivalence is motivated by the fact that the LC orbit of a graph state is usually much smaller than the full permutation group, and there are efficient algorithms for checking graph isomorphism, which is maybe more efficient than the brute-force checking of all permutations.
    """
    stab_state_1 = _stab_code_to_stab_state(c1)
    stab_state_2 = _stab_code_to_stab_state(c2)

    graph_state_1 = _stab_state_to_graph_state(stab_state_1, c1.n + c1.k)
    graph_state_2 = _stab_state_to_graph_state(stab_state_2, c2.n + c2.k)

    permutation_candidates = _traverse_lc_orbit(graph_state_1, graph_state_2, c1.n, c1.k)

    def _rank(A: np.ndarray) -> int:
        if A.shape[0] == 0 or A.shape[1] == 0:
            return 0
        return mod2.rank(A)

    c1_rank = _rank(c1.symplectic)

    for perm in permutation_candidates:
        perm = np.array(perm)
        perm_symplectic = np.concatenate([perm, perm + c1.n])

        if (c1_rank == _rank(c2.symplectic[:, perm_symplectic]) == _rank(np.vstack([c1.symplectic, c2.symplectic[:, perm_symplectic]]))):
            return True
    
    return False
