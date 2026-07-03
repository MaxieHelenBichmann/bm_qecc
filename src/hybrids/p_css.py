"""Best hybrid solution for checking whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

import multiprocessing
from collections import Counter, defaultdict
import hashlib
from itertools import permutations

import numpy as np
import numpy.typing as npt

import ldpc.mod2.mod2_numpy as mod2

import z3
from pynauty import Graph, certificate

from ..core.css_code import CSSCode

def are_peq_css(c1: CSSCode, c2: CSSCode) -> bool:
    cheap_invariants = (
        preserved_n,
        preserved_k,
        preserved_d,
        preserved_rank,
        preserved_number_zero_columns,
        preserved_number_duplicate_columns
    )

    if not all(invariant(c1, c2) for invariant in cheap_invariants):
        return False
    
    if c1.n < 1:
        return True
    
    reduced_Hx1 = _row_basis(c1.Hx)
    reduced_Hz1 = _row_basis(c1.Hz)

    reduced_Hx2 = _row_basis(c2.Hx)
    reduced_Hz2 = _row_basis(c2.Hz)

    if reduced_Hx1.shape[0] == 0 and reduced_Hz1.shape[0] == 0:
        return True
    
    if c1.n <= 4:
        return _bruteforce(reduced_Hx1, reduced_Hz1, reduced_Hx2, reduced_Hz2)

    more_expensive_invariants = (
        preserved_linear_dependencies,
        preserved_punctured_hull_weight_enumerator,
    )

    if not all(invariant(reduced_Hx1, reduced_Hz1, reduced_Hx2, reduced_Hz2) for invariant in more_expensive_invariants):
        return False
    
    return _matroid(reduced_Hx1, reduced_Hz1, reduced_Hx2, reduced_Hz2) # TODO: maybe _sat for big n?

# ----------------------------------------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------------------------------------

def preserved_n(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k

def preserved_d(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the distance is preserved, which is a necessary condition for P-equivalence."""
    return c1.x_distance == c2.x_distance and c1.z_distance == c2.z_distance

def preserved_rank(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the rank of the stabilizer tableau is preserved, which is a necessary condition for P-equivalence."""
    return _rank(c1.Hx) == _rank(c2.Hx) and _rank(c1.Hz) == _rank(c2.Hz)

def preserved_number_zero_columns(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of zero columns is preserved, which is a necessary condition for P-equivalence."""
    return int(np.count_nonzero(np.all(c1.symplectic == 0, axis=0))) == int(np.count_nonzero(np.all(c2.symplectic == 0, axis=0)))

def preserved_number_duplicate_columns(c1: CSSCode, c2: CSSCode) -> bool:
    """Check whether the number of duplicate columns is preserved, which is a necessary condition for P-equivalence."""
    def _duplicate_column(M: np.ndarray) -> list[int]:
        columns = [tuple(M[:, j].tolist()) for j in range(M.shape[1])]
        counts = Counter(columns)
        return sorted(counts.values())

    return _duplicate_column(c1.symplectic) == _duplicate_column(c2.symplectic)

def preserved_linear_dependencies(Hx1: np.ndarray, Hz1: np.ndarray, Hx2: np.ndarray, Hz2: np.ndarray) -> bool:
    """Check whether the linear dependencies between columns are preserved, which is a necessary condition for P-equivalence.
    Similar to pm_css_matroid.py"""
    def _linear_dependencies(M: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        n = M.shape[1] // 2
        
        one_columns = [ _rank(np.column_stack([M[:, q], M[:, q + n]])) for q in range(n) ]

        two_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n]])) for i in range(n) for j in range(i + 1, n) ]
        three_columns = [ _rank(np.column_stack([M[:, i], M[:, i + n], M[:, j], M[:, j + n], M[:, k], M[:, k + n]])) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n) ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))
    
    symplectic1 = np.hstack([np.vstack([Hx1, np.zeros_like(Hz1)]), np.vstack([np.zeros_like(Hx1), Hz1])])
    symplectic2 = np.hstack([np.vstack([Hx2, np.zeros_like(Hz2)]), np.vstack([np.zeros_like(Hx2), Hz2])])

    return _linear_dependencies(symplectic1) == _linear_dependencies(symplectic2)

def preserved_punctured_hull_weight_enumerator(Hx1: np.ndarray, Hz1: np.ndarray, Hx2: np.ndarray, Hz2: np.ndarray) -> bool:
    """SENDRIER - p_css_classical.py"""
    def _generator_matrix_from_parity_check(H: np.ndarray, n: int) -> np.ndarray:
        if H.size == 0 or H.shape[0] == 0:
            return np.eye(n, dtype=np.uint8)
        return _kernel_basis(H)
    
    def _compute_signatures(G1: np.ndarray, G2: np.ndarray) -> list[int]:
        def _weight_enumerator_of_hull_punctured(G: np.ndarray, col_idx: int) -> list[int]:
            Gp = np.delete(G, col_idx, axis=1)
            g_p = Gp.shape[1]

            gram = (Gp @ Gp.T) & 1

            if gram.size == 0:
                hull_basis = np.zeros((0, g_p), dtype=np.uint8)
            elif not gram.any():
                hull_basis = _row_basis(Gp)
            else:
                coeff_basis = _kernel_basis(gram)

                if coeff_basis.shape[0] == 0:
                    hull_basis = np.zeros((0, g_p), dtype=np.uint8)
                else:
                    hull_basis = _row_basis((coeff_basis @ Gp) & 1)

            h = hull_basis.shape[0]
            enumerator = [1] + [0] * g_p

            word = np.zeros(g_p, dtype=np.uint8)
            previous_gray = 0

            for t in range(1, 1 << h):
                gray = t ^ (t >> 1)
                changed = gray ^ previous_gray
                row_idx = changed.bit_length() - 1

                word ^= hull_basis[row_idx]
                enumerator[int(word.sum())] += 1

                previous_gray = gray

            return enumerator

        def _combine_invariants(inv_hx: list[int], inv_hz: list[int]) -> int:
            payload = (
                ",".join(map(str, inv_hx))
                + "|"
                + ",".join(map(str, inv_hz))
            ).encode("ascii")
            return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")

        invariants = []

        for col_idx in range(G1.shape[1]):
            inv1 = _weight_enumerator_of_hull_punctured(G1, col_idx)
            inv2 = _weight_enumerator_of_hull_punctured(G2, col_idx)

            invariants.append(_combine_invariants(inv1, inv2))

        return invariants
    
    def _partition_columns_by_invariants(invariants: list[int]) -> dict[int, list[int]]:
        partition = defaultdict(list)
        for idx, inv in enumerate(invariants):
            partition[inv].append(idx)
        return {k: v for k, v in sorted(partition.items())}

    n = Hx1.shape[1]

    Gx1 = _generator_matrix_from_parity_check(Hx1, n)
    Gz1 = _generator_matrix_from_parity_check(Hz1, n)
    Gx2 = _generator_matrix_from_parity_check(Hx2, n)
    Gz2 = _generator_matrix_from_parity_check(Hz2, n)

    signatures_c1 = _compute_signatures(Gx1, Gz1)
    signatures_c2 = _compute_signatures(Gx2, Gz2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    if partition_c1.keys() != partition_c2.keys():
        return False
    if any(len(partition_c1[k]) != len(partition_c2[k]) for k in partition_c1):
        return False

    for key1, key2 in zip(partition_c1.keys(), partition_c2.keys()):
        if key1 != key2:
            return False
        if len(partition_c1[key1]) != len(partition_c2[key2]):
            return False
        
    return True

# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------

def _bruteforce(Hx1, Hz1, Hx2, Hz2) -> bool:
    """p_css_bruteforce.py"""
    n = Hx1.shape[1]

    hx_rank = Hx1.shape[0]
    hz_rank = Hz1.shape[0]

    for perm in permutations(range(n)):
        if hx_rank and hx_rank != mod2.rank(np.vstack([Hx1, Hx2[:, perm]])):
            continue
        if hz_rank and hz_rank != mod2.rank(np.vstack([Hz1, Hz2[:, perm]])):
            continue
        return True

    return False

def _sat(Hx1: np.ndarray, Hz1: np.ndarray, Hx2: np.ndarray, Hz2: np.ndarray) -> bool:
    """pm_css_sat.py"""
    solver = z3.Solver()

    n = Hx1.shape[1]
    rx = Hx1.shape[0]
    rz = Hz1.shape[0]
    

    # permutations
    aux_tableau_x = [z3.Bool(f'aux_x_{row}_{col}') for row in range(rx) for col in range(n)]
    aux_tableau_z = [z3.Bool(f'aux_z_{row}_{col}') for row in range(rz) for col in range(n)]


    permutation_variables = [z3.Bool(f'p_{i}_{j}') for i in range(n) for j in range(n)]

    for i in range(n):
        solver.add(_exactly_one([permutation_variables[i * n + j] for j in range(n)]))
    for j in range(n):
        solver.add(_exactly_one([permutation_variables[i * n + j] for i in range(n)]))

    for i in range(n):
        for j in range(n):
            x_column_original = Hx1[:, i]
            z_column_original = Hz1[:, i]

            x_column_permuted = [aux_tableau_x[row * n + j] for row in range(rx)]
            z_column_permuted = [aux_tableau_z[row * n + j] for row in range(rz)]

            solver.add(z3.Implies(permutation_variables[i * n + j], z3.And(_elementwise_map(x_column_original, x_column_permuted), _elementwise_map(z_column_original, z_column_permuted))))

    # row operations
    row_operation_coefficients_x = [z3.Bool(f'r_x_{i}_{j}') for i in range(rx) for j in range(rx)]
    row_operation_coefficients_z = [z3.Bool(f'r_z_{i}_{j}') for i in range(rz) for j in range(rz)]

    for row in range(rx):
        for q in range(n):

            row_contributions = []
            for contribution in range(rx):
                if Hx2[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients_x[row * rx + contribution])

            solver.add(aux_tableau_x[row * n + q] == _xor_list(row_contributions))

    for row in range(rz):
        for q in range(n):

            row_contributions = []
            for contribution in range(rz):
                if Hz2[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients_z[row * rz + contribution])

            solver.add(aux_tableau_z[row * n + q] == _xor_list(row_contributions))

    return solver.check() == z3.sat

def _matroid(Hx1: np.ndarray, Hz1: np.ndarray, Hx2: np.ndarray, Hz2: np.ndarray) -> bool:
    """pm_css_matroid.py"""
    def _circuits_binary_matroid(A: npt.NDArray[np.int8]) -> list[int]:
        K = _kernel_basis(A)

        k, _ = K.shape
        row_supports = [_row_support_as_mask(row) for row in K]
        circuits_by_size: list[list[int]] = [[] for _ in range(A.shape[1] + 1)]

        support = 0
        previous_gray = 0
        for mask in range(1, 1 << k):
            gray = mask ^ (mask >> 1)
            changed = gray ^ previous_gray
            support ^= row_supports[changed.bit_length() - 1]
            previous_gray = gray

            if not support:
                continue

            support_size = support.bit_count()

            if any(
                (circuit & support) == circuit
                for size in range(1, support_size + 1)
                for circuit in circuits_by_size[size]
            ):
                continue

            for size in range(support_size + 1, len(circuits_by_size)):
                if not circuits_by_size[size]:
                    continue
                circuits_by_size[size] = [
                    circuit
                    for circuit in circuits_by_size[size]
                    if (support & circuit) != support
                ]

            circuits_by_size[support_size].append(support)

        return [
            circuit
            for circuits in circuits_by_size
            for circuit in sorted(circuits)
        ]


    def _graph_from_circuits(n: int, circuits_hx: list[int], circuits_hz: list[int]) -> Graph:
        adj = defaultdict(list)

        def _add_edges_from_circuits(circuits: list[int], offset: int) -> None:
            for i, circuit in enumerate(circuits):
                circuit_vertex = offset + i
                for q in _iter_mask_bits(circuit):
                    adj[q].append(circuit_vertex)
                    adj[circuit_vertex].append(q)

        n_hx = len(circuits_hx)
        n_hz = len(circuits_hz)

        hx_offset = n
        hz_offset = n + n_hx

        _add_edges_from_circuits(circuits_hx, hx_offset)
        _add_edges_from_circuits(circuits_hz, hz_offset)

        return Graph(
            number_of_vertices=n + n_hx + n_hz,
            directed=False,
            adjacency_dict=adj,
            vertex_coloring=[
                set(range(n)),
                set(range(hx_offset, hx_offset + n_hx)),
                set(range(hz_offset, hz_offset + n_hz))
            ]
        )

    n = Hx1.shape[1]

    circuits_c1_hx = _circuits_binary_matroid(Hx1)
    circuits_c1_hz = _circuits_binary_matroid(Hz1)

    graph_c1 = _graph_from_circuits(n, circuits_c1_hx, circuits_c1_hz)

    circuits_c2_hx = _circuits_binary_matroid(Hx2)
    circuits_c2_hz = _circuits_binary_matroid(Hz2)

    graph_c2 = _graph_from_circuits(n, circuits_c2_hx, circuits_c2_hz)

    return certificate(graph_c1) == certificate(graph_c2)

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

def _row_support_as_mask(row: npt.NDArray[np.uint8]) -> int:
    support = 0
    for col in np.flatnonzero(row):
        support |= 1 << int(col)
    return support

def _iter_mask_bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit
