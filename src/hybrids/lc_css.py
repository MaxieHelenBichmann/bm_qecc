"""Best hybrid solution for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import z3
from collections import deque
from itertools import product

from typing import TYPE_CHECKING

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

from ..core.stabilizer_code import StabilizerCode


def is_lceq_css(code: StabilizerCode) -> bool:
    """Check whether the stabilizer code is local-clifford-equivalent to a CSS code."""
    if code.n < 1:
        return True

    reduced_symplectic = _row_basis(code.symplectic)

    if code.n < 4:
        return _bruteforce(reduced_symplectic)

    return _sat(reduced_symplectic)

# ----------------------------------------------------------------------------------------------------
# algorithms
# ----------------------------------------------------------------------------------------------------
LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")

def _bruteforce(tableau) -> bool:
    """lc_css_bruteforce.py"""
    r, n = tableau.shape[0], tableau.shape[1] // 2

    def apply_lc(tableau: npt.NDArray[np.int8], lc: str, qubit: int):
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

    for action in product(LOCAL_CLIFFORDS, repeat=n):
        lc_tableau = tableau.copy()

        for qubit, lc in enumerate(action):
            apply_lc(lc_tableau, lc, qubit)

        if _rank(lc_tableau[:, :n]) + _rank(lc_tableau[:, n:]) == r:
            return True

    return False


def _sat(tableau: npt.NDArray[np.int8]) -> bool:
    """lc_css_sat.py"""
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

    r, n = tableau.shape[0], tableau.shape[1] // 2
    k = n - r

    # cliffords
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
        x_column_original = tableau[:, i]
        z_column_original = tableau[:, i + n]
        x_z_column_original = (x_column_original + z_column_original) % 2
        zero_column_original = np.zeros_like(x_column_original)

        x_column_aux = [aux_tableau[row * (2*n) + i] for row in range(r)]
        z_column_aux = [aux_tableau[row * (2*n) + i + n] for row in range(r)]

        # I^(-1) P_x I : (x, z) -> (x, 0)
        solver.add(z3.Implies(local_clifford_variables[i]["I"], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(zero_column_original, z_column_aux))))

        # H^(-1) P_x H : (x, z) -> (0, z)
        solver.add(z3.Implies(local_clifford_variables[i]["H"], z3.And(_elementwise_map(zero_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

        # S^(-1) P_x S : (x, z) -> (x, x)
        solver.add(z3.Implies(local_clifford_variables[i]["S"], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux))))

        # (HS)^(-1) P_x (HS)  : (x, z) -> (z, z)
        solver.add(z3.Implies(local_clifford_variables[i]["HS"], z3.And(_elementwise_map(z_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

        # (SH)^(-1) P_x (SH) : (x, z) -> (0, x + z)
        solver.add(z3.Implies(local_clifford_variables[i]["SH"], z3.And(_elementwise_map(zero_column_original, x_column_aux), _elementwise_map(x_z_column_original, z_column_aux))))

        # (HSH)^(-1) P_x (HSH) : (x, z) -> (x + z, 0)
        solver.add(z3.Implies(local_clifford_variables[i]["HSH"], z3.And(_elementwise_map(x_z_column_original, x_column_aux), _elementwise_map(zero_column_original, z_column_aux))))

    # row operations
    row_operation_coefficients = [z3.Bool(f'r_{i}_{j}') for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):

            row_contributions = []
            for contribution in range(r):
                if tableau[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients[row * r + contribution])

            solver.add(aux_tableau[row * (2*n) + q] == _xor_list(row_contributions))

    return solver.check() == z3.sat

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