"""SAT based approach for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import z3
import numpy as np

from ...core.stabilizer_code import StabilizerCode

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


def is_lceq_css_sat(code: StabilizerCode) -> bool:
    """Check local-clifford equivalence to a CSS code by reducing it to a SAT problem and using a SAT solver.
    
    The idea is to create boolean variables for the local clifford operations and the row operations.
    It is used that the projection of the tableau to the only-x-space has to be a subset of the stabilizer, if the code is CSS.
    1.) Create a auxiliary tableau that encodes the projected local clifford operations and row operations of the code.
    2.) Create boolean variables c_{i,j} for the projected local cliffords, where c_{i,j} is true if the i-th projected local clifford operation (I, H, S, HS, SH, HSH) is applied to the j-th qubit of the code in the auxiliary tableau.
    3.) Create boolean variables r_{i,j} for the row operations, where r_{i,j} is true if the j-th row of the code is added to the i-th row in the auxiliary tableau.
    4.) Check satisfiability of the resulting formula. If it is satisfiable, then the code is local clifford equivalent to a CSS code, otherwise it is not.
    """
    solver = z3.Solver()

    n = code.n
    k = code.k
    r = n - k # assume that tableau is minimal and has no dependent rows, and both tableaus have the same rank

    # permutations
    aux_tableau = [z3.Bool(f'aux_{row}_{col}') for row in range(r) for col in range(2*n)]
    local_clifford_variables = [z3.Bool(f'c_{c}_{i}') for i in range(n) for c in range(6)]

    for i in range(n):
        solver.add(_exactly_one([local_clifford_variables[i * 6 + j] for j in range(6)]))

    for i in range(n):
        x_column_original = code.symplectic[:, i]
        z_column_original = code.symplectic[:, i + n]
        x_z_column_original = (x_column_original + z_column_original) % 2
        zero_column_original = np.zeros_like(x_column_original)

        x_column_aux = [aux_tableau[row * (2*n) + i] for row in range(r)]
        z_column_aux = [aux_tableau[row * (2*n) + i + n] for row in range(r)]

        # I^(-1) P_x I : (x, z) -> (x, 0)
        solver.add(z3.Implies(local_clifford_variables[i * 6 + 0], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(zero_column_original, z_column_aux))))

        # H^(-1) P_x H : (x, z) -> (0, z)
        solver.add(z3.Implies(local_clifford_variables[i * 6 + 1], z3.And(_elementwise_map(zero_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

        # S^(-1) P_x S : (x, z) -> (x, x)
        solver.add(z3.Implies(local_clifford_variables[i * 6 + 2], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux))))

        # (HS)^(-1) P_x (HS)  : (x, z) -> (z, z)
        solver.add(z3.Implies(local_clifford_variables[i * 6 + 3], z3.And(_elementwise_map(z_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

        # (SH)^(-1) P_x (SH) : (x, z) -> (0, x + z)
        solver.add(z3.Implies(local_clifford_variables[i * 6 + 4], z3.And(_elementwise_map(zero_column_original, x_column_aux), _elementwise_map(x_z_column_original, z_column_aux))))

        # (HSH)^(-1) P_x (HSH) : (x, z) -> (x + z, 0)
        solver.add(z3.Implies(local_clifford_variables[i * 6 + 5], z3.And(_elementwise_map(x_z_column_original, x_column_aux), _elementwise_map(zero_column_original, z_column_aux))))

    # row operations
    row_operation_coefficients = [z3.Bool(f'r_{i}_{j}') for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):

            row_contributions = []
            for contribution in range(r):
                if code.symplectic[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients[row * r + contribution])

            solver.add(aux_tableau[row * (2*n) + q] == _xor_list(row_contributions))

    return solver.check() == z3.sat
