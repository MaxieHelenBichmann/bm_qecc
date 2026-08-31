"""SAT based permutation equivalence checking for CSS Codes."""

from __future__ import annotations

import z3

from ...core.css_code import CSSCode

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


def _build_peq_css_sat_solver(
    c1: CSSCode,
    c2: CSSCode,
) -> z3.Solver:
    """Build the SAT instance."""
    solver = z3.Solver()

    n = c1.n
    rx = c1.Hx.shape[0]
    rz = c1.Hz.shape[0]
    

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
            x_column_original = c1.Hx[:, i]
            z_column_original = c1.Hz[:, i]

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
                if c2.Hx[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients_x[row * rx + contribution])

            solver.add(aux_tableau_x[row * n + q] == _xor_list(row_contributions))

    for row in range(rz):
        for q in range(n):

            row_contributions = []
            for contribution in range(rz):
                if c2.Hz[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients_z[row * rz + contribution])

            solver.add(aux_tableau_z[row * n + q] == _xor_list(row_contributions))

    return solver


def are_peq_css_sat(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence by reducing it to a SAT problem and using a SAT solver.

    The idea is to create boolean variables for the permutation and the row operations, and then add constraints that enforce that the permuted matrices of c1 are equal to the row-operated matrices of c2.
    1.) Create a auxiliary matrices for Hx and Hz that encode the column permutations of c1, and the row operations of c2.
    2.) Create boolean variables p_{i,j} for the permutation, where p_{i,j} is true if the i-th qubit of c1 is mapped to the j-th qubit of the auxiliary matrices.
    3.) Create boolean variables r_x/z_{i,j} for the row operations of the Hx or Hz matrix, where r_x/z_{i,j} is true if the j-th row is added to the i-th row in the auxiliary matrices.
    4.) Check satisfiability of the resulting formula. If it is satisfiable, then c1 and c2 are permutation equivalent, otherwise they are not.
    """
    return _build_peq_css_sat_solver(c1, c2).check() == z3.sat
