"""SAT based permutation equivalence checking for CSS Codes."""

from __future__ import annotations

import z3

from ..core.css_code import CSSCode

def _elementwise_map(normal_bool, variables):
    elem = []
    for k in range(len(normal_bool)):
        if normal_bool[k] == 1:
            elem.append(variables[k])
        else:
            elem.append(z3.Not(variables[k]))
    return z3.And(elem)

def _exactly_one(variables):

    def _at_least_one():
        return z3.Or(variables)

    def _at_most_one():
        return z3.And([
            z3.Or(z3.Not(variables[i]), z3.Not(variables[j]))
            for i in range(len(variables))
            for j in range(i + 1, len(variables))
        ])

    return z3.And(_at_least_one(), _at_most_one())

def _xor_list(variables):
    if len(variables) == 0:
        return z3.BoolVal(False)
    if len(variables) == 1:
        return variables[0]
    return z3.Xor(variables[0], _xor_list(variables[1:]))

def are_peq_css_sat(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence by reducing it to a SAT problem and using a SAT solver.

    The idea is to create boolean variables for the permutation and the row operations, and then add constraints that enforce that the permuted matrices of c1 are equal to the row-operated matrices of c2.
    1.) Create a auxiliary matrices for Hx and Hz that encode the column permutations of c1, and the row operations of c2.
    2.) Create boolean variables p_{i,j} for the permutation, where p_{i,j} is true if the i-th qubit of c1 is mapped to the j-th qubit of the auxiliary matrices.
    3.) Create boolean variables r_x/z_{i,j} for the row operations of the Hx or Hz matrix, where r_x/z_{i,j} is true if the j-th row is added to the i-th row in the auxiliary matrices.
    4.) Check satisfiability of the resulting formula. If it is satisfiable, then c1 and c2 are permutation equivalent, otherwise they are not.
    """
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

    return solver.check() == z3.sat
