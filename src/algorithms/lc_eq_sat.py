"""SAT based local clifford equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

import z3

from ..core.stabilizer_code import StabilizerCode

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


def are_lceq_sat(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check local-clifford equivalence by reducing it to a SAT problem and using a SAT solver.

    The idea is to create boolean variables for the local clifford operations and the row operations, and then add constraints that enforce that the local-clifford tableau of c1 is equal to the row-operated tableau of c2.
    1.) Create a auxiliary tableau that encodes the local clifford operations of c1, and the row operations of c2.
    2.) Create boolean variables c_{i,j} for the local cliffords, where c_{i,j} is true if the i-th local clifford operation (I, H, S, HS, SH, HSH)is applied to the j-th qubit of c1 in the auxiliary tableau.
    3.) Create boolean variables r_{i,j} for the row operations, where r_{i,j} is true if the j-th row is added to the i-th row in the auxiliary tableau.
    4.) Check satisfiability of the resulting formula. If it is satisfiable, then c1 and c2 are local clifford equivalent, otherwise they are not.
    """
    solver = z3.Solver()

    n = c1.n
    k = c1.k
    r = n - k # assume that tableau is minimal and has no dependent rows, and both tableaus have the same rank

    # permutations
    aux_tableau = [z3.Bool(f'aux_{row}_{col}') for row in range(r) for col in range(2*n)]
    local_clifford_variables = [z3.Bool(f'c_{c}_{i}') for c in range(6) for i in range(n)]

    for i in range(n):
        solver.add(_exactly_one([local_clifford_variables[i * n + j] for j in range(6)]))

    for i in range(n):
        x_column_original = c1.symplectic[:, i]
        z_column_original = c1.symplectic[:, i + n]
        x_z_column_original = (x_column_original + z_column_original) % 2

        x_column_aux = [aux_tableau[row * (2*n) + i] for row in range(r)]
        z_column_aux = [aux_tableau[row * (2*n) + i + n] for row in range(r)]

        # I : (x, z) -> (x, z)
        solver.add(z3.Implies(local_clifford_variables[i * n + 0], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

        # H : (x, z) -> (z, x)
        solver.add(z3.Implies(local_clifford_variables[i * n + 1], z3.And(_elementwise_map(z_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux))))

        # S : (x, z) -> (x, x + z)
        solver.add(z3.Implies(local_clifford_variables[i * n + 2], z3.And(_elementwise_map(x_column_original, x_column_aux), _elementwise_map(x_z_column_original, z_column_aux))))

        # HS : (x, z) -> (x + z, x)
        solver.add(z3.Implies(local_clifford_variables[i * n + 3], z3.And(_elementwise_map(x_z_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux))))

        # SH : (x, z) -> (z, x + z)
        solver.add(z3.Implies(local_clifford_variables[i * n + 4], z3.And(_elementwise_map(z_column_original, x_column_aux), _elementwise_map(x_z_column_original, z_column_aux))))

        # HSH : (x, z) -> (x + z, z)
        solver.add(z3.Implies(local_clifford_variables[i * n + 5], z3.And(_elementwise_map(x_z_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux))))

    # row operations
    row_operation_coefficients = [z3.Bool(f'r_{i}_{j}') for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):

            row_contributions = []
            for contribution in range(r):
                if c2.symplectic[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients[row * r + contribution])

            solver.add(aux_tableau[row * (2*n) + q] == _xor_list(row_contributions))

    return solver.check() == z3.sat
