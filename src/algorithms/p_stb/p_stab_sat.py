"""SAT based permutation equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

import z3

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

def _build_peq_stab_sat_solver(c1: StabilizerCode, c2: StabilizerCode) -> z3.Solver:
    """Build the SAT instance."""
    solver = z3.Solver()

    n = c1.n
    k = c1.k
    r = n - k # assume that tableau is minimal and has no dependent rows, and both tableaus have the same rank

    # permutations
    aux_tableau = [z3.Bool(f'aux_{row}_{col}') for row in range(r) for col in range(2*n)]
    permutation_variables = [z3.Bool(f'p_{i}_{j}') for i in range(n) for j in range(n)]

    for i in range(n):
        solver.add(_exactly_one([permutation_variables[i * n + j] for j in range(n)]))
    for j in range(n):
        solver.add(_exactly_one([permutation_variables[i * n + j] for i in range(n)]))

    for i in range(n):
        for j in range(n):
            x_column_original = c1.symplectic[:, i]
            z_column_original = c1.symplectic[:, i + n]

            x_column_permuted = [aux_tableau[row * (2*n) + j] for row in range(r)]
            z_column_permuted = [aux_tableau[row * (2*n) + j + n] for row in range(r)]

            solver.add(z3.Implies(permutation_variables[i * n + j], z3.And(_elementwise_map(x_column_original, x_column_permuted), _elementwise_map(z_column_original, z_column_permuted))))

    # row operations
    row_operation_coefficients = [z3.Bool(f'r_{i}_{j}') for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):

            row_contributions = []
            for contribution in range(r):
                if c2.symplectic[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients[row * r + contribution])

            solver.add(aux_tableau[row * (2*n) + q] == _xor_list(row_contributions))

    return solver


def are_peq_stab_sat(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check permutation equivalence by reducing it to a SAT problem and using a SAT solver.

    The idea is to create boolean variables for the permutation and the row operations, and then add constraints that enforce that the permuted tableau of c1 is equal to the row-operated tableau of c2.
    1.) Create a auxiliary tableau that encodes the column permutations of c1, and the row operations of c2.
    2.) Create boolean variables p_{i,j} for the permutation, where p_{i,j} is true if the i-th qubit of c1 is mapped to the j-th qubit of the auxiliary tableau.
    3.) Create boolean variables r_{i,j} for the row operations, where r_{i,j} is true if the j-th row is added to the i-th row in the auxiliary tableau.
    4.) Check satisfiability of the resulting formula. If it is satisfiable, then c1 and c2 are permutation equivalent, otherwise they are not.
    """
    return _build_peq_stab_sat_solver(c1, c2).check() == z3.sat
