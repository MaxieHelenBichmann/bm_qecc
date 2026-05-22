"""SAT based local clifford equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

import z3

from ..core.stabilizer_code import StabilizerCode

def are_lceq_sat(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check local-clifford equivalence by reducing it to a SAT problem and using a SAT solver.
    """
    solver = z3.Solver()

    return solver.check() == z3.sat
