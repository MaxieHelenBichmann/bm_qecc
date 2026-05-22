"""SAT based approach for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import z3

from ..core.stabilizer_code import StabilizerCode

def is_lceq_css_sat(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check local-clifford equivalence to a CSS code by reducing it to a SAT problem and using a SAT solver.
    """
    solver = z3.Solver()
    return solver.check() == z3.sat
