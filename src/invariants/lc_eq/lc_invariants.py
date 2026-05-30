"""Small invariants that are preserved under local Clifford equivalence, which can be used to quickly rule out LC-equivalence in some cases.
"""

from __future__ import annotations

import numpy as np

from ...core.stabilizer_code import StabilizerCode

def preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.k == c2.k
