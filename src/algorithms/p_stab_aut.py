"""Automorphism-group based equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab_aut(c1: StabilizerCode, c2: StabilizerCode) -> bool:    
    return False
