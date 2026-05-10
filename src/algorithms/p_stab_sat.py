"""SAT based permutation equivalence checking for general Stabilizer Codes."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from ldpc.mod2.mod2_numpy import rank

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab_sat(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    return False
