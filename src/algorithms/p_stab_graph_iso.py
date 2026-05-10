"""Graph-isomorphism based permutation equivalence checking."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab_graph_iso(c1: StabilizerCode, c2: StabilizerCode) -> bool:    
    return False