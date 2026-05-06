"""Brute-force permutation equivalence checking."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import numpy as np
from ldpc.mod2.mod2_numpy import rank

from ..core.css_code import CSSCode

def are_permutation_equivalent(c1: CSSCode, c2: CSSCode) -> bool:
    raise NotImplementedError("Brute-force permutation equivalence is not implemented yet.")
