"""Brute-force permutation equivalence checking."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import numpy as np
from ldpc.mod2.mod2_numpy import rank

from ..core.css_code import CSSCode

def are_peq_css_bruteforce(c1: CSSCode, c2: CSSCode) -> bool:
    return False
