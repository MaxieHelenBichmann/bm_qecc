"""Brute-force permutation equivalence checking for CSS codes."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from ldpc.mod2.mod2_numpy import rank

from ..core.css_code import CSSCode


def are_peq_css_bruteforce(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence by brute-force search over all elements of S_n.

    Each permutation is checked by permuting the columns of the parity-check matrices Hx and Hz and checking for equality of their according row spaces using the rank.

    Each row space check should be done in O(n^3) time, and there are O(n!) permutations, so the overall runtime is O(n! * n^3) which is obviously not efficient at all.
    """
    
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return int(rank(matrix))

    c1_hx_rank = _rank(c1.Hx)
    c1_hz_rank = _rank(c1.Hz)

    for perm in permutations(range(c1.n)):
        perm = np.array(perm)
        permuted_hx = c2.Hx[:, perm]
        permuted_hz = c2.Hz[:, perm]

        if (c1_hx_rank == _rank(permuted_hx) and
            c1_hz_rank == _rank(permuted_hz) and
            c1_hx_rank == _rank(np.vstack([c1.Hx, permuted_hx])) and
            c1_hz_rank == _rank(np.vstack([c1.Hz, permuted_hz]))):
            return True
    
    return False
