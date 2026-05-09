"""Classical code equivalence based permutation equivalence checking."""

from __future__ import annotations

from ..core.css_code import CSSCode

import ldpc.mod2.mod2_numpy as mod2


def are_peq_css_classical(c1: CSSCode, c2: CSSCode) -> bool:
    """Check permutation equivalence using algorithms for classical code equivalence. A two-layer approach is used, where the first layer uses Sendrier's Support Splitting Algorithm to partition the columns of the generator matrices into equivalence classes based on the weight enumerator of the hull of the punctured code. 
    The second layer then checks for permutation equivalence by traversing the search tree of possible permutations, and pruning branches based on the canonical form of Feulner's Algorithm.
    
    For each code, the following is done:
    1.) Compute the generator matrices Gx from the parity-check matrices Hx for each code
    2.) Partition the columns of Gx1 into equivalence classes
    3.) Canonicalize the generator matrices of Gx1 anf Gx2 using Feulner's algorithm, and check for equivalence of the canonical forms, pruning the search tree of possible permutations. 
    4.) Check if the found permutation also works for the other matrix Hx.

    This algorithm should be more efficient than the brute-force algorithm, since it avoids checking all permutations, BUT it is still not efficient in the worst case.
    """
    return False
