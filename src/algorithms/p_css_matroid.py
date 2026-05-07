"""Matroid-isomorphism based permutation equivalence checking."""

from __future__ import annotations

from ..core.css_code import CSSCode

def are_peq_css_matroid(c1: CSSCode, c2: CSSCode) -> bool:
    return False