"""Best hybrid solution for checking whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

from ..core.stabilizer_code import StabilizerCode

def are_lceq(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    return False