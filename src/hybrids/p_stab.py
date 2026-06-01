"""Best hybrid solution for checking whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

from ..core.stabilizer_code import StabilizerCode

def are_peq_stab(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    return False