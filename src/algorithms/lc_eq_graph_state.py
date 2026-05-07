"""Graph-state machinery for local-Clifford equivalence checking."""

from __future__ import annotations

from ..core.stabilizer_code import StabilizerCode


def are_lceq_graph_state(code: StabilizerCode, other: StabilizerCode) -> bool:
    return False
