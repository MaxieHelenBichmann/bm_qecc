"""Graph-state machinery for local-Clifford equivalence checking."""

from __future__ import annotations

from ..core.stabilizer_code import StabilizerCode


def are_lc_equivalent(code: StabilizerCode, other: StabilizerCode) -> bool:
    """Check local-Clifford equivalence via graph-state machinery."""
    raise NotImplementedError("Graph-state local-Clifford equivalence is not implemented yet.")
