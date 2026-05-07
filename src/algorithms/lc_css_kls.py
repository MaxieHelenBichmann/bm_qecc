"""KLS normal form for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

from ..core.stabilizer_code import StabilizerCode


def is_lceq_css_kls(code: StabilizerCode) -> bool:
    """Check whether a code is local-Clifford equivalent to some CSS code."""
    raise NotImplementedError("LC-to-CSS equivalence is not implemented yet.")
