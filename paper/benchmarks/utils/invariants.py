"""Invariant functions used by the rejection, signature, and timing collectors."""

from __future__ import annotations

from typing import Any

from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids import lc_stb, p_css, p_stab

_row_basis = p_stab._row_basis

INVARIANTS = {
    "pm_stb": ("linear_dependency", "signatures"),
    "pm_css": ("linear_dependency", "signatures"),
    "lc_stb": ("local_invariant",),
}


def _partition_sizes(partition: dict[Any, list[int]] | None) -> list[int]:
    return sorted((len(group) for group in (partition or {}).values()), reverse=True)


def _prepared(problem: str, left: StabilizerCode, right: StabilizerCode) -> tuple:
    if problem == "pm_css":
        if not isinstance(left, CSSCode) or not isinstance(right, CSSCode):
            raise TypeError("pm_css invariants require CSSCode inputs")
        return (
            _row_basis(left.Hx),
            _row_basis(left.Hz),
            _row_basis(right.Hx),
            _row_basis(right.Hz),
        )
    return _row_basis(left.symplectic), _row_basis(right.symplectic)


def evaluate_signature(
    problem: str,
    left: StabilizerCode,
    right: StabilizerCode,
) -> tuple[bool, list[int], list[int]]:
    matrices = _prepared(problem, left, right)
    if problem == "pm_stb":
        compatible, first, second = p_stab.preserved_punctured_hull_weight_enumerator(
            *matrices
        )
    elif problem == "pm_css":
        compatible, first, second = p_css.preserved_punctured_hull_weight_enumerator(
            *matrices
        )
    else:
        raise ValueError(f"no signature invariant for {problem}")
    return bool(compatible), _partition_sizes(first), _partition_sizes(second)


def evaluate_invariant(
    name: str,
    problem: str,
    left: StabilizerCode,
    right: StabilizerCode,
) -> bool:
    matrices = _prepared(problem, left, right)
    if name == "linear_dependency" and problem == "pm_stb":
        return bool(p_stab.preserved_linear_dependencies(*matrices))
    if name == "linear_dependency" and problem == "pm_css":
        return bool(p_css.preserved_linear_dependencies(*matrices))
    if name == "signatures":
        return evaluate_signature(problem, left, right)[0]
    if name == "local_invariant" and problem == "lc_stb":
        return bool(lc_stb.preserved_low_degree_local_invariant(*matrices))
    raise ValueError(f"unknown invariant {name!r} for {problem!r}")
