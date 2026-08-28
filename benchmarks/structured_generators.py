"""Generators derived from the repository's named structured codes.

Every entry point loads its source by name and follows one of these shapes::

    <family>_codes_<method>(name, seed) -> tuple[Code, Code]
    <family>_code_<method>(name, seed) -> Code

Unlike :mod:`benchmarks.random_generators`, these generators do not sample the
source dimensions or source code. The named code is always the first member of
a pair, or the CSS representative used for a positive LC-CSS instance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode

from .random_generators import (
    _perturbed_stabilizer_code,
    non_lc_equivalent_code,
    non_permutation_equivalent_css_code,
    non_permutation_equivalent_css_code_cnot,
    non_permutation_equivalent_css_code_decoupled,
    non_permutation_equivalent_css_code_independent,
    non_permutation_equivalent_stabilizer_code,
    non_permutation_equivalent_stabilizer_code_anchored,
    non_permutation_equivalent_stabilizer_code_independent,
)
from .utils import (
    _permute_stabilizer_code,
    _permute_tableau,
    _random_permutation,
    lc_equivalent_code,
    lc_equivalent_code_and_log_ops,
    non_lc_css_code,
    permutation_equivalent_code,
    permutation_equivalent_css_code,
)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_MAX_SEED = np.iinfo(np.int32).max

# The values are (file stem, is_css). Aliases are the stable names historically
# exposed by benchmarks.run.NAMED_CODES.
NAMED_CODE_SPECS: dict[str, tuple[str | None, bool]] = {
    "bell": (None, True),
    "3q_rep": ("three_bit_repetition", True),
    "5q_prf": ("five_qubit_perfect", False),
    "steane": ("steane", True),
    "gottesman": ("eight_qubit_gottesman", False),
    "shor": ("shor", True),
    "carbon": ("carbon", True),
    "tetrahedral": ("tetrahedral", True),
    "15q_optimal": ("fifteen_qubit_optimal", False),
    "hamming_15": ("hamming_15", True),
    "golay": ("golay", True),
    "rot_surf_d5": ("rotated_surface_d5", True),
    "bring": ("bring", True),
    "coco_488": ("coco_488", True),
    "hamming_31": ("hamming_31", True),
    "coco_666": ("coco_666", True),
    "bb_72": ("bb_72", True),
    "bb_90": ("bb_90", True),
    "bb_108": ("bb_108", True),
    "bb_144": ("bb_144", True),
}


def named_code_names(*, css_only: bool = False) -> tuple[str, ...]:
    """Return the accepted structured-code names in registry order."""
    return tuple(
        name
        for name, (_, is_css) in NAMED_CODE_SPECS.items()
        if is_css or not css_only
    )


def load_named_code(name: str) -> StabilizerCode:
    """Load a fresh named code, raising ``ValueError`` for an unknown name."""
    try:
        stem, is_css = NAMED_CODE_SPECS[name]
    except KeyError as exc:
        available = ", ".join(NAMED_CODE_SPECS)
        raise ValueError(f"Unknown structured code {name!r}; choose from {available}.") from exc

    if stem is None:
        return CSSCode(
            Hx=np.array([[1, 1]], dtype=np.int8),
            Hz=np.array([[1, 1]], dtype=np.int8),
        )
    path = _DATA_DIR / stem
    return CSSCode.from_file(path) if is_css else StabilizerCode.from_file(path)


def _load_css_code(name: str) -> CSSCode:
    code = load_named_code(name)
    if not isinstance(code, CSSCode):
        raise ValueError(f"Structured code {name!r} is not stored as a CSS code.")
    return code


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, _MAX_SEED))


class PEqCodePairGenerator:
    """Generator for permutation-equivalent pairs from named codes."""

    @staticmethod
    def stabilizer_codes_permuted(
        name: str, seed: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and a physical-qubit permutation of it.

        Sampling bias: the source is one fixed named code and the generator rows
        retain an exact correspondence.
        NOT USABLE whenever conclusions require variation over codes within a
        ``[[n,k]]`` cell or presentation-independent samples; this method varies
        only the permutation of one named source.
        """
        code = load_named_code(name)
        permutation = _random_permutation(code.n, seed=seed)
        return code, _permute_stabilizer_code(code, permutation)

    @staticmethod
    def stabilizer_codes_basis_changed(
        name: str, seed: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and a permuted, basis-changed copy.

        Sampling bias: every seed remains in the permutation orbit of one fixed
        named source, although row correspondence is removed.
        NOT USABLE whenever the target population varies the underlying code or
        code family rather than only its presentation.
        """
        code = load_named_code(name)
        return code, permutation_equivalent_code(code, seed=seed)

    @staticmethod
    def stabilizer_codes_with_logicals(
        name: str, seed: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named pair with its logical frame transported exactly.

        Sampling bias: the code and logical frame are tied to one named source,
        and logical compatibility is guaranteed by transport.
        NOT USABLE whenever logical frames should vary independently or their
        compatibility is part of the measured difficulty.
        """
        code = load_named_code(name)
        permutation = _random_permutation(code.n, seed=seed)
        return code, StabilizerCode(
            generators=_permute_tableau(code.generators, permutation),
            distance=code.distance,
            x_logicals=_permute_tableau(code.x_logicals, permutation),
            z_logicals=_permute_tableau(code.z_logicals, permutation),
        )

    @staticmethod
    def css_codes_permuted(
        name: str, seed: int | None = None
    ) -> tuple[CSSCode, CSSCode]:
        """Return a named CSS code and a physical-qubit permutation of it.

        Sampling bias: the source structure is fixed and the check rows retain
        exact correspondence.
        NOT USABLE whenever presentation normalization or variation within a
        structured-code family is being estimated.
        """
        code = _load_css_code(name)
        permutation = _random_permutation(code.n, seed=seed)
        return code, CSSCode(
            np.asarray(code.Hx, dtype=np.int8)[:, list(permutation)],
            np.asarray(code.Hz, dtype=np.int8)[:, list(permutation)],
            distance=code.distance,
            x_distance=code.x_distance,
            z_distance=code.z_distance,
        )

    @staticmethod
    def css_codes_basis_changed(
        name: str, seed: int | None = None
    ) -> tuple[CSSCode, CSSCode]:
        """Return a named CSS code and a permuted, basis-changed copy.

        Sampling bias: seeds vary only the presentation of one fixed named CSS
        code; they are not independent structured codes.
        NOT USABLE whenever the target population requires different code
        geometries, constructions, or equivalence orbits within one dimension.
        """
        code = _load_css_code(name)
        return code, permutation_equivalent_css_code(code, seed=seed)


class NonPEqCodePairGenerator:
    """Generator for permutation-negative pairs with a named first code."""

    @staticmethod
    def stabilizer_codes_x_z_rank_projection(
        name: str, seed: int | None = None, *, max_attempts: int = 1_000
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and an LC-derived X+Z-rank negative.

        Sampling bias: both codes share the named source's LC orbit and the
        partner is accepted only when its X+Z projection rank changes.
        NOT USABLE whenever the measured property is LC-invariant or is the
        X+Z-rank certificate; those outcomes are fixed by construction.
        """
        code = load_named_code(name)
        partner = non_permutation_equivalent_stabilizer_code(
            code, seed=seed, max_attempts=max_attempts
        )
        return code, partner

    @staticmethod
    def stabilizer_codes_x_z_rank_projection_triple_construction(
        name: str, seed: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and a projection-rank-separated anchored code.

        Sampling bias: the partner contains a selected weight-one anchor and
        shares only ``[[n,k]]`` with the named source.
        NOT USABLE whenever the result should retain the named source's local
        structure or represent ordinary structured negatives.
        """
        code = load_named_code(name)
        partner = non_permutation_equivalent_stabilizer_code_anchored(code, seed=seed)
        return code, partner

    @staticmethod
    def stabilizer_codes_x_z_rank_projection_triple_independent(
        name: str, seed: int | None = None, *, max_attempts: int = 10_000
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and an independent certified proposal.

        Sampling bias: the partner is layered-random rather than structured and
        is retained only when its projection-rank triple differs.
        NOT USABLE whenever both members should belong to the named structured
        family or the projection-rank certificate's natural rate is measured.
        """
        code = load_named_code(name)
        partner = non_permutation_equivalent_stabilizer_code_independent(
            code, seed=seed, max_attempts=max_attempts
        )
        return code, partner

    @staticmethod
    def stabilizer_codes_clifford(
        name: str, seed: int | None = None, *, gate_steps: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and an uncertified Clifford perturbation.

        Sampling bias: the partner is correlated with one named source and its
        distribution depends on the perturbation depth.
        NOT USABLE whenever two independent structured codes or an unconditional
        negative label are required; certify inequivalence with an exact backend.
        """
        code = load_named_code(name)
        return code, _perturbed_stabilizer_code(
            code, seed=seed, gate_steps=gate_steps
        )

    @staticmethod
    def css_codes_cascaded(
        name: str, seed: int | None = None
    ) -> tuple[CSSCode, CSSCode]:
        """Return a named CSS code and the first available certified negative.

        Sampling bias: the method is a structure-dependent mixture of CNOT,
        decoupled-permutation, and independent partners.
        NOT USABLE whenever one stable construction is required across named
        codes or certificate-selected invariant rates are interpreted naturally.
        """
        code = _load_css_code(name)
        return code, non_permutation_equivalent_css_code(code, seed=seed)

    @staticmethod
    def css_codes_cnot(
        name: str, seed: int | None = None, *, max_attempts: int = 110
    ) -> tuple[CSSCode, CSSCode]:
        """Return a named CSS code and a CNOT-derived certified negative.

        Sampling bias: the partner retains much of the named source's density
        and structure but is certificate-selected within its CNOT orbit.
        NOT USABLE whenever independent structured codes or the certificate's
        unconditioned rejection rate are required.
        """
        code = _load_css_code(name)
        partner = non_permutation_equivalent_css_code_cnot(
            code, seed=seed, max_attempts=max_attempts
        )
        return code, partner

    @staticmethod
    def css_codes_decoupled(
        name: str, seed: int | None = None, *, max_attempts: int = 500
    ) -> tuple[CSSCode, CSSCode]:
        """Return a named CSS code with independently permuted X/Z sectors.

        Sampling bias: both sector column multisets are fixed to those of the
        named source and candidates are conditioned on CSS orthogonality.
        NOT USABLE whenever sector structure should vary independently or the
        coupled certificate's natural rejection rate is measured.
        """
        code = _load_css_code(name)
        partner = non_permutation_equivalent_css_code_decoupled(
            code, seed=seed, max_attempts=max_attempts
        )
        return code, partner

    @staticmethod
    def css_codes_independent(
        name: str, seed: int | None = None, *, max_attempts: int = 10_000
    ) -> tuple[CSSCode, CSSCode]:
        """Return a named CSS code and an independent certified proposal.

        Sampling bias: the partner is dense random CSS, not another member of
        the named source's structured family, and is certificate-selected.
        NOT USABLE whenever both members should preserve the named construction
        or certificate-conditioned statistics are interpreted as natural rates.
        """
        code = _load_css_code(name)
        partner = non_permutation_equivalent_css_code_independent(
            code, seed=seed, max_attempts=max_attempts
        )
        return code, partner


class LCEqCodePairGenerator:
    """Generator for LC-equivalent pairs from named codes."""

    @staticmethod
    def stabilizer_codes_local_clifford(
        name: str, seed: int | None = None, *, row_steps: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and a local-Clifford image.

        Sampling bias: all seeds remain in one named LC orbit.
        NOT USABLE whenever the target requires variation across source codes,
        LC orbits, or automorphism groups within one dimension.
        """
        code = load_named_code(name)
        return code, lc_equivalent_code(code, seed=seed, row_steps=row_steps)

    @staticmethod
    def stabilizer_codes_with_logicals(
        name: str, seed: int | None = None, *, row_steps: int | None = None
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named LC pair with an exactly transported logical frame.

        Sampling bias: source structure and logical compatibility are fixed.
        NOT USABLE whenever logical frames should vary independently or their
        compatibility is part of the measured difficulty.
        """
        code = load_named_code(name)
        return code, lc_equivalent_code_and_log_ops(
            code, seed=seed, row_steps=row_steps
        )


class NonLCEqCodePairGenerator:
    """Generator for LC-negative pairs with a named first code."""

    @staticmethod
    def stabilizer_codes_independent(
        name: str, seed: int | None = None, *, max_attempts: int = 10_000
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a named code and a support-rank-certified random partner.

        Sampling bias: the partner is layered-random, shares only ``[[n,k]]``,
        and is accepted because its support-rank profile differs.
        NOT USABLE whenever both members should retain named structure or the
        support-rank certificate's natural rejection rate is measured.
        """
        code = load_named_code(name)
        return code, non_lc_equivalent_code(
            code, seed=seed, max_attempts=max_attempts
        )


class LCEqCodeGenerator:
    """Generator for named codes known to be in a CSS LC orbit."""

    @staticmethod
    def stabilizer_code_local_clifford(
        name: str, seed: int | None = None, *, row_steps: int | None = None
    ) -> StabilizerCode:
        """Hide a named CSS code with local Cliffords and a basis change.

        Sampling bias: all outputs belong to the LC orbit of one named CSS code.
        NOT USABLE whenever prevalence of CSS LC orbits or variation across
        structured CSS constructions is being estimated.
        """
        code = _load_css_code(name)
        return lc_equivalent_code(code, seed=seed, row_steps=row_steps)


class NonLCEqCodeGenerator:
    """Generator for LC-CSS negatives matched to a named CSS code's dimensions."""

    @staticmethod
    def stabilizer_code_locally_rank_one(
        name: str,
        seed: int | None = None,
        *,
        max_attempts: int = 10_000,
        max_exact_rank: int = 16,
    ) -> StabilizerCode:
        """Return the historical structured LC-CSS negative for a named code.

        The named CSS code supplies only ``[[n,k]]``. The returned code is built
        from fixed five-qubit blocks plus a layered-random remainder and then
        certified outside every CSS LC orbit.

        Sampling bias: the negative does not preserve the named source's
        structure; it is a fixed-block, certificate-selected dimension match.
        NOT USABLE whenever the negative should be derived from the named code,
        both samples should share its family, or the locally-rank-one
        certificate's natural rejection rate is measured.
        """
        code = _load_css_code(name)
        return non_lc_css_code(
            code,
            seed=seed,
            max_attempts=max_attempts,
            max_exact_rank=max_exact_rank,
        )
