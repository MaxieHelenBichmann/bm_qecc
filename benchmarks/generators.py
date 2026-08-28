"""Benchmark generators for (non-)equivalent code pairs.

Every public entry point is a static method of one of the four generator
classes below and follows the same shape:

    <family>_codes_<method>(n, k, seed) -> tuple[Code, Code]

``<family>`` is the code family the pair is drawn from (``stabilizer`` or
``css``) and ``<method>`` names the *construction* used to obtain the second
code of the pair. The classes carry no state; they only group the constructions
belonging to one decision problem.

Positive constructions are exact: the second code is obtained by applying a
transformation from the equivalence group, so the pair is equivalent by
construction. Negative constructions instead document the *certificate* they
use and raise :class:`RandomizeError` when no candidate carrying that
certificate is found inside the search budget. The one exception is the
``clifford`` method, which is deliberately uncertified; see its docstring.

The choice of method matters, and different methods are not interchangeable.
A certified negative is selected with respect to an invariant. It therefore
cannot estimate that invariant's rejection rate on an unconditioned population.
Runtime measurements remain valid for the explicitly named generated family,
but may be artificially easy for solvers that evaluate the certificate or a
correlated invariant.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode

from .utils import (
    RandomizeError,
    _apply_random_clifford_gate,
    _permute_stabilizer_code,
    _permute_tableau,
    _random_css_check_matrices,
    _random_permutation,
    _random_tableau_row_space_base_change,
    _rank_binary,
    _without_phases,
    lc_equivalent_code,
    lc_equivalent_code_and_log_ops,
    permutation_equivalent_code,
    permutation_equivalent_css_code,
    random_css_code,
    random_stabilizer_code,
)

_MAX_SEED = np.iinfo(np.int32).max


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, _MAX_SEED))


def _perturbed_stabilizer_code(
    code: StabilizerCode,
    *,
    seed: int | None = None,
    gate_steps: int | None = None,
) -> StabilizerCode:
    """Apply an independent random Clifford circuit to ``code``.

    This invariant-neutral construction intentionally does not certify that
    the result left the source's permutation orbit; callers using this family
    must certify the label separately with an exact backend.
    """
    rng = np.random.default_rng(seed)
    steps = 2 * code.n if gate_steps is None else gate_steps
    if steps < 0:
        raise ValueError("gate_steps must be non-negative.")

    tableau = code.generators.copy()
    for _ in range(steps):
        _apply_random_clifford_gate(tableau, rng)
    return StabilizerCode(_without_phases(tableau), distance=code.distance)


# --------------------------------------------------------------------------
# Permutation equivalence
# --------------------------------------------------------------------------

class PEqCodePairGenerator:
    """Generator for seeded benchmark pairs of permutation-equivalent codes."""

    @staticmethod
    def stabilizer_codes_permuted(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        clifford_steps: int | None = None,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a random code and a copy with its physical qubits permuted.

        The plainest positive construction: only the qubit order differs, the
        generator basis is left alone. Solvers that normalize the generator
        basis see an easier instance here than under
        :meth:`stabilizer_codes_basis_changed`.

        Sampling bias: the source comes from the layered random-Clifford
        ensemble, not uniformly from ``[[n,k]]`` codes, and the two tableaux
        retain an exact row-by-row correspondence.
        NOT USABLE whenever the measured quantity can exploit or is affected by
        generator-row correspondence, because the partner did not receive an
        independent basis change. This includes typical presentation-
        normalization cost and generator-basis-sensitive representation sizes.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        permutation_seed = _seed(rng)

        code = random_stabilizer_code(
            n, k, seed=code_seed, clifford_steps=clifford_steps
        )
        permutation = _random_permutation(n, seed=permutation_seed)
        return code, _permute_stabilizer_code(code, permutation)

    @staticmethod
    def stabilizer_codes_basis_changed(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        clifford_steps: int | None = None,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a random code and a permuted copy in a different generator basis.

        The second code additionally undergoes seeded invertible row operations,
        so the two tableaux generate the same stabilizer group through different
        generating sets. This hides the row correspondence a plain permutation
        leaves visible.

        Sampling bias: both codes are in the same permutation orbit and the
        source still follows the layered random-Clifford ensemble, but the
        additional row operations remove the artificial row correspondence.
        NOT USABLE whenever the target population is uniform stabilizer codes,
        structured/named codes, or independently sampled equivalent
        presentations, because the source ensemble and shared orbit do not
        represent those populations.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        partner_seed = _seed(rng)

        code = random_stabilizer_code(
            n, k, seed=code_seed, clifford_steps=clifford_steps
        )
        return code, permutation_equivalent_code(code, seed=partner_seed)

    @staticmethod
    def stabilizer_codes_with_logicals(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        clifford_steps: int | None = None,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a permuted pair whose logical operators are permuted too.

        :meth:`stabilizer_codes_permuted` lets the second code recompute its
        logical operators from the permuted generators, which generally yields a
        different logical basis. This variant transports the stored logical X
        and Z operators through the same permutation, which is what solvers
        consuming logical operators as part of their input would require.

        Sampling bias: the logical frame is deliberately correlated by exact
        transport, while the stabilizer source follows the layered random-Clifford
        ensemble.
        NOT USABLE whenever logical frames are meant to vary independently or
        logical-frame compatibility is itself being measured, because
        compatibility is guaranteed by construction.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        permutation_seed = _seed(rng)

        code = random_stabilizer_code(
            n, k, seed=code_seed, clifford_steps=clifford_steps
        )
        permutation = _random_permutation(n, seed=permutation_seed)
        return code, StabilizerCode(
            generators=_permute_tableau(code.generators, permutation),
            distance=code.distance,
            x_logicals=_permute_tableau(code.x_logicals, permutation),
            z_logicals=_permute_tableau(code.z_logicals, permutation),
        )

    @staticmethod
    def css_codes_permuted(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        rx: int | None = None,
    ) -> tuple[CSSCode, CSSCode]:
        """Return a random CSS code and a copy with its physical qubits permuted.

        ``rx`` fixes the X-check rank; by default it is drawn uniformly from
        ``0..n-k`` so the benchmark family is not restricted to balanced codes.

        Sampling bias: ``Hx`` is sampled first and ``Hz`` from its kernel, so
        this is a usually dense, X/Z-asymmetric presentation ensemble.
        The unchanged row bases also expose row correspondence.
        NOT USABLE whenever the measurement is sensitive to row correspondence,
        sparsity, or X/Z symmetry, because those properties are fixed by this
        construction rather than sampled naturally. This includes typical
        presentation-normalization cost and basis-sensitive representation size.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        permutation_seed = _seed(rng)
        x_rank = int(rng.integers(0, n - k + 1)) if rx is None else rx

        code = random_css_code(n, k, x_rank, seed=code_seed)
        permutation = _random_permutation(n, seed=permutation_seed)
        return code, CSSCode(
            np.asarray(code.Hx, dtype=np.int8)[:, list(permutation)],
            np.asarray(code.Hz, dtype=np.int8)[:, list(permutation)],
            distance=code.distance,
            x_distance=code.x_distance,
            z_distance=code.z_distance,
        )

    @staticmethod
    def css_codes_basis_changed(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        rx: int | None = None,
    ) -> tuple[CSSCode, CSSCode]:
        """Return a random CSS code and a permuted copy in a different check basis.

        The X and Z check matrices undergo *independent* seeded row operations,
        matching the actual freedom of a CSS presentation: only the two row
        spaces are fixed, not the rows themselves.

        Sampling bias: ``Hx`` is sampled first and ``Hz`` from its kernel, so
        this is a usually dense, X/Z-asymmetric presentation ensemble.
        NOT USABLE whenever drawing conclusions about sparse, LDPC, geometric,
        quasi-cyclic, X/Z-symmetric, or uniformly sampled CSS populations,
        because basis randomization removes row correspondence but does not
        remove the source ensemble's structural bias.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        permutation_seed = _seed(rng)
        x_rank = int(rng.integers(0, n - k + 1)) if rx is None else rx

        code = random_css_code(n, k, x_rank, seed=code_seed)
        return code, permutation_equivalent_css_code(code, permutation_seed)

class NonPEqCodePairGenerator:
    """Generator for seeded benchmark pairs outside one permutation orbit."""

    @staticmethod
    def stabilizer_codes_x_z_rank_projection(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        clifford_steps: int | None = None,
        max_attempts: int = 10_000,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a pair separated by the rank of the X+Z projection.

        The partner is a local-Clifford image of the source, kept only when the
        X+Z projection rank differs. That rank is a permutation invariant but is
        *not* one of the quantities the ``p_stab`` hybrid inspects, so the pair
        survives the hybrid's cheap filters and reaches its expensive stage.

        Sampling bias: acceptance explicitly selects a changed X+Z projection
        rank while keeping the pair in one LC orbit.
        NOT USABLE whenever the measured property is LC-invariant, because both
        members belong to the same LC orbit and such a property is forced to
        agree. This includes the linear-dependency invariant and Sendrier's
        signatures. It is also circular for the X+Z-rank certificate used to
        accept the pair.
        """
        return _retry_stabilizer_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            clifford_steps=clifford_steps,
            partner=lambda code, partner_seed: non_permutation_equivalent_stabilizer_code(
                code, seed=partner_seed, clifford_steps=clifford_steps
            ),
        )

    @staticmethod
    def stabilizer_codes_x_z_rank_projection_triple_construction(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        clifford_steps: int | None = None,
        max_attempts: int = 10_000,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a pair separated by the X/Z/X+Z projection-rank triple.

        The partner is a random ``[[n-1, k]]`` block plus a single-qubit X, Z or
        Y anchor selected so its projection-rank triple differs from the random
        source. Because the certificate is decided from the anchor alone, no
        candidate-rejection loop is needed inside the partner constructor. The
        outer pair driver may still retry with a new source if no anchor separates
        the first source from the constructed partner.

        Sampling bias: one member has a guaranteed weight-one stabilizer/direct-
        sum structure and its anchor is chosen using the certificate. This results
        in a scalability benefit, but not a natural random code.
        NOT USABLE whenever the result is intended to describe ordinary random
        codes or is sensitive to low-weight/direct-sum structure, because every
        partner contains a selected weight-one anchor. This includes generic
        weight, signature, invariant, and representation distributions; it is
        also circular for the projection-rank certificate.
        """
        return _retry_stabilizer_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            clifford_steps=clifford_steps,
            partner=lambda code, partner_seed: non_permutation_equivalent_stabilizer_code_anchored(
                code, seed=partner_seed, clifford_steps=clifford_steps
            ),
        )

    @staticmethod
    def stabilizer_codes_x_z_rank_projection_triple_independent(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        clifford_steps: int | None = None,
        max_attempts: int = 10_000,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return independently proposed codes separated by a projection-rank triple.

        Each partner proposal is sampled independently from the source and kept
        only when its X/Z/X+Z projection-rank triple differs.

        Sampling bias: although the two draws are independent, the candidate is
        retained only when the X/Z/X+Z projection-rank certificate differs.
        This makes it a useful easy-negative baseline for solver, preprocessing-
        cost, representation-cost, and winner-map comparisons.
        NOT USABLE whenever the target population is unconditioned independent
        pairs or the measured statistic is the selection certificate (or a
        deterministic function of it), because rejection sampling removes every
        same-triple pair. Correlated invariant/signature results are valid only
        for this ``independent-certified`` family, not as generic-negative rates.
        """
        return _retry_stabilizer_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            clifford_steps=clifford_steps,
            partner=lambda code, partner_seed: non_permutation_equivalent_stabilizer_code_independent(
                code, seed=partner_seed, clifford_steps=clifford_steps
            ),
        )

    @staticmethod
    def stabilizer_codes_clifford(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        gate_steps: int | None = None,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return an *uncertified* candidate negative pair, invariant-neutral.

        The partner is the source perturbed by an independently drawn random
        Clifford circuit over the repository's own gate set, which mixes local
        gates with CNOT, CZ and SWAP. Nothing here consults an invariant, so the
        family is aligned with no preprocessor: it neither hides the difference
        from them, as a purely local circuit does, nor selects for a difference
        they can see, as a certified search does.

        The price is that the pair is *not* guaranteed inequivalent -- a random
        Clifford circuit occasionally lands back in the permutation orbit.
        Acceptance belongs to the caller's exact backend, which is exactly what
        makes the family neutral for rejection-rate measurements.
        Crucially, no measured invariant or signature selects candidates.

        Sampling bias: the second code is a short Clifford perturbation of the
        first, so the members are correlated and the distribution depends on
        ``gate_steps``; exact-backend rejection also conditions the retained
        sample on leaving the permutation orbit.
        NOT USABLE whenever the target population requires two independent code
        draws, a specified random-Clifford measure, or an unconditional sample,
        because the pair shares a source and retained negatives are conditioned
        on the exact verdict. It remains suitable for invariant/signature rates
        on the explicitly labeled Clifford-perturbed population.
        """
        rng = np.random.default_rng(seed)
        source_seed = _seed(rng)
        partner_seed = _seed(rng)

        code = random_stabilizer_code(n, k, seed=source_seed)
        partner = _perturbed_stabilizer_code(
            code, seed=partner_seed, gate_steps=gate_steps
        )
        return code, partner

    @staticmethod
    def css_codes_cascaded(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        max_attempts: int = 10_000,
    ) -> tuple[CSSCode, CSSCode]:
        """Return a certified negative CSS pair, trying every method in turn.

        Tries the partner constructions underlying :meth:`css_codes_cnot`,
        :meth:`css_codes_decoupled`, and :meth:`css_codes_independent` in that
        order when applicable, keeping the first candidate that carries a
        certificate. CNOT candidates are attempted only for large sparse source
        codes. This is the general-purpose choice: the cheaper, structure-
        preserving methods do not apply to every code, and an independent
        proposal provides the broadest fallback.

        Sampling bias: the output is a size/structure-dependent mixture of
        CNOT, decoupled-permutation, and independent candidates, each accepted
        because a CSS permutation invariant changes. The family composition can
        change across parameter cells when earlier methods fail.
        NOT USABLE whenever a comparison assumes one stable construction across
        cells or an unconditioned population, because method availability changes
        the mixture and every component is certificate-selected. This includes
        measuring rejection by those certificates and reporting one generic
        invariant/signature rate for the mixture.
        """
        return _retry_css_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            partner=non_permutation_equivalent_css_code,
        )

    @staticmethod
    def css_codes_cnot(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        max_attempts: int = 10_000,
    ) -> tuple[CSSCode, CSSCode]:
        """Return a negative CSS pair whose partner is a CNOT image of the source.

        A physical CNOT acts on the X columns as ``target ^= control`` and
        contragrediently on the Z columns, which preserves ``Hx @ Hz.T == 0``
        and both check ranks while generally leaving the permutation orbit. The
        partner therefore keeps the source's density and much of its structure,
        which an unrelated random sample does not -- important for large sparse
        codes, where a dense random negative is rejected on sight.

        This public pair method nevertheless draws its source with
        :func:`~benchmarks.utils.random_css_code`, which normally produces a
        dense code. To perturb an existing sparse or named code, call
        :func:`non_permutation_equivalent_css_code_cnot` directly.

        Sampling bias: both codes lie in the same physical-CNOT orbit, cheap
        visible invariants are forced to match, and candidates are accepted only
        when the chosen CSS permutation certificate changes.
        NOT USABLE whenever independent codes, an unconditioned physical-orbit
        distribution, or the certificate's natural rejection rate is required,
        because the pair shares a CNOT orbit and is rejection-sampled by that
        certificate. Signatures remain measurable only as properties of this
        CNOT-derived, certificate-selected family.
        """
        return _retry_css_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            partner=non_permutation_equivalent_css_code_cnot,
        )

    @staticmethod
    def css_codes_decoupled(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        max_attempts: int = 10_000,
    ) -> tuple[CSSCode, CSSCode]:
        """Return a negative CSS pair built from two independent column permutations.

        ``Hx`` and ``Hz`` are permuted by *different* qubit permutations. Every
        column-local quantity is preserved exactly -- weights, zero columns,
        duplicate-column multiplicities, both ranks -- so the pair can only be
        separated by an invariant that couples the X and Z sectors. Candidates
        that break CSS orthogonality are discarded.

        Sampling bias: each sector's column multiset is fixed exactly, while
        conditioning on CSS orthogonality and a changed coupled certificate can
        be very selective.
        NOT USABLE whenever independently varying sector structure, generic CSS
        structure, or the coupled certificate's natural rejection rate is
        required, because both sector column multisets are fixed and candidates
        are conditioned on orthogonality and certificate change. Other invariant
        or signature results describe only this decoupled-permutation family.
        """
        return _retry_css_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            partner=non_permutation_equivalent_css_code_decoupled,
        )

    @staticmethod
    def css_codes_independent(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        max_attempts: int = 10_000,
    ) -> tuple[CSSCode, CSSCode]:
        """Return independently proposed CSS codes with a different invariant.

        Candidates are constrained to match the source's cheap visible
        invariants -- ranks, zero columns, duplicate-column multiplicities -- so
        the pair is not rejected for a trivial reason, but they share no
        structure otherwise. The control family, as for
        :meth:`stabilizer_codes_x_z_rank_projection_triple_independent`.

        Sampling bias: sources and candidates use the dense ``Hx``-first CSS
        ensemble, candidates are selected for a changed certificate, and cheap
        visible invariants are matched early in the search (then possibly
        relaxed for dense codes).
        NOT USABLE whenever the target is an unconditioned independent-pair
        distribution or the measured statistic is the certificate or a forced
        cheap invariant, because candidates are accepted/rejected using those
        quantities. Other invariant/signature statistics describe only this
        explicitly labeled independent-certified family.
        """
        return _retry_css_pair(
            n,
            k,
            seed=seed,
            max_attempts=max_attempts,
            partner=non_permutation_equivalent_css_code_independent,
        )


# --------------------------------------------------------------------------
# local Clifford equivalence
# --------------------------------------------------------------------------

class LCEqCodePairGenerator:
    """Generator for seeded benchmark pairs in one local-Clifford orbit."""

    @staticmethod
    def stabilizer_codes_local_clifford(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        row_steps: int | None = None,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return a random code and a local-Clifford image of it.

        One single-qubit Clifford is drawn per qubit from the six-element coset
        representative set, followed by a seeded generator-basis randomization.
        At least one nonidentity local Clifford is always applied, so the pair is
        never trivially identical.

        Sampling bias: the source is from the layered random-Clifford ensemble and the
        partner is in exactly the same LC orbit, with one independently sampled
        single-qubit Clifford per coordinate.
        NOT USABLE whenever the target requires uniform sampling of codes, LC
        orbits, or group elements, because a finite layered random-Clifford source and
        one product distribution of local Cliffords do not provide those
        measures. This includes inferring LC-orbit or automorphism-group sizes.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        partner_seed = _seed(rng)

        code = random_stabilizer_code(n, k, seed=code_seed)
        return code, lc_equivalent_code(code, seed=partner_seed, row_steps=row_steps)

    @staticmethod
    def stabilizer_codes_with_logicals(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        row_steps: int | None = None,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return an LC-equivalent pair whose logical operators are transformed too.

        Unlike :meth:`stabilizer_codes_local_clifford`, the partner's logical X
        and Z operators are carried through the same local Cliffords instead of
        being recomputed from the transformed generators. Solvers that consume
        logical operators need the transported basis to agree.

        Sampling bias: both the code and logical frame are exactly correlated by
        transport, and the source follows the layered random-Clifford ensemble.
        NOT USABLE whenever logical frames should be independent, potentially
        incompatible, or part of the difficulty being measured, because exact
        transport guarantees compatibility. This includes unrestricted logical-
        basis benchmarks; the method is intended for compatible-frame algorithms.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        partner_seed = _seed(rng)

        code = random_stabilizer_code(n, k, seed=code_seed)
        return code, lc_equivalent_code_and_log_ops(
            code, seed=partner_seed, row_steps=row_steps
        )

class NonLCEqCodePairGenerator:
    """Generator for seeded benchmark pairs in different local-Clifford orbits."""

    @staticmethod
    def stabilizer_codes_independent(
        n: int,
        k: int,
        seed: int | None = None,
        *,
        row_steps: int | None = None,
        max_attempts: int = 10_000,
    ) -> tuple[StabilizerCode, StabilizerCode]:
        """Return two codes separated by the support-projection rank profile.

        A local Clifford applies an invertible 2x2 transform to each qubit's
        X/Z column pair, so the rank of the columns restricted to any fixed
        qubit subset cannot change. Ordering those ranks over all subsets up to
        size three gives an LC invariant. Partner proposals are sampled
        independently and retained only when their profile differs.

        Sampling bias: both codes use the layered random-Clifford ensemble, but the
        second is rejection-sampled specifically for a different support-rank
        profile.
        NOT USABLE whenever the target is an unconditioned independent-negative
        population or the measured statistic is that support-rank profile (or a
        deterministic rejecting function of it), because differing profiles are
        the acceptance condition. Correlated degree-2 invariants and signatures
        may be measured only as properties of this selected family, not as
        generic random-negative rates.
        """
        rng = np.random.default_rng(seed)
        code_seed = _seed(rng)
        partner_seed = _seed(rng)

        code = random_stabilizer_code(n, k, seed=code_seed)
        return code, non_lc_equivalent_code(
            code, seed=partner_seed, row_steps=row_steps, max_attempts=max_attempts
        )


# --------------------------------------------------------------------------
# Retry drivers
# --------------------------------------------------------------------------


def _retry_stabilizer_pair(
    n: int,
    k: int,
    *,
    seed: int | None,
    max_attempts: int,
    clifford_steps: int | None,
    partner: Any,
) -> tuple[StabilizerCode, StabilizerCode]:
    """Resample the source code until ``partner`` finds a certified candidate.

    The certificate is not available for every randomly sampled source, so a
    failed partner search is retried against a fresh source rather than
    propagated.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        code_seed = _seed(rng)
        partner_seed = _seed(rng)
        code = random_stabilizer_code(
            n, k, seed=code_seed, clifford_steps=clifford_steps
        )
        try:
            return code, partner(code, partner_seed)
        except RandomizeError:
            continue

    raise RandomizeError(
        f"Could not generate a certified stabilizer pair after {max_attempts} attempts."
    )


def _retry_css_pair(
    n: int,
    k: int,
    *,
    seed: int | None,
    max_attempts: int,
    partner: Any,
) -> tuple[CSSCode, CSSCode]:
    """Resample the source CSS code until ``partner`` finds a certified candidate.

    The X-check rank is drawn away from the degenerate ``rx in {0, n-k}`` ends
    whenever possible: a code with an empty X or Z sector has no non-equivalent
    partner under most of the certificates used here.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        code_seed = _seed(rng)
        partner_seed = _seed(rng)
        total_checks = n - k
        if total_checks >= 2:
            rx = int(rng.integers(1, total_checks))
        else:
            rx = int(rng.integers(0, total_checks + 1))
        code = random_css_code(n, k, rx, seed=code_seed)
        try:
            return code, partner(code, partner_seed)
        except RandomizeError:
            continue

    raise RandomizeError(
        f"Could not generate a certified CSS pair after {max_attempts} attempts."
    )


# --------------------------------------------------------------------------
# Negative partner constructors: permutation equivalence, stabilizer codes
# --------------------------------------------------------------------------


def non_permutation_equivalent_stabilizer_code(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    clifford_steps: int | None = None,
    max_attempts: int = 1_000,
) -> StabilizerCode:
    """Return a non-permutation-equivalent partner that passes hybrid filters.

    The certificate is the rank of the X+Z projection.  Unlike the separate X
    and Z ranks, this invariant is not inspected by the stabilizer hybrid.  A
    candidate is returned only after the hybrid's cheap shape/rank/column
    filters accept the pair.  Later polynomial invariants may still reject some
    generated cases; avoiding that exhaustively would make large-instance
    generation prohibitively expensive.
    """
    if isinstance(code, CSSCode):
        # A qubit permutation maps the pure-X and pure-Z subspaces separately.
        # Thus a CSS-certified negative is also a certified negative when both
        # inputs are viewed as general stabilizer codes.
        return non_permutation_equivalent_css_code(code, seed=seed)

    rng = np.random.default_rng(seed)
    if code.k == code.n:
        raise RandomizeError(
            "No non-equivalent stabilizer code exists with these small invariants."
        )

    invariant = _projection_rank_invariant(code)[2]
    for _ in range(max_attempts):
        candidate = lc_equivalent_code(code, seed=_seed(rng))
        if not _passes_stabilizer_hybrid_cheap_invariants(code, candidate):
            continue
        if _projection_rank_invariant(candidate)[2] != invariant:
            return candidate

    raise RandomizeError(
        "Could not find a cheap-filter-preserving candidate with a different X+Z projection rank."
    )


def non_permutation_equivalent_stabilizer_code_anchored(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    clifford_steps: int | None = None,
) -> StabilizerCode:
    """Return a same-``[[n, k]]`` code certified non-equivalent by projection ranks.

    The certificate is the rank triple of the X projection, Z projection, and
    X+Z projection of the stabilizer row space. Physical-qubit permutations only
    permute the columns inside each projection, so these ranks are invariant.

    The candidate is a random ``[[n-1, k]]`` block extended by a single-qubit
    anchor stabilizer. Appending an X, Z or Y anchor shifts the rank triple in
    three different directions, so a separating anchor can be *computed* rather
    than searched for: at most three options are examined, independently of
    ``n``.
    """
    rng = np.random.default_rng(seed)
    if code.k == code.n:
        raise RandomizeError(
            "No non-equivalent stabilizer code exists with these small invariants."
        )

    invariant = _projection_rank_invariant(code)
    # Use a stronger separation than PEQ strictly needs so color-symmetric
    # benchmark reductions do not collapse X-only and Z-only negatives.
    xz_swapped_invariant = (invariant[1], invariant[0], invariant[2])
    base_code = _random_anchor_base_code(
        code.n, code.k, rng=rng, clifford_steps=clifford_steps
    )
    base_invariant = (
        (0, 0, 0) if base_code is None else _projection_rank_invariant(base_code)
    )

    anchors = ["X", "Z", "Y"]
    rng.shuffle(anchors)
    for anchor in anchors:
        candidate_invariant = _anchor_projection_rank_invariant(base_invariant, anchor)
        if candidate_invariant not in {invariant, xz_swapped_invariant}:
            return _random_anchored_stabilizer_code(
                code.n,
                code.k,
                anchor,
                base_code=base_code,
                rng=rng,
            )

    raise RandomizeError(
        "Could not construct a candidate with a different projection-rank invariant."
    )


def non_permutation_equivalent_stabilizer_code_independent(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    clifford_steps: int | None = None,
    max_attempts: int = 10_000,
) -> StabilizerCode:
    """Return an independent proposal with a different projection-rank triple.

    Nothing links the candidate to ``code`` beyond ``[[n, k]]``; acceptance is
    decided by the same permutation-invariant rank triple used by
    :func:`non_permutation_equivalent_stabilizer_code_anchored`.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    if code.k == code.n:
        raise RandomizeError(
            "No non-equivalent stabilizer code exists with these small invariants."
        )

    rng = np.random.default_rng(seed)
    invariant = _projection_rank_invariant(code)
    for _ in range(max_attempts):
        candidate = random_stabilizer_code(
            code.n, code.k, seed=_seed(rng), clifford_steps=clifford_steps
        )
        if _projection_rank_invariant(candidate) != invariant:
            return candidate

    raise RandomizeError(
        "Could not find an independent candidate with a different projection-rank invariant."
    )


# --------------------------------------------------------------------------
# Negative partner constructors: permutation equivalence, CSS codes
# --------------------------------------------------------------------------


def non_permutation_equivalent_css_code(
    code: CSSCode, seed: int | None = None
) -> CSSCode:
    """Return a CSS code certified non-equivalent by permutation invariants.

    Tries the CNOT, decoupled-permutation and independent-sample methods in
    turn, returning the first candidate that carries a certificate. See the
    individual constructors for what each one preserves.
    """
    rng = np.random.default_rng(seed)
    _certificate(code)  # fail fast on codes that admit no certified partner

    if _uses_additive_invariant(code):
        try:
            return non_permutation_equivalent_css_code_cnot(code, seed=_seed(rng))
        except RandomizeError:
            pass

    rx = _rank_binary(code.Hx)
    rz = _rank_binary(code.Hz)
    if rx and rz:
        try:
            return non_permutation_equivalent_css_code_decoupled(code, seed=_seed(rng))
        except RandomizeError:
            pass

    return non_permutation_equivalent_css_code_independent(code, seed=_seed(rng))


def non_permutation_equivalent_css_code_cnot(
    code: CSSCode,
    seed: int | None = None,
    *,
    max_attempts: int = 110,
) -> CSSCode:
    """Return a CNOT image of ``code`` with a different permutation invariant.

    Coupled CNOT column operations preserve CSS orthogonality, dimensions, and
    much of the source code's structure. For large structured codes this
    produces substantially more representative negatives than replacing the code
    with an unrelated dense random sample.

    Some highly symmetric codes (notably BB ``[[90, 8]]``) have repeated column
    multiplicities that essentially no nontrivial CNOT preserves, so the search
    may exhaust its budget.
    """
    rng = np.random.default_rng(seed)
    invariant = _certificate(code)
    visible_invariant = _visible_css_invariant(code)

    for _ in range(max_attempts):
        candidate_hx, candidate_hz = _random_css_cnot_candidate_matrices(code, rng=rng)
        candidate = _css_like(code, candidate_hx, candidate_hz)
        if _visible_css_invariant(candidate) != visible_invariant:
            continue
        if _certificate(candidate) != invariant:
            return candidate

    raise RandomizeError(
        "Could not find a CNOT candidate with a different CSS permutation invariant."
    )


def non_permutation_equivalent_css_code_decoupled(
    code: CSSCode,
    seed: int | None = None,
    *,
    max_attempts: int = 500,
) -> CSSCode:
    """Return a CSS code whose X and Z checks are permuted independently.

    Both sectors keep their column multisets exactly, so every cheap
    column-local filter accepts the pair and the separation has to come from an
    invariant that couples X and Z. Candidates whose two permutations coincide,
    or which break ``Hx @ Hz.T == 0``, are discarded.
    """
    rx = _rank_binary(code.Hx)
    rz = _rank_binary(code.Hz)
    if not (rx and rz):
        raise RandomizeError(
            "The decoupled construction needs a nonempty X and Z check sector."
        )

    rng = np.random.default_rng(seed)
    invariant = _certificate(code)

    for _ in range(max_attempts):
        candidate = _decoupled_css_column_permutation_candidate(code, rng=rng)
        if candidate is None:
            continue
        if _certificate(candidate) != invariant:
            return candidate

    raise RandomizeError(
        "Could not find a decoupled candidate with a different CSS permutation invariant."
    )


def non_permutation_equivalent_css_code_independent(
    code: CSSCode,
    seed: int | None = None,
    *,
    max_attempts: int = 10_000,
) -> CSSCode:
    """Return an independent CSS proposal with a different invariant.

    Candidates must preserve the very cheap invariants used as early filters by
    the benchmark solvers, so random negative pairs are not rejected just
    because of zero columns or duplicate-column multiplicities. That constraint
    is relaxed late in the search for dense codes, where it is the binding one.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)
    rx = _rank_binary(code.Hx)
    invariant = _certificate(code)
    visible_invariant = _visible_css_invariant(code)
    strict_attempts = max_attempts if _uses_additive_invariant(code) else 1_000

    for attempt in range(max_attempts):
        candidate_hx, candidate_hz = _random_css_check_matrices(
            code.n, code.k, rx=rx, seed=_seed(rng)
        )
        if attempt < strict_attempts and _visible_css_invariant_matrices(
            candidate_hx, candidate_hz, k=code.k
        ) != visible_invariant:
            continue

        candidate = _css_like(code, candidate_hx, candidate_hz)
        if _certificate(candidate) != invariant:
            return candidate

    raise RandomizeError(
        "Could not find a same-cheap-invariant candidate with a different CSS invariant."
    )


# --------------------------------------------------------------------------
# Negative partner constructors: local-Clifford equivalence
# --------------------------------------------------------------------------


def non_lc_equivalent_code(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    row_steps: int | None = None,
    max_attempts: int = 10_000,
) -> StabilizerCode:
    """Return a random same-``[[n, k]]`` code certified non-LC-equivalent.

    The certificate is the ordered support-projection rank profile. A local
    Clifford applies an invertible 2x2 transform to each qubit's X/Z column
    pair, so these ranks cannot change under LC-equivalence.
    """
    if max_attempts < 1:
        msg = "max_attempts must be positive."
        raise ValueError(msg)
    if code.k == code.n:
        raise RandomizeError(
            "No non-LC-equivalent stabilizer code exists for the trivial code."
        )
    if code.n == 1:
        raise RandomizeError("No non-LC-equivalent one-qubit stabilizer code exists.")

    rng = np.random.default_rng(seed)
    invariant = _lc_projection_rank_invariant(code)

    for _ in range(max_attempts):
        candidate_seed = _seed(rng)
        candidate = random_stabilizer_code(code.n, code.k, seed=candidate_seed)

        if _lc_projection_rank_invariant(candidate) != invariant:
            if row_steps is None:
                return candidate
            return StabilizerCode(
                _random_tableau_row_space_base_change(
                    candidate.generators, rng=rng, steps=row_steps
                )
            )

    raise RandomizeError(
        "Could not find a candidate with a different LC support-rank invariant."
    )


# --------------------------------------------------------------------------
# Candidate constructors
# --------------------------------------------------------------------------


def _css_like(code: CSSCode, hx: np.ndarray, hz: np.ndarray) -> CSSCode:
    """Return a CSS code with new check matrices but ``code``'s distance metadata."""
    return CSSCode(
        hx,
        hz,
        distance=code.distance,
        x_distance=code.x_distance,
        z_distance=code.z_distance,
    )


def _random_css_cnot_candidate_matrices(
    code: CSSCode,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small random network of physical CNOTs to a CSS code.

    For a CNOT ``control -> target``, X columns transform as
    ``target ^= control`` and Z columns contragrediently as
    ``control ^= target``. This preserves ``Hx @ Hz.T == 0`` and both check
    ranks, while generally leaving the physical-permutation orbit.
    """
    hx = np.asarray(code.Hx, dtype=np.int8).copy() % 2
    hz = np.asarray(code.Hz, dtype=np.int8).copy() % 2
    steps = int(rng.integers(1, min(9, code.n) + 1))
    for _ in range(steps):
        control, target = (int(q) for q in rng.choice(code.n, size=2, replace=False))
        hx[:, target] ^= hx[:, control]
        hz[:, control] ^= hz[:, target]
    return hx, hz


def _decoupled_css_column_permutation_candidate(
    code: CSSCode,
    *,
    rng: np.random.Generator,
) -> CSSCode | None:
    """Shuffle X and Z check columns independently while preserving cheap counts."""
    hx_permutation = rng.permutation(code.n)
    hz_permutation = rng.permutation(code.n)

    if np.array_equal(hx_permutation, hz_permutation):
        return None

    hx = np.asarray(code.Hx[:, hx_permutation], dtype=np.int8) % 2
    hz = np.asarray(code.Hz[:, hz_permutation], dtype=np.int8) % 2

    if hx.shape[0] and hz.shape[0] and np.any((hx @ hz.T) % 2):
        return None

    return _css_like(code, hx, hz)


def _random_anchor_base_code(
    n: int,
    k: int,
    *,
    rng: np.random.Generator,
    clifford_steps: int | None = None,
) -> StabilizerCode | None:
    """Return the random non-anchor block of an anchored ``[[n, k]]`` code."""
    r = n - k
    if r < 1:
        msg = "Anchored construction requires at least one stabilizer."
        raise ValueError(msg)
    if r == 1:
        return None

    base_seed = _seed(rng)
    return random_stabilizer_code(
        n - 1, k, seed=base_seed, clifford_steps=clifford_steps
    )


def _random_anchored_stabilizer_code(
    n: int,
    k: int,
    anchor: str,
    *,
    base_code: StabilizerCode | None,
    rng: np.random.Generator,
) -> StabilizerCode:
    """Return a randomized direct sum of ``base_code`` with a one-qubit anchor."""
    from .utils import _permute_tableau

    tableau = _anchored_stabilizer_tableau(n, k, anchor, base_code=base_code)
    permutation = tuple(int(q) for q in rng.permutation(n))
    permuted = _permute_tableau(tableau, permutation)
    randomized = _random_tableau_row_space_base_change(permuted, rng=rng)
    return StabilizerCode(randomized)


def _anchored_stabilizer_tableau(
    n: int,
    k: int,
    anchor: str,
    *,
    base_code: StabilizerCode | None,
):
    """Append an X/Z/Y one-qubit stabilizer to a base ``[[n-1, k]]`` code."""
    from src.core.pauli import StabilizerTableau

    if anchor not in {"X", "Z", "Y"}:
        msg = f"Unknown anchor {anchor!r}."
        raise ValueError(msg)

    r = n - k
    if r < 1:
        msg = "Anchored construction requires at least one stabilizer."
        raise ValueError(msg)

    matrix = np.zeros((r, 2 * n), dtype=np.int8)
    if base_code is not None:
        base_n = n - 1
        if base_code.n != base_n or base_code.k != k:
            msg = (
                f"Expected a base [[{base_n}, {k}]] code, "
                f"got [[{base_code.n}, {base_code.k}]]."
            )
            raise ValueError(msg)

        base_matrix = np.asarray(base_code.symplectic, dtype=np.int8) % 2
        expected_base_rows = r - 1
        base_rank = _rank_binary(base_matrix)
        if base_rank != expected_base_rows:
            msg = f"Expected base rank {expected_base_rows}, got {base_rank}."
            raise ValueError(msg)

        base_basis = np.asarray(mod2.row_basis(base_matrix), dtype=np.int8) % 2
        matrix[:expected_base_rows, :base_n] = base_basis[:, :base_n]
        matrix[:expected_base_rows, n : n + base_n] = base_basis[:, base_n:]
    elif r != 1:
        msg = (
            "A base code is required when the anchored construction has more "
            "than one stabilizer."
        )
        raise ValueError(msg)

    anchor_row = r - 1
    anchor_qubit = n - 1
    if anchor in {"X", "Y"}:
        matrix[anchor_row, anchor_qubit] = 1
    if anchor in {"Z", "Y"}:
        matrix[anchor_row, anchor_qubit + n] = 1

    return StabilizerTableau(matrix)


# --------------------------------------------------------------------------
# Certificates
# --------------------------------------------------------------------------


def _uses_additive_invariant(code: CSSCode) -> bool:
    """Return whether ``code`` is large and sparse enough for the additive profile."""
    rx = _rank_binary(code.Hx)
    rz = _rank_binary(code.Hz)
    return rx + rz > 20 and _is_sparse_css(code.Hx, code.Hz)


def _certificate(code: CSSCode) -> tuple[Any, ...]:
    """Return the strongest CSS permutation invariant affordable for ``code``.

    For small stabilizer ranks this is the exact stabilizer weight enumerator.
    Large sparse codes use the additive collision profile, which is the one that
    separates quasi-cyclic LDPC codes from dense random ones; the remaining
    large codes fall back to the support-rank profile.
    """
    rx = _rank_binary(code.Hx)
    rz = _rank_binary(code.Hz)
    if (rx, rz) in {(0, 0), (code.n, 0), (0, code.n)}:
        raise RandomizeError(
            "No non-equivalent CSS code exists with these small invariants."
        )

    if _uses_additive_invariant(code):
        return _css_additive_collision_invariant_matrices(code.Hx, code.Hz)
    if rx + rz > 20:
        return _css_support_rank_invariant_matrices(code.Hx, code.Hz)
    return _css_stabilizer_weight_enumerator_matrices(code.Hx, code.Hz)


def _projection_rank_invariant(code: StabilizerCode) -> tuple[int, int, int]:
    """Return permutation-invariant ranks of X, Z, and X+Z projections."""
    M = np.asarray(code.symplectic, dtype=np.uint8) & 1
    n = code.n
    return (
        _rank_binary(M[:, :n]),
        _rank_binary(M[:, n:]),
        _rank_binary(M[:, :n] ^ M[:, n:]),
    )


def _anchor_projection_rank_invariant(
    base_invariant: tuple[int, int, int],
    anchor: str,
) -> tuple[int, int, int]:
    """Return the projection-rank invariant after appending one anchor row."""
    if anchor not in {"X", "Z", "Y"}:
        msg = f"Unknown anchor {anchor!r}."
        raise ValueError(msg)

    rank_x, rank_z, rank_x_plus_z = base_invariant
    return (
        rank_x + int(anchor in {"X", "Y"}),
        rank_z + int(anchor in {"Z", "Y"}),
        rank_x_plus_z + int(anchor in {"X", "Z"}),
    )


def _lc_projection_rank_invariant(
    code: StabilizerCode, max_w: int = 3
) -> tuple[tuple[int, ...], ...]:
    """Return ordered subset projection ranks preserved by local Cliffords."""
    M = np.asarray(code.symplectic, dtype=np.uint8) & 1
    n = code.n

    return tuple(
        tuple(
            _rank_binary(M[:, [c for q in qubits for c in (q, q + n)]])
            for qubits in combinations(range(n), w)
        )
        for w in range(1, min(max_w, n) + 1)
    )


def _passes_stabilizer_hybrid_cheap_invariants(
    source: StabilizerCode,
    candidate: StabilizerCode,
) -> bool:
    """Return whether the stabilizer hybrid passes its constant-cost filters."""
    from src.hybrids import p_stab

    cheap_invariants = (
        p_stab.preserved_n,
        p_stab.preserved_k,
        p_stab.preserved_d,
        p_stab.preserved_rank,
        p_stab.preserved_number_zero_columns,
        p_stab.preserved_number_duplicate_columns,
    )
    return all(invariant(source, candidate) for invariant in cheap_invariants)


def _visible_css_invariant(
    code: CSSCode,
) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """Return the cheap CSS invariants used as early benchmark filters."""
    return _visible_css_invariant_matrices(code.Hx, code.Hz, k=code.k)


def _visible_css_invariant_matrices(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    k: int,
) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """Return the cheap CSS invariants from check matrices."""
    symplectic = _css_symplectic_matrix(hx, hz)
    n = hx.shape[1]
    return (
        n,
        k,
        _rank_binary(hx),
        _rank_binary(hz),
        int(np.count_nonzero(np.all(symplectic == 0, axis=0))),
        _duplicate_column_multiplicities(symplectic),
    )


def _css_symplectic_matrix(hx: np.ndarray, hz: np.ndarray) -> np.ndarray:
    hx = np.asarray(hx, dtype=np.int8) % 2
    hz = np.asarray(hz, dtype=np.int8) % 2
    x_padded = np.hstack([hx, np.zeros_like(hx)])
    z_padded = np.hstack([np.zeros_like(hz), hz])
    return np.vstack((x_padded, z_padded))


def _duplicate_column_multiplicities(matrix: np.ndarray) -> tuple[int, ...]:
    columns = [tuple(matrix[:, j].tolist()) for j in range(matrix.shape[1])]
    return tuple(sorted(Counter(columns).values()))


def _binary_columns_as_ints(matrix: np.ndarray) -> tuple[int, ...]:
    """Pack the columns of a binary matrix into basis-independent labels."""
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    values = [0] * matrix.shape[1]
    for row_index, row in enumerate(matrix):
        bit = 1 << row_index
        for column in np.flatnonzero(row):
            values[int(column)] |= bit
    return tuple(values)


def _additive_collision_profile(
    matrix: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return multiplicities of columns and two-/three-column sums.

    An invertible row-basis change maps every column through the same injective
    linear map, so it preserves both equality of columns and equality of their
    GF(2) pairwise sums. A column permutation merely reorders the pairs. The
    sorted multiplicities are therefore a classical-code permutation
    invariant. Triple sums also capture dependencies involving up to six
    columns, which occur frequently in sparse BB constructions.
    """
    columns = _binary_columns_as_ints(matrix)
    column_counts = Counter(columns)
    sum_counts: Counter[int] = Counter()
    for left in range(len(columns)):
        left_column = columns[left]
        for right in range(left + 1, len(columns)):
            sum_counts[left_column ^ columns[right]] += 1
    triple_sum_counts: Counter[int] = Counter()
    for left in range(len(columns)):
        left_column = columns[left]
        for middle in range(left + 1, len(columns)):
            partial_sum = left_column ^ columns[middle]
            for right in range(middle + 1, len(columns)):
                triple_sum_counts[partial_sum ^ columns[right]] += 1
    return (
        tuple(sorted(column_counts.values())),
        tuple(sorted(sum_counts.values())),
        tuple(sorted(triple_sum_counts.values())),
    )


def _css_additive_collision_invariant_matrices(
    hx: np.ndarray,
    hz: np.ndarray,
) -> tuple[Any, ...]:
    """Return a fast CSS permutation invariant for large sparse codes.

    X and Z stabilizer row spaces may undergo independent basis changes, while
    a physical permutation acts identically on their columns. Keeping the two
    additive profiles ordered is consequently invariant under CSS code
    permutation equivalence. It is especially effective for quasi-cyclic LDPC
    codes, whose repeated pair-sum collisions differ sharply from dense random
    codes with the same ranks and cheap visible invariants.
    """
    hx = np.asarray(mod2.row_basis(np.asarray(hx, dtype=np.uint8) & 1), dtype=np.uint8)
    hz = np.asarray(mod2.row_basis(np.asarray(hz, dtype=np.uint8) & 1), dtype=np.uint8)
    return _additive_collision_profile(hx), _additive_collision_profile(hz)


def _is_sparse_css(hx: np.ndarray, hz: np.ndarray, max_density: float = 0.2) -> bool:
    """Return whether both supplied check matrices have LDPC-like density."""
    matrices = (np.asarray(hx), np.asarray(hz))
    return all(
        matrix.size == 0 or np.count_nonzero(matrix) / matrix.size <= max_density
        for matrix in matrices
    )


def _css_support_rank_invariant_matrices(
    hx: np.ndarray,
    hz: np.ndarray,
    max_w: int = 3,
) -> tuple[Any, ...]:
    """Return a polynomial CSS invariant from check matrices."""
    hx = np.asarray(hx, dtype=np.uint8) & 1
    hz = np.asarray(hz, dtype=np.uint8) & 1
    n = hx.shape[1]
    rx = _rank_binary(hx)
    rz = _rank_binary(hz)

    profile = []
    for w in range(1, min(max_w, n) + 1):
        subset_ranks: Counter[tuple[int, int]] = Counter()
        subset_support_dims: Counter[tuple[int, int]] = Counter()

        for qubits in combinations(range(n), w):
            qubits_set = set(qubits)
            cols = list(qubits)
            complement_cols = [q for q in range(n) if q not in qubits_set]

            hx_rank = _rank_binary(hx[:, cols])
            hz_rank = _rank_binary(hz[:, cols])
            subset_ranks[(hx_rank, hz_rank)] += 1

            hx_support_dim = rx - _rank_binary(hx[:, complement_cols])
            hz_support_dim = rz - _rank_binary(hz[:, complement_cols])
            subset_support_dims[(hx_support_dim, hz_support_dim)] += 1

        profile.append(
            (
                tuple(sorted(subset_ranks.items())),
                tuple(sorted(subset_support_dims.items())),
            )
        )

    return (rx, rz, tuple(profile))


def _css_stabilizer_weight_enumerator_matrices(
    hx: np.ndarray,
    hz: np.ndarray,
) -> tuple[tuple[tuple[int, int, int, int], int], ...]:
    """Return the exact CSS stabilizer weight enumerator from check matrices."""
    from .utils import _row_space_words

    hx = np.asarray(hx, dtype=np.uint8) & 1
    hz = np.asarray(hz, dtype=np.uint8) & 1
    n = hx.shape[1]
    x_words = _row_space_words(hx)
    z_words = _row_space_words(hz)

    enumerator: dict[tuple[int, int, int, int], int] = {}
    for x_word in x_words:
        for z_word in z_words:
            both = int(np.count_nonzero(x_word & z_word))
            x_only = int(np.count_nonzero(x_word)) - both
            z_only = int(np.count_nonzero(z_word)) - both
            key = (n - x_only - z_only - both, x_only, z_only, both)
            enumerator[key] = enumerator.get(key, 0) + 1

    return tuple(sorted(enumerator.items()))
