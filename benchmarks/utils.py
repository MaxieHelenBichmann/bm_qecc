"""Deterministic constructors for randomized benchmark codes and pairs.

The public helpers in this module generate reproducible benchmark inputs from
an optional NumPy seed. They are deliberately *not* uniform samplers over code
equivalence classes. Negative-pair constructors return only candidates
separated by a documented invariant and raise `RandomizeError` when no
such candidate is found within their search budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from src.core.pauli import StabilizerTableau
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


class RandomizeError(ValueError):
    """Raised when a randomization helper fails to find a suitable code."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def random_stabilizer_code(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    clifford_steps: int | None = None,
) -> StabilizerCode:
    """Return a seeded random-looking stabilizer code with parameters ``[[n, k]]``.

    The construction places ``n-k`` single-qubit Z stabilizers on a seeded
    random subset of distinct physical qubits, so the initial generators are
    linearly independent without privileging the first ``n-k`` coordinates.
    It then applies seeded random Clifford layers. Every layer applies a random
    local Clifford to every qubit and a random matching of two-qubit entangling
    gates, so every qubit receives a local gate and all but at most one qubit
    per layer participates in an entangling gate. Random matchings rotate the
    unmatched coordinate when ``n`` is odd.

    ``clifford_steps`` is the number of full layers and defaults to four. This
    is meant for benchmark instances, not for uniform sampling from all
    stabilizer codes or Clifford orbits.
    """
    if n < 1:
        msg = "n must be at least 1."
        raise ValueError(msg)
    if not 0 <= k <= n:
        msg = f"k must satisfy 0 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    num_stabilizers = n - k
    occupied_qubits = tuple(
        int(qubit) for qubit in rng.choice(n, size=num_stabilizers, replace=False)
    )
    generators = [
        "I" * qubit + "Z" + "I" * (n - qubit - 1)
        for qubit in occupied_qubits
    ]
    tableau = (
        StabilizerTableau.from_pauli_strings(generators)
        if generators
        else StabilizerTableau.empty(n)
    )

    steps = clifford_steps if clifford_steps is not None else 4
    if steps < 0:
        msg = "clifford_steps must be non-negative."
        raise ValueError(msg)

    for _ in range(steps):
        _apply_random_clifford_layer(tableau, rng)

    return StabilizerCode(_without_phases(tableau))


def random_css_code(
    n: int,
    k: int,
    rx: int | None = None,
    seed: int | None = None,
) -> CSSCode:
    """Return a seeded random-looking CSS code with parameters ``[[n, k]]``.

    ``rx`` fixes the X-check rank; by default the ``n-k`` checks are split as
    evenly as possible between X and Z. The matrices are full row rank and
    satisfy ``Hx @ Hz.T == 0`` over GF(2). Sampling is intended for benchmark
    construction rather than uniform sampling from all CSS codes.
    """
    Hx, Hz = _random_css_check_matrices(n, k, rx=rx, seed=seed)
    return CSSCode(Hx=Hx, Hz=Hz)


def _random_css_check_matrices(
    n: int,
    k: int,
    rx: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random full-rank CSS check matrices without building a code object."""
    if not 0 <= k <= n:
        raise ValueError("Require 0 <= k <= n.")

    total_checks = n - k

    if rx is None:
        rx = total_checks // 2
    rz = total_checks - rx

    if not 0 <= rx <= n or not 0 <= rz <= n:
        raise ValueError("Require 0 <= rx <= n and 0 <= rz <= n.")

    rng = np.random.default_rng(seed)

    if rx == 0:
        Hx = np.zeros((0, n), dtype=np.int8)
    else:
        while True:
            Hx = rng.integers(0, 2, size=(rx, n), dtype=np.int8)
            if mod2.rank(Hx) == rx:
                break

    ker_hx = mod2.nullspace(Hx)

    if hasattr(ker_hx, "toarray"):
        ker_hx = ker_hx.toarray()

    ker_hx = np.asarray(ker_hx, dtype=np.int8) % 2

    if rz > ker_hx.shape[0]:
        raise RandomizeError(
            f"Cannot construct {rz} independent Z checks: "
            f"ker(Hx) has dimension {ker_hx.shape[0]}."
        )

    if rz == 0:
        Hz = np.zeros((0, n), dtype=np.int8)
    else:
        while True:
            coeffs = rng.integers(0, 2, size=(rz, ker_hx.shape[0]), dtype=np.int8)
            if mod2.rank(coeffs) == rz:
                break
        Hz = np.asarray(coeffs @ ker_hx, dtype=np.int8) % 2

    return Hx, Hz


def permutation_equivalent_code(
    code: StabilizerCode, seed: int | None = None
) -> StabilizerCode:
    """Return ``code`` after a seeded qubit permutation and row-basis change."""
    rng = np.random.default_rng(seed)
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    row_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    permutation = _random_permutation(code.n, seed=permutation_seed)
    base_changed_tableau = _random_tableau_row_space_base_change(
        code.generators, seed=row_seed
    )
    return StabilizerCode(
        _permute_tableau(base_changed_tableau, permutation), distance=code.distance
    )


def permutation_equivalent_css_code(code: CSSCode, seed: int | None = None) -> CSSCode:
    """Return ``code`` after a seeded qubit permutation and CSS basis changes."""
    rng = np.random.default_rng(seed)
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    x_row_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    z_row_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    permutation = _random_permutation(code.n, seed=permutation_seed)
    hx = _random_row_space_base_change(code.Hx, seed=x_row_seed)[:, permutation]
    hz = _random_row_space_base_change(code.Hz, seed=z_row_seed)[:, permutation]
    return CSSCode(
        hx,
        hz,
        distance=code.distance,
        x_distance=code.x_distance,
        z_distance=code.z_distance,
    )


def non_permutation_equivalent_css_code(
    code: CSSCode, seed: int | None = None
) -> CSSCode:
    """Return a CSS code certified non-equivalent by permutation invariants.

    For small stabilizer ranks this uses the exact stabilizer weight enumerator.
    For larger ranks it uses a polynomial-time CSS support-rank profile. In both
    cases candidates must preserve the very cheap invariants used as early
    filters by the benchmark solvers, so random negative pairs are not rejected
    just because of zero columns or duplicate-column multiplicities.
    """
    rng = np.random.default_rng(seed)
    rx = _rank_binary(code.Hx)
    rz = _rank_binary(code.Hz)
    use_additive_invariant = rx + rz > 20 and _is_sparse_css(code.Hx, code.Hz)
    if (rx, rz) in {(0, 0), (code.n, 0), (0, code.n)}:
        raise RandomizeError(
            "No non-equivalent CSS code exists with these small invariants."
        )

    visible_invariant = _visible_css_invariant(code)
    if use_additive_invariant:
        invariant = _css_additive_collision_invariant_matrices(code.Hx, code.Hz)
    elif rx + rz > 20:
        invariant = _css_support_rank_invariant_matrices(code.Hx, code.Hz)
    else:
        invariant = _css_stabilizer_weight_enumerator_matrices(code.Hx, code.Hz)

    # Coupled CNOT column operations preserve CSS orthogonality, dimensions,
    # and much of the source code's structure. For large structured codes this
    # produces substantially more representative negatives than replacing the
    # code with an unrelated dense random sample.
    if use_additive_invariant:
        for _ in range(10):
            candidate_hx, candidate_hz = _random_css_cnot_candidate_matrices(
                code, rng=rng
            )
            candidate = CSSCode(
                candidate_hx,
                candidate_hz,
                distance=code.distance,
                x_distance=code.x_distance,
                z_distance=code.z_distance,
            )
            if _visible_css_invariant(candidate) != visible_invariant:
                continue
            other_invariant = _css_additive_collision_invariant_matrices(
                candidate_hx, candidate_hz
            )
            if other_invariant != invariant:
                return candidate

        # Some highly symmetric codes (notably BB [[90,8]]) have repeated
        # column multiplicities that essentially no nontrivial CNOT preserves.
        # Prefer a nearby CNOT-derived negative over an unrelated random code.
        for _ in range(100):
            candidate_hx, candidate_hz = _random_css_cnot_candidate_matrices(
                code, rng=rng
            )
            candidate = CSSCode(
                candidate_hx,
                candidate_hz,
                distance=code.distance,
                x_distance=code.x_distance,
                z_distance=code.z_distance,
            )
            if _visible_css_invariant(candidate) != visible_invariant:
                continue
            other_invariant = _css_additive_collision_invariant_matrices(
                candidate_hx, candidate_hz
            )
            if other_invariant != invariant:
                return candidate

    if rx and rz:
        for _ in range(500):
            decoupled_candidate = _decoupled_css_column_permutation_candidate(
                code, rng=rng
            )
            if decoupled_candidate is None:
                continue
            candidate = decoupled_candidate

            if use_additive_invariant:
                other_invariant = _css_additive_collision_invariant_matrices(
                    candidate.Hx, candidate.Hz
                )
            elif rx + rz > 20:
                other_invariant = _css_support_rank_invariant_matrices(
                    candidate.Hx, candidate.Hz
                )
            else:
                other_invariant = _css_stabilizer_weight_enumerator_matrices(
                    candidate.Hx, candidate.Hz
                )

            if other_invariant != invariant:
                return candidate

    for attempt in range(10_000):
        candidate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        candidate_hx, candidate_hz = _random_css_check_matrices(
            code.n, code.k, rx=rx, seed=candidate_seed
        )

        if (
            use_additive_invariant or attempt < 1_000
        ) and _visible_css_invariant_matrices(
            candidate_hx, candidate_hz, k=code.k
        ) != visible_invariant:
            continue

        candidate = CSSCode(
            candidate_hx,
            candidate_hz,
            distance=code.distance,
            x_distance=code.x_distance,
            z_distance=code.z_distance,
        )
        if use_additive_invariant:
            other_invariant = _css_additive_collision_invariant_matrices(
                candidate_hx, candidate_hz
            )
        elif rx + rz > 20:
            other_invariant = _css_support_rank_invariant_matrices(
                candidate_hx, candidate_hz
            )
        else:
            other_invariant = _css_stabilizer_weight_enumerator_matrices(
                candidate_hx, candidate_hz
            )

        if other_invariant != invariant:
            return candidate

    raise RandomizeError(
        "Could not find a same-cheap-invariant candidate with a different CSS invariant."
    )


def non_permutation_equivalent_stabilizer_code(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    clifford_steps: int | None = None,
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
    for _ in range(1_000):
        candidate = lc_equivalent_code(
            code, seed=int(rng.integers(0, np.iinfo(np.int32).max))
        )
        if not _passes_stabilizer_hybrid_cheap_invariants(code, candidate):
            continue
        if _projection_rank_invariant(candidate)[2] != invariant:
            return candidate

    raise RandomizeError(
        "Could not find a cheap-filter-preserving candidate with a different X+Z projection rank."
    )


def random_permuted_stabilizer_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    clifford_steps: int | None = None,
) -> tuple[StabilizerCode, StabilizerCode]:
    """Return a seeded random code together with a permuted equivalent copy."""
    rng = np.random.default_rng(seed)
    code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    code = random_stabilizer_code(n, k, seed=code_seed, clifford_steps=clifford_steps)
    permutation = _random_permutation(n, seed=permutation_seed)
    return code, _permute_stabilizer_code(code, permutation)


def random_non_permuted_stabilizer_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    clifford_steps: int | None = None,
    max_attempts: int = 10_000,
) -> tuple[StabilizerCode, StabilizerCode]:
    """Return a seeded stabilizer pair certified non-permutation-equivalent.

    Candidate bases are retried up to ``max_attempts`` because the invariant
    certificate is not available for every randomly sampled code.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        other_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        code = random_stabilizer_code(
            n, k, seed=code_seed, clifford_steps=clifford_steps
        )
        try:
            return code, non_permutation_equivalent_stabilizer_code(
                code,
                seed=other_seed,
                clifford_steps=clifford_steps,
            )
        except RandomizeError:
            continue

    raise RandomizeError(
        f"Could not generate non-permuted stabilizer pair after {max_attempts} attempts."
    )


def random_non_permuted_css_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    max_attempts: int = 10_000,
) -> tuple[CSSCode, CSSCode]:
    """Return a seeded CSS pair certified non-permutation-equivalent.

    Candidate bases are retried up to ``max_attempts`` because the invariant
    certificate is not available for every randomly sampled code.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        other_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        total_checks = n - k
        if total_checks >= 2:
            rx = int(rng.integers(1, total_checks))
        else:
            rx = int(rng.integers(0, total_checks + 1))
        code = random_css_code(n, k, rx, seed=code_seed)
        try:
            return code, non_permutation_equivalent_css_code(code, seed=other_seed)
        except RandomizeError:
            continue

    raise RandomizeError(
        f"Could not generate non-permuted CSS pair after {max_attempts} attempts."
    )


def random_permuted_css_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
) -> tuple[CSSCode, CSSCode]:
    """Return a seeded random CSS code and a permutation-equivalent copy."""
    rng = np.random.default_rng(seed)
    code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    rx = int(rng.integers(0, n - k + 1))

    code = random_css_code(n, k, rx, seed=code_seed)
    return code, permutation_equivalent_css_code(code, permutation_seed)


def lc_equivalent_code(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    row_steps: int | None = None,
) -> StabilizerCode:
    """Return an LC-equivalent copy after local Cliffords and row operations.

    At least one nonidentity local Clifford is applied when the code has qubits.
    ``row_steps`` controls the subsequent seeded generator-basis randomization.
    """
    rng = np.random.default_rng(seed)
    tableau = code.generators.copy()
    local_cliffords = [str(rng.choice(_LOCAL_CLIFFORDS)) for _ in range(tableau.n)]

    if tableau.n > 0 and all(
        local_clifford == "I" for local_clifford in local_cliffords
    ):
        local_cliffords[int(rng.integers(0, tableau.n))] = str(
            rng.choice(_LOCAL_CLIFFORDS[1:])
        )

    for qubit, local_clifford in enumerate(local_cliffords):
        _apply_local_clifford(tableau, local_clifford, qubit)

    base_changed_tableau = _random_tableau_row_space_base_change(
        tableau, rng=rng, steps=row_steps
    )
    return StabilizerCode(base_changed_tableau, distance=code.distance)


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
        candidate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
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


def non_lc_css_code(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    max_attempts: int = 10_000,
    max_exact_rank: int = 16,
) -> StabilizerCode:
    """Return a same-``[[n,k]]`` code certified outside every CSS LC orbit.

    A locally rank-one stabilizer subspace has, on each qubit, at most one
    nonidentity Pauli type among all its elements.  Local Cliffords preserve
    this property.  A rank-``r`` CSS stabilizer is the direct sum of its pure-X
    and pure-Z subspaces, so at least one locally rank-one subspace has dimension
    ``ceil(r / 2)``.  Absence of such a subspace is therefore a sound negative
    certificate independent of the LC-CSS decision algorithms.

    Up to ``max_exact_rank`` the invariant is checked exactly.  Above that, the
    generator uses an additive upper bound from invariant-certified disjoint
    components, avoiding enumeration of the full stabilizer group.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    if max_exact_rank < 1:
        raise ValueError("max_exact_rank must be positive.")

    if code.n <= 6:
        # At these sizes the exact 6^n LC orbit is small enough to use as the
        # generator certificate, so every constructed candidate is checked.
        from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce

        rng = np.random.default_rng(seed)
        for _ in range(max_attempts):
            candidate, _ = _structured_non_lc_css_candidate(code.n, code.k, rng=rng)
            lc_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            candidate = lc_equivalent_code(candidate, seed=lc_seed)
            if not is_lceq_css_bruteforce(candidate):
                return candidate
        raise RandomizeError(
            "Could not find a brute-force-certified non-LC-CSS candidate."
        )

    stabilizer_rank = code.n - code.k
    if stabilizer_rank < 3:
        raise RandomizeError(
            "The locally-rank-one invariant cannot certify negatives below stabilizer rank 3."
        )
    rng = np.random.default_rng(seed)
    target_dimension = (stabilizer_rank + 1) // 2
    for _ in range(max_attempts):
        candidate, dimension_upper_bound = _structured_non_lc_css_candidate(
            code.n, code.k, rng=rng
        )
        certified = dimension_upper_bound < target_dimension
        if not certified and stabilizer_rank <= max_exact_rank:
            certified = not _has_locally_rank_one_subspace(candidate, target_dimension)
        if certified:
            lc_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            return lc_equivalent_code(candidate, seed=lc_seed)

    raise RandomizeError(
        "Could not find a candidate excluded from every CSS LC orbit by the "
        "locally-rank-one invariant."
    )


def _structured_non_lc_css_candidate(
    n: int,
    k: int,
    *,
    rng: np.random.Generator,
) -> tuple[StabilizerCode, int]:
    """Build a candidate and an upper bound on its locally rank-one dimension."""
    blocks: list[StabilizerCode] = []
    dimension_upper_bound = 0
    remaining_n = n
    remaining_k = k
    remaining_rank = n - k

    perfect = StabilizerCode(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    cycle_state = StabilizerCode(["XZIIZ", "ZXZII", "IZXZI", "IIZXZ", "ZIIZX"])

    while remaining_n >= 5 and remaining_k >= 1 and remaining_rank >= 4:
        blocks.append(perfect)
        # Exhaustive five-qubit calculation: its maximum dimension is one.
        dimension_upper_bound += 1
        remaining_n -= 5
        remaining_k -= 1
        remaining_rank -= 4

    while remaining_n >= 5 and remaining_rank >= 5:
        blocks.append(cycle_state)
        # Exhaustive five-qubit calculation: its maximum dimension is two.
        dimension_upper_bound += 2
        remaining_n -= 5
        remaining_rank -= 5

    if remaining_n:
        remainder_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        blocks.append(
            random_stabilizer_code(remaining_n, remaining_k, seed=remainder_seed)
        )
        # The stabilizer rank is always a valid (possibly loose) upper bound.
        dimension_upper_bound += remaining_rank

    if not blocks:
        candidate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return random_stabilizer_code(n, k, seed=candidate_seed), n - k
    return _direct_sum_stabilizer_codes(blocks), dimension_upper_bound


def _direct_sum_stabilizer_codes(codes: Sequence[StabilizerCode]) -> StabilizerCode:
    """Return the tensor product of stabilizer codes on disjoint qubits."""
    total_n = sum(code.n for code in codes)
    total_rank = sum(code.n - code.k for code in codes)
    matrix = np.zeros((total_rank, 2 * total_n), dtype=np.int8)
    row_offset = 0
    qubit_offset = 0
    for code in codes:
        rows = code.n - code.k
        matrix[row_offset : row_offset + rows, qubit_offset : qubit_offset + code.n] = (
            code.symplectic[:, : code.n]
        )
        matrix[
            row_offset : row_offset + rows,
            total_n + qubit_offset : total_n + qubit_offset + code.n,
        ] = code.symplectic[:, code.n :]
        row_offset += rows
        qubit_offset += code.n
    return StabilizerCode(StabilizerTableau(matrix))


def lc_equivalent_code_and_log_ops(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    row_steps: int | None = None,
) -> StabilizerCode:
    """Return an LC-equivalent copy while transforming logical operators too.

    Unlike :func:`lc_equivalent_code`, this helper applies the same local
    Cliffords to the stored logical X and Z operators instead of recomputing
    them from the transformed stabilizer generators.
    """
    rng = np.random.default_rng(seed)
    tableau = code.generators.copy()
    x_logicals = code.x_logicals.copy()
    z_logicals = code.z_logicals.copy()
    local_cliffords = [str(rng.choice(_LOCAL_CLIFFORDS)) for _ in range(tableau.n)]

    if tableau.n > 0 and all(
        local_clifford == "I" for local_clifford in local_cliffords
    ):
        local_cliffords[int(rng.integers(0, tableau.n))] = str(
            rng.choice(_LOCAL_CLIFFORDS[1:])
        )

    for qubit, local_clifford in enumerate(local_cliffords):
        _apply_local_clifford(tableau, local_clifford, qubit)
        _apply_local_clifford(x_logicals, local_clifford, qubit)
        _apply_local_clifford(z_logicals, local_clifford, qubit)

    base_changed_tableau = _random_tableau_row_space_base_change(
        tableau, rng=rng, steps=row_steps
    )
    return StabilizerCode(
        generators=base_changed_tableau,
        distance=code.distance,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
    )


_LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")


def _random_permutation(n: int, *, seed: int | None = None) -> tuple[int, ...]:
    """Return a seeded random permutation of ``range(n)``."""
    if n < 0:
        msg = "n must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    return tuple(int(q) for q in rng.permutation(n))


def _rank_binary(matrix: np.ndarray) -> int:
    matrix = np.asarray(matrix, dtype=np.int8) % 2
    return 0 if matrix.size == 0 or matrix.shape[0] == 0 else int(mod2.rank(matrix))


def _projection_rank_invariant(code: StabilizerCode) -> tuple[int, int, int]:
    """Return permutation-invariant ranks of X, Z, and X+Z projections."""
    M = np.asarray(code.symplectic, dtype=np.uint8) & 1
    n = code.n
    return (
        _rank_binary(M[:, :n]),
        _rank_binary(M[:, n:]),
        _rank_binary(M[:, :n] ^ M[:, n:]),
    )


def _visible_css_invariant(
    code: CSSCode,
) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """Return the cheap CSS invariants used as early benchmark filters."""
    return _visible_css_invariant_matrices(code.Hx, code.Hz, k=code.k)


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

    return CSSCode(
        hx,
        hz,
        distance=code.distance,
        x_distance=code.x_distance,
        z_distance=code.z_distance,
    )


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


def _has_locally_rank_one_subspace(code: StabilizerCode, target_dimension: int) -> bool:
    """Exactly test for a locally rank-one subspace of the requested dimension.

    Nonzero stabilizers are vertices of an implicit compatibility graph.  Two
    vertices are compatible precisely when they never contain two different
    nonidentity Paulis on the same qubit.  A linearly independent clique is a
    basis of a locally rank-one subspace.  The graph is generated lazily to
    avoid a quadratic compatibility matrix.
    """
    matrix = np.asarray(mod2.row_basis(code.symplectic), dtype=np.uint8) % 2
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    rank = matrix.shape[0]
    if target_dimension <= 0:
        return True
    if target_dimension > rank:
        return False

    n = code.n
    row_x = [_binary_row_to_int(row[:n]) for row in matrix]
    row_z = [_binary_row_to_int(row[n:]) for row in matrix]
    count = 1 << rank
    xs = [0] * count
    zs = [0] * count
    for coefficient in range(1, count):
        bit = coefficient & -coefficient
        index = bit.bit_length() - 1
        previous = coefficient ^ bit
        xs[coefficient] = xs[previous] ^ row_x[index]
        zs[coefficient] = zs[previous] ^ row_z[index]

    def compatible(left: int, right: int) -> bool:
        left_support = xs[left] | zs[left]
        right_support = xs[right] | zs[right]
        different = (xs[left] ^ xs[right]) | (zs[left] ^ zs[right])
        return (left_support & right_support & different) == 0

    def add_to_basis(vector: int, basis: tuple[int, ...]) -> tuple[int, ...] | None:
        reduced = vector
        mutable = list(basis)
        for pivot in mutable:
            reduced = min(reduced, reduced ^ pivot)
        if reduced == 0:
            return None
        for i, pivot in enumerate(mutable):
            mutable[i] = min(pivot, pivot ^ reduced)
        mutable.append(reduced)
        mutable.sort(reverse=True)
        return tuple(mutable)

    def candidate_rank(
        candidates: list[int], basis: tuple[int, ...], needed: int
    ) -> int:
        working = basis
        gained = 0
        for vector in candidates:
            extended = add_to_basis(vector, working)
            if extended is not None:
                working = extended
                gained += 1
                if gained >= needed:
                    break
        return gained

    def search(candidates: list[int], basis: tuple[int, ...]) -> bool:
        needed = target_dimension - len(basis)
        if needed <= 0:
            return True
        if candidate_rank(candidates, basis, needed) < needed:
            return False

        while candidates:
            vector = candidates.pop()
            extended = add_to_basis(vector, basis)
            if extended is None:
                continue
            compatible_candidates = [
                other for other in candidates if compatible(vector, other)
            ]
            if search(compatible_candidates, extended):
                return True
            if candidate_rank(candidates, basis, needed) < needed:
                return False
        return False

    return search(list(range(1, count)), ())


def _binary_row_to_int(row: np.ndarray) -> int:
    """Pack a little-endian binary row into a Python integer."""
    value = 0
    for index in np.flatnonzero(row):
        value |= 1 << int(index)
    return value


def _row_space_words(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    basis = np.asarray(mod2.row_basis(matrix), dtype=np.uint8) % 2
    if basis.size == 0:
        basis = np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    if basis.ndim == 1:
        basis = basis.reshape(1, -1)

    rank = basis.shape[0]
    if rank == 0:
        return np.zeros((1, matrix.shape[1]), dtype=np.uint8)

    words = np.zeros((1 << rank, matrix.shape[1]), dtype=np.uint8)
    num_words = 1
    for row in basis:
        words[num_words : 2 * num_words] = words[:num_words] ^ row
        num_words *= 2
    return words


def _random_row_space_base_change(
    matrix: np.ndarray,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    steps: int | None = None,
) -> np.ndarray:
    """Return ``matrix`` after seeded invertible row operations over GF(2).

    This keeps the generated row space unchanged, but may replace the rows by
    different linear combinations. For stabilizer generator matrices, this gives
    an equivalent generating set for the same stabilizer group.
    """
    changed = np.asarray(matrix, dtype=np.int8).copy() % 2
    if changed.ndim != 2:
        msg = f"Expected a 2D matrix, got shape {changed.shape}."
        raise ValueError(msg)

    n_rows = changed.shape[0]
    num_steps = steps if steps is not None else 4 * n_rows
    if num_steps < 0:
        msg = "steps must be non-negative."
        raise ValueError(msg)

    if n_rows < 2:
        return changed

    if rng is None:
        rng = np.random.default_rng(seed)

    for _ in range(num_steps):
        row_a, row_b = rng.choice(n_rows, size=2, replace=False)
        row_a = int(row_a)
        row_b = int(row_b)
        if bool(rng.integers(0, 2)):
            changed[[row_a, row_b]] = changed[[row_b, row_a]]
        else:
            changed[row_b] ^= changed[row_a]

    return changed


def _random_tableau_row_space_base_change(
    tableau: StabilizerTableau,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    steps: int | None = None,
) -> StabilizerTableau:
    """Return ``tableau`` after seeded invertible row operations on its symplectic rows."""
    if rng is None:
        rng = np.random.default_rng(seed)

    changed = _random_row_space_base_change(
        tableau.tableau.matrix, rng=rng, steps=steps
    )
    return StabilizerTableau(changed)


def _without_phases(tableau: StabilizerTableau) -> StabilizerTableau:
    """Return a copy of ``tableau`` with zero phases."""
    return StabilizerTableau(tableau.tableau.matrix.copy())


def _permute_tableau(
    tableau: StabilizerTableau, permutation: Sequence[int]
) -> StabilizerTableau:
    """Return a copy of ``tableau`` with physical qubits permuted."""
    permutation = _checked_permutation(tableau.n, permutation)
    columns = list(permutation) + [q + tableau.n for q in permutation]
    return StabilizerTableau(tableau.tableau.matrix[:, columns].copy())


def _permute_stabilizer_code(
    code: StabilizerCode, permutation: Sequence[int]
) -> StabilizerCode:
    """Return a copy of ``code`` with physical qubits permuted in generators, but logical operators are recomputed."""
    return StabilizerCode(
        _permute_tableau(code.generators.copy(), permutation), distance=code.distance
    )


def _apply_local_clifford(
    tableau: StabilizerTableau, local_clifford: str, qubit: int
) -> None:
    """Apply one single-qubit Clifford representative to ``qubit`` in-place."""
    if local_clifford not in _LOCAL_CLIFFORDS:
        msg = f"Unknown local Clifford {local_clifford!r}."
        raise ValueError(msg)

    for gate in reversed(local_clifford):
        if gate == "H":
            tableau.apply_h(qubit)
        elif gate == "S":
            tableau.apply_s(qubit)


def _apply_random_clifford_layer(
    tableau: StabilizerTableau, rng: np.random.Generator
) -> None:
    """Apply one seeded local-plus-entangling Clifford layer in-place.

    Local Cliffords randomize the Pauli axes before a fresh random matching is
    entangled by CNOT or CZ gates. A random CNOT orientation is used for each
    matched pair. The construction guarantees interaction coverage per layer;
    it is not a uniform sampler from the Clifford group.
    """
    for qubit in range(tableau.n):
        local_clifford = str(rng.choice(_LOCAL_CLIFFORDS))
        _apply_local_clifford(tableau, local_clifford, qubit)

    matching = tuple(int(qubit) for qubit in rng.permutation(tableau.n))
    for index in range(0, tableau.n - 1, 2):
        left, right = matching[index], matching[index + 1]
        if bool(rng.integers(0, 2)):
            if bool(rng.integers(0, 2)):
                left, right = right, left
            tableau.apply_cx(left, right)
        else:
            tableau.apply_cz(left, right)


def _apply_random_clifford_gate(
    tableau: StabilizerTableau, rng: np.random.Generator
) -> None:
    """Apply one seeded random Clifford gate (not only generators) to a tableau in-place."""
    if tableau.n == 1:
        gate = rng.choice(("h", "s", "sdg", "x", "y", "z"))
    else:
        gate = rng.choice(("h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"))

    if gate == "h":
        tableau.apply_h(int(rng.integers(0, tableau.n)))
    elif gate == "s":
        tableau.apply_s(int(rng.integers(0, tableau.n)))
    elif gate == "sdg":
        tableau.apply_sdg(int(rng.integers(0, tableau.n)))
    elif gate == "x":
        tableau.apply_x(int(rng.integers(0, tableau.n)))
    elif gate == "y":
        tableau.apply_y(int(rng.integers(0, tableau.n)))
    elif gate == "z":
        tableau.apply_z(int(rng.integers(0, tableau.n)))
    elif gate == "cx":
        ctrl, target = rng.choice(tableau.n, size=2, replace=False)
        tableau.apply_cx(int(ctrl), int(target))
    elif gate == "cz":
        ctrl, target = rng.choice(tableau.n, size=2, replace=False)
        tableau.apply_cz(int(ctrl), int(target))
    elif gate == "swap":
        ctrl, target = rng.choice(tableau.n, size=2, replace=False)
        tableau.apply_swap(int(ctrl), int(target))
    else:
        ctrl, target = rng.choice(tableau.n, size=2, replace=False)
        tableau.apply_cx(int(ctrl), int(target))


def _checked_permutation(n: int, permutation: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a physical-qubit permutation."""
    normalized = tuple(int(q) for q in permutation)
    if sorted(normalized) != list(range(n)):
        msg = f"Expected a permutation of 0..{n - 1}, got {list(permutation)}."
        raise ValueError(msg)
    return normalized
