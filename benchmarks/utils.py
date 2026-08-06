"""Small helpers for constructing benchmark inputs."""

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

    The construction starts with ``n-k`` single-qubit Z stabilizers and applies a
    seeded random Clifford circuit to the stabilizer tableau. This is meant for
    benchmark instances, not for uniform sampling from all stabilizer codes.
    """
    if n < 1:
        msg = "n must be at least 1."
        raise ValueError(msg)
    if not 0 <= k <= n:
        msg = f"k must satisfy 0 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)

    num_stabilizers = n - k
    generators = ["I" * q + "Z" + "I" * (n - q - 1) for q in range(num_stabilizers)]
    tableau = StabilizerTableau.from_pauli_strings(generators) if generators else StabilizerTableau.empty(n)

    rng = np.random.default_rng(seed)
    steps = clifford_steps if clifford_steps is not None else 4 * n
    if steps < 0:
        msg = "clifford_steps must be non-negative."
        raise ValueError(msg)

    for _ in range(steps):
        _apply_random_clifford_gate(tableau, rng)

    return StabilizerCode(_without_phases(tableau))


def random_css_code(
    n: int,
    k: int,
    rx: int | None = None,
    seed: int | None = None,
) -> CSSCode:
    """
    Generate random CSS parity-check matrices Hx, Hz over GF(2).
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

def permutation_equivalent_code(code: StabilizerCode, seed: int | None = None) -> StabilizerCode:
    """Return a permuted equivalent code to the given code."""
    rng = np.random.default_rng(seed)
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    row_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    permutation = _random_permutation(code.n, seed=permutation_seed)
    base_changed_tableau = _random_tableau_row_space_base_change(code.generators, seed=row_seed)
    return StabilizerCode(_permute_tableau(base_changed_tableau, permutation), distance=code.distance)


def permutation_equivalent_css_code(code: CSSCode, seed: int | None = None) -> CSSCode:
    """Return a permuted equivalent CSS code to the given code."""
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

def non_permutation_equivalent_css_code(code: CSSCode, seed: int | None = None) -> CSSCode:
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
        raise RandomizeError("No non-equivalent CSS code exists with these small invariants.")

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
            candidate_hx, candidate_hz = _random_css_cnot_candidate_matrices(code, rng=rng)
            candidate = CSSCode(
                candidate_hx,
                candidate_hz,
                distance=code.distance,
                x_distance=code.x_distance,
                z_distance=code.z_distance,
            )
            if _visible_css_invariant(candidate) != visible_invariant:
                continue
            other_invariant = _css_additive_collision_invariant_matrices(candidate_hx, candidate_hz)
            if other_invariant != invariant:
                return candidate

        # Some highly symmetric codes (notably BB [[90,8]]) have repeated
        # column multiplicities that essentially no nontrivial CNOT preserves.
        # Prefer a nearby CNOT-derived negative over an unrelated random code.
        for _ in range(100):
            candidate_hx, candidate_hz = _random_css_cnot_candidate_matrices(code, rng=rng)
            candidate = CSSCode(
                candidate_hx,
                candidate_hz,
                distance=code.distance,
                x_distance=code.x_distance,
                z_distance=code.z_distance,
            )
            if _visible_css_invariant(candidate) != visible_invariant:
                continue
            other_invariant = _css_additive_collision_invariant_matrices(candidate_hx, candidate_hz)
            if other_invariant != invariant:
                return candidate

    if rx and rz:
        for _ in range(500):
            candidate = _decoupled_css_column_permutation_candidate(code, rng=rng)
            if candidate is None:
                continue

            if use_additive_invariant:
                other_invariant = _css_additive_collision_invariant_matrices(candidate.Hx, candidate.Hz)
            elif rx + rz > 20:
                other_invariant = _css_support_rank_invariant_matrices(candidate.Hx, candidate.Hz)
            else:
                other_invariant = _css_stabilizer_weight_enumerator_matrices(candidate.Hx, candidate.Hz)

            if other_invariant != invariant:
                return candidate
        
    for attempt in range(10_000):
        candidate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        candidate_hx, candidate_hz = _random_css_check_matrices(code.n, code.k, rx=rx, seed=candidate_seed)

        if _visible_css_invariant_matrices(candidate_hx, candidate_hz, k=code.k) != visible_invariant:
            continue

        candidate = CSSCode(
            candidate_hx,
            candidate_hz,
            distance=code.distance,
            x_distance=code.x_distance,
            z_distance=code.z_distance,
        )
        if use_additive_invariant:
            other_invariant = _css_additive_collision_invariant_matrices(candidate_hx, candidate_hz)
        elif rx + rz > 20:
            other_invariant = _css_support_rank_invariant_matrices(candidate_hx, candidate_hz)
        else:
            other_invariant = _css_stabilizer_weight_enumerator_matrices(candidate_hx, candidate_hz)

        if other_invariant != invariant:
            return candidate

    raise RandomizeError("Could not find a same-cheap-invariant candidate with a different CSS invariant.")


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
        raise RandomizeError("No non-equivalent stabilizer code exists with these small invariants.")

    invariant = _projection_rank_invariant(code)[2]
    for _ in range(1_000):
        candidate = lc_equivalent_code(code, seed=int(rng.integers(0, np.iinfo(np.int32).max)))
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

def random_permuted_stabilizer_pair_and_log_ops(
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
    return code, _permute_stabilizer_code_and_log_ops(code, permutation)

def random_non_permuted_stabilizer_pair(    
    n: int,
    k: int,
    *,
    seed: int | None = None,
    clifford_steps: int | None = None,
    max_attempts: int = 10_000,
) -> tuple[StabilizerCode, StabilizerCode]:
    """Return a seeded random stabilizer code together with another random stabilizer code that is guaranteed to be non-permuted."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        other_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        code = random_stabilizer_code(n, k, seed=code_seed, clifford_steps=clifford_steps)
        try:
            return code, non_permutation_equivalent_stabilizer_code(
                code,
                seed=other_seed,
                clifford_steps=clifford_steps,
            )
        except RandomizeError:
            continue

    raise RandomizeError(f"Could not generate non-permuted stabilizer pair after {max_attempts} attempts.")

def random_non_permuted_css_pair(    
    n: int,
    k: int,
    *,
    seed: int | None = None,
    max_attempts: int = 10_000,
) -> tuple[CSSCode, CSSCode]:
    """Return a seeded random css code together with another random css code that is guaranteed to be non-permuted."""
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

    raise RandomizeError(f"Could not generate non-permuted CSS pair after {max_attempts} attempts.")

def random_permuted_css_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
) -> tuple[CSSCode, CSSCode]:
    """Return a seeded random css code together with a permuted equivalent copy."""
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
    """Return an LC-equivalent code."""
    rng = np.random.default_rng(seed)
    tableau = code.generators.copy()
    local_cliffords = [str(rng.choice(_LOCAL_CLIFFORDS)) for _ in range(tableau.n)]

    if tableau.n > 0 and all(local_clifford == "I" for local_clifford in local_cliffords):
        local_cliffords[int(rng.integers(0, tableau.n))] = str(rng.choice(_LOCAL_CLIFFORDS[1:]))

    for qubit, local_clifford in enumerate(local_cliffords):
        _apply_local_clifford(tableau, local_clifford, qubit)

    base_changed_tableau = _random_tableau_row_space_base_change(tableau, rng=rng, steps=row_steps)
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
        raise RandomizeError("No non-LC-equivalent stabilizer code exists for the trivial code.")
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
                _random_tableau_row_space_base_change(candidate.generators, rng=rng, steps=row_steps)
            )

    raise RandomizeError("Could not find a candidate with a different LC support-rank invariant.")


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
        blocks.append(random_stabilizer_code(remaining_n, remaining_k, seed=remainder_seed))
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
    """Return an LC-equivalent code, where the logical operators are transferred with the same local cliffords."""
    rng = np.random.default_rng(seed)
    tableau = code.generators.copy()
    x_logicals = code.x_logicals.copy()
    z_logicals = code.z_logicals.copy()
    local_cliffords = [str(rng.choice(_LOCAL_CLIFFORDS)) for _ in range(tableau.n)]

    if tableau.n > 0 and all(local_clifford == "I" for local_clifford in local_cliffords):
        local_cliffords[int(rng.integers(0, tableau.n))] = str(rng.choice(_LOCAL_CLIFFORDS[1:]))

    for qubit, local_clifford in enumerate(local_cliffords):
        _apply_local_clifford(tableau, local_clifford, qubit)
        _apply_local_clifford(x_logicals, local_clifford, qubit)
        _apply_local_clifford(z_logicals, local_clifford, qubit)

    base_changed_tableau = _random_tableau_row_space_base_change(tableau, rng=rng, steps=row_steps)
    return StabilizerCode(generators=base_changed_tableau, distance=code.distance, x_logicals=x_logicals, z_logicals=z_logicals)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------

def random_density_stabilizer_code(
        n: int,
        k: int,
        ones_fraction: float = 0.1,
        seed: int | None = None,
) -> StabilizerCode:
    """Return a seeded random ``[[n, k]]`` code with controlled tableau density.

    Rows are sampled directly as random symplectic vectors and accepted only
    when they commute with the rows already chosen and increase the row rank.
    The total number of ones is targeted from ``ones_fraction`` up to rounding,
    so the resulting generators mix X/Z/Y positions without using a special
    structured construction.
    """
    if n < 1:
        msg = "n must be at least 1."
        raise ValueError(msg)
    if not 0 <= k <= n:
        msg = f"k must satisfy 0 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)
    if not 0 <= ones_fraction <= 1:
        msg = f"ones_fraction must satisfy 0 <= ones_fraction <= 1, got {ones_fraction}."
        raise ValueError(msg)

    num_stabilizers = n - k
    if num_stabilizers == 0:
        return StabilizerCode(StabilizerTableau.empty(n))

    num_entries = num_stabilizers * 2 * n
    min_ones = num_stabilizers
    target_ones = max(min_ones, int(np.ceil(ones_fraction * num_entries)))

    rng = np.random.default_rng(seed)
    tableau = _random_commuting_full_rank_tableau_with_weight(
        num_stabilizers=num_stabilizers,
        n=n,
        target_ones=target_ones,
        rng=rng,
    )
    return StabilizerCode(StabilizerTableau(tableau))

def random_symmetry_stabilizer_code(
        n: int,
        k: int,
        symmetry_fraction: float = 0.1,
        seed: int | None = None,
) -> StabilizerCode:
    """Return a seeded random ``[[n, k]]`` code with controlled tableau symmetry.

    At least ``ceil(symmetry_fraction * n)`` physical qubits are made symmetric
    by giving them identical X and Z columns in the stabilizer tableau. Rows are
    otherwise sampled randomly under this column-equality constraint and are
    accepted only when they commute and increase the stabilizer rank.
    """
    if n < 1:
        msg = "n must be at least 1."
        raise ValueError(msg)
    if not 0 <= k <= n:
        msg = f"k must satisfy 0 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)
    if not 0 <= symmetry_fraction <= 1:
        msg = f"symmetry_fraction must satisfy 0 <= symmetry_fraction <= 1, got {symmetry_fraction}."
        raise ValueError(msg)

    num_stabilizers = n - k
    if num_stabilizers == 0:
        return StabilizerCode(StabilizerTableau.empty(n))

    num_symmetric_qubits = int(np.ceil(symmetry_fraction * n))
    if num_symmetric_qubits <= 1:
        return random_stabilizer_code(n, k, seed=seed)

    max_rank = _max_rank_with_same_column_block(n, num_symmetric_qubits)
    if num_stabilizers > max_rank:
        msg = (
            f"Cannot construct a [[{n}, {k}]] code with {num_symmetric_qubits} "
            f"same-column symmetric qubits; maximum stabilizer rank is {max_rank}."
        )
        raise RandomizeError(msg)

    rng = np.random.default_rng(seed)
    symmetric_qubits = tuple(int(q) for q in rng.choice(n, size=num_symmetric_qubits, replace=False))
    tableau = _random_symmetric_full_rank_tableau(
        num_stabilizers=num_stabilizers,
        n=n,
        symmetric_qubits=symmetric_qubits,
        rng=rng,
    )
    return StabilizerCode(StabilizerTableau(tableau))

# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------


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

def _random_commuting_full_rank_tableau_with_weight(
    *,
    num_stabilizers: int,
    n: int,
    target_ones: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a random full-rank commuting tableau with exactly ``target_ones`` ones."""
    row_width = 2 * n
    for _ in range(200):
        rows = np.zeros((0, row_width), dtype=np.int8)
        remaining_ones = target_ones

        for row_idx in range(num_stabilizers):
            remaining_rows = num_stabilizers - row_idx
            min_weight = max(1, remaining_ones - (remaining_rows - 1) * row_width)
            max_weight = min(row_width, remaining_ones - (remaining_rows - 1))
            if min_weight > max_weight:
                break

            row = _sample_commuting_independent_row(
                rows,
                n=n,
                min_weight=min_weight,
                max_weight=max_weight,
                target_fraction=target_ones / (num_stabilizers * row_width),
                rng=rng,
            )
            if row is None:
                break

            rows = np.vstack((rows, row))
            remaining_ones -= int(np.count_nonzero(row))
        else:
            if remaining_ones == 0 and _rank_binary(rows) == num_stabilizers:
                return rows

    msg = (
        f"Could not construct a random commuting full-rank tableau with "
        f"{target_ones} ones."
    )
    raise RandomizeError(msg)

def _sample_commuting_independent_row(
    rows: np.ndarray,
    *,
    n: int,
    min_weight: int,
    max_weight: int,
    target_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Sample one row that commutes with ``rows`` and increases their rank."""
    row_width = 2 * n
    for _ in range(10_000):
        weight = int(rng.binomial(row_width, target_fraction))
        weight = min(max(weight, min_weight), max_weight)
        candidate = np.zeros(row_width, dtype=np.int8)
        candidate[rng.choice(row_width, size=weight, replace=False)] = 1

        if rows.shape[0] == 0:
            return candidate
        commutations = (rows[:, :n] @ candidate[n:] + rows[:, n:] @ candidate[:n]) % 2
        if np.any(commutations):
            continue
        if _rank_binary(np.vstack((rows, candidate))) == rows.shape[0] + 1:
            return candidate

    return None

def _max_rank_with_same_column_block(n: int, block_size: int) -> int:
    """Return the largest commuting row rank possible with one identical-column block."""
    if block_size <= 1:
        return n
    if block_size % 2 == 1:
        return n - block_size + 1
    return n - block_size + 2

def _random_symmetric_full_rank_tableau(
    *,
    num_stabilizers: int,
    n: int,
    symmetric_qubits: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a random full-rank commuting tableau with identical columns on a qubit block."""
    symmetric_qubits = tuple(int(q) for q in symmetric_qubits)
    symmetric_set = set(symmetric_qubits)
    ordinary_qubits = tuple(q for q in range(n) if q not in symmetric_set)
    effective_n = len(ordinary_qubits) + 1

    for _ in range(200):
        rows = np.zeros((0, 2 * n), dtype=np.int8)

        for _row_idx in range(num_stabilizers):
            row = _sample_symmetric_commuting_independent_row(
                rows,
                n=n,
                effective_n=effective_n,
                symmetric_qubits=symmetric_qubits,
                ordinary_qubits=ordinary_qubits,
                rng=rng,
            )
            if row is None:
                break

            rows = np.vstack((rows, row))
        else:
            if _rank_binary(rows) == num_stabilizers:
                return rows

    msg = "Could not construct a random commuting full-rank tableau with the requested symmetry."
    raise RandomizeError(msg)

def _sample_symmetric_commuting_independent_row(
    rows: np.ndarray,
    *,
    n: int,
    effective_n: int,
    symmetric_qubits: Sequence[int],
    ordinary_qubits: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Sample one constrained row that commutes with ``rows`` and increases rank."""
    for _ in range(10_000):
        effective_row = rng.integers(0, 2, size=2 * effective_n, dtype=np.int8)
        if not np.any(effective_row):
            continue

        candidate = _expand_symmetric_effective_row(
            effective_row,
            n=n,
            symmetric_qubits=symmetric_qubits,
            ordinary_qubits=ordinary_qubits,
        )
        commutations = (rows[:, :n] @ candidate[n:] + rows[:, n:] @ candidate[:n]) % 2
        if np.any(commutations):
            continue
        if _rank_binary(np.vstack((rows, candidate))) == rows.shape[0] + 1:
            return candidate

    return None

def _expand_symmetric_effective_row(
    effective_row: np.ndarray,
    *,
    n: int,
    symmetric_qubits: Sequence[int],
    ordinary_qubits: Sequence[int],
) -> np.ndarray:
    """Expand one effective row, copying one column pair onto the symmetric block."""
    effective_n = len(ordinary_qubits) + 1
    row = np.zeros(2 * n, dtype=np.int8)

    block_x = effective_row[0]
    block_z = effective_row[effective_n]
    for qubit in symmetric_qubits:
        row[qubit] = block_x
        row[qubit + n] = block_z

    for effective_idx, qubit in enumerate(ordinary_qubits, start=1):
        row[qubit] = effective_row[effective_idx]
        row[qubit + n] = effective_row[effective_idx + effective_n]

    return row

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

def _random_anchor_base_code(
    n: int,
    k: int,
    *,
    rng: np.random.Generator,
    clifford_steps: int | None = None,
) -> StabilizerCode | None:
    """Return the random shared part for an anchored ``[[n, k]]`` code."""
    r = n - k
    if r < 1:
        msg = "Anchored construction requires at least one stabilizer."
        raise ValueError(msg)
    if r == 1:
        return None

    base_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    return random_stabilizer_code(n - 1, k, seed=base_seed, clifford_steps=clifford_steps)

def _random_anchored_stabilizer_code(
    n: int,
    k: int,
    anchor: str,
    *,
    base_code: StabilizerCode | None,
    rng: np.random.Generator,
) -> StabilizerCode:
    """Return a randomized direct sum of ``base_code`` with a one-qubit anchor."""
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
) -> StabilizerTableau:
    """Append an X/Z/Y one-qubit stabilizer to a base ``[[n-1, k]]`` code."""
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
            msg = f"Expected a base [[{base_n}, {k}]] code, got [[{base_code.n}, {base_code.k}]]."
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
        msg = "A base code is required when the anchored construction has more than one stabilizer."
        raise ValueError(msg)

    anchor_row = r - 1
    anchor_qubit = n - 1
    if anchor in {"X", "Y"}:
        matrix[anchor_row, anchor_qubit] = 1
    if anchor in {"Z", "Y"}:
        matrix[anchor_row, anchor_qubit + n] = 1

    return StabilizerTableau(matrix)

def _support_rank_invariant(code: StabilizerCode, max_w: int = 3):
    M = np.asarray(code.symplectic, dtype=np.uint8) & 1
    n = code.n
    r = _rank_binary(M)

    profile = []

    for w in range(1, max_w + 1):
        projection_ranks : Counter[int] = Counter()
        normalizer_support_dims : Counter[int] = Counter()
        stabilizer_support_dims : Counter[int] = Counter()

        for qubits in combinations(range(n), w):
            qubits_set : set[int] = set(qubits)

            cols = [c for q in qubits_set for c in (q, q + n)]
            proj_rank = _rank_binary(M[:, cols])

            projection_ranks[proj_rank] += 1

            # Dimension of normalizer elements supported inside this subset.
            normalizer_support_dims[2 * w - proj_rank] += 1

            complement_cols = [
                c
                for q in range(n)
                if q not in qubits_set
                for c in (q, q + n)
            ]

            # Dimension of stabilizers supported inside this subset.
            stabilizer_support_dims[r - _rank_binary(M[:, complement_cols])] += 1

        profile.append((
            tuple(sorted(projection_ranks.items())),
            tuple(sorted(normalizer_support_dims.items())),
            tuple(sorted(stabilizer_support_dims.items())),
        ))

    return tuple(profile)

def _visible_css_invariant(code: CSSCode) -> tuple[int, int, int, int, int, tuple[int, ...]]:
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

def _css_support_rank_invariant(code: CSSCode, max_w: int = 3) -> tuple[Any, ...]:
    """Return a polynomial CSS invariant under physical-qubit permutations."""
    return _css_support_rank_invariant_matrices(code.Hx, code.Hz, max_w=max_w)


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

        profile.append((
            tuple(sorted(subset_ranks.items())),
            tuple(sorted(subset_support_dims.items())),
        ))

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

def _lc_projection_rank_invariant(code: StabilizerCode, max_w: int = 3) -> tuple[tuple[int, ...], ...]:
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

    def candidate_rank(candidates: list[int], basis: tuple[int, ...], needed: int) -> int:
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

def _cheap_invariant(code: StabilizerCode) -> tuple[int, int, int, int, Any]:
    M = np.asarray(code.symplectic, dtype=np.uint8) & 1
    n = code.n
    return (
        _rank_binary(M),
        _rank_binary(M[:, :n]),
        _rank_binary(M[:, n:]),
        _rank_binary(M[:, :n] ^ M[:, n:]),
        _support_rank_invariant(code, max_w=3)
    )

def _stabilizer_weight_enumerator(code: StabilizerCode) -> tuple[tuple[tuple[int, int, int, int], int], ...]:
    enumerator: dict[tuple[int, int, int, int], int] = {}
    for word in _row_space_words(code.symplectic):
        x_word, z_word = word[: code.n], word[code.n :]
        both = int(np.count_nonzero(x_word & z_word))
        x_only = int(np.count_nonzero(x_word)) - both
        z_only = int(np.count_nonzero(z_word)) - both
        key = (code.n - x_only - z_only - both, x_only, z_only, both)
        enumerator[key] = enumerator.get(key, 0) + 1
    return tuple(sorted(enumerator.items()))


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

    changed = _random_row_space_base_change(tableau.tableau.matrix, rng=rng, steps=steps)
    return StabilizerTableau(changed)


def _without_phases(tableau: StabilizerTableau) -> StabilizerTableau:
    """Return a copy of ``tableau`` with zero phases."""
    return StabilizerTableau(tableau.tableau.matrix.copy())


def _permute_tableau(tableau: StabilizerTableau, permutation: Sequence[int]) -> StabilizerTableau:
    """Return a copy of ``tableau`` with physical qubits permuted."""
    permutation = _checked_permutation(tableau.n, permutation)
    columns = list(permutation) + [q + tableau.n for q in permutation]
    return StabilizerTableau(tableau.tableau.matrix[:, columns].copy())


def _permute_stabilizer_code(code: StabilizerCode, permutation: Sequence[int]) -> StabilizerCode:
    """Return a copy of ``code`` with physical qubits permuted in generators, but logical operators are recomputed."""
    return StabilizerCode(_permute_tableau(code.generators.copy(), permutation), distance=code.distance)

def _permute_stabilizer_code_and_log_ops(code: StabilizerCode, permutation: Sequence[int]) -> StabilizerCode:
    """Return a copy of ``code`` with physical qubits permuted in generators and logical operators."""
    return StabilizerCode(_permute_tableau(code.generators.copy(), permutation), x_logicals=_permute_tableau(code.x_logicals.copy(), permutation), z_logicals=_permute_tableau(code.z_logicals.copy(), permutation), distance=code.distance)


def _apply_local_clifford(tableau: StabilizerTableau, local_clifford: str, qubit: int) -> None:
    """Apply one single-qubit Clifford representative to ``qubit`` in-place."""
    if local_clifford not in _LOCAL_CLIFFORDS:
        msg = f"Unknown local Clifford {local_clifford!r}."
        raise ValueError(msg)

    for gate in reversed(local_clifford):
        if gate == "H":
            tableau.apply_h(qubit)
        elif gate == "S":
            tableau.apply_s(qubit)


def _apply_random_clifford_gate(tableau: StabilizerTableau, rng: np.random.Generator) -> None:
    """Apply one seeded random Clifford gate (not only generators, to make more random hopefully) to a tableau in-place."""
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
