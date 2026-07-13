"""Focused checks for the hybrid solution to whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import pytest

from benchmarks.utils import (
    RandomizeError,
    permutation_equivalent_code,
    random_non_permuted_stabilizer_pair,
    random_permuted_stabilizer_pair,
    random_stabilizer_code,
)
from src.core.stabilizer_code import StabilizerCode
from src.hybrids.p_stab import are_peq_stab


def _apply_permutation(symplectic: np.ndarray, permutation: list[int]) -> np.ndarray:
    """Apply p with p[source] = target to a symplectic tableau."""
    n = symplectic.shape[1] // 2
    assert sorted(permutation) == list(range(n))

    transformed = np.empty_like(symplectic)
    for source, target in enumerate(permutation):
        transformed[:, target] = symplectic[:, source]
        transformed[:, target + n] = symplectic[:, source + n]
    return transformed


def _assert_maps_rowspace(
    code1: StabilizerCode,
    code2: StabilizerCode,
    permutation: list[int],
) -> None:
    transformed = _apply_permutation(code1.symplectic, permutation)
    rank = 0 if code1.symplectic.shape[0] == 0 else mod2.rank(code1.symplectic)
    code2_rank = 0 if code2.symplectic.shape[0] == 0 else mod2.rank(code2.symplectic)
    combined = np.vstack([transformed, code2.symplectic])
    combined_rank = 0 if combined.shape[0] == 0 else mod2.rank(combined)

    assert code2_rank == rank
    assert combined_rank == rank

# ----------------------------------------------------------------------------------------------------
# are_peq_stab
# ----------------------------------------------------------------------------------------------------

def test_are_peq_stab_preserves_n() -> None:
    assert are_peq_stab(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_are_peq_stab_preserves_k() -> None:
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_peq_stab(code1, code2) is None


def test_are_peq_stab_does_not_swap_x_and_z() -> None:
    code1 = StabilizerCode(["XII"])
    code2 = StabilizerCode(["ZII"])

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_peq_stab(code1, code2) is None


def test_are_peq_stab_hardcoded_positive() -> None:
    code1 = StabilizerCode(["XXII", "IIZZ"])
    code2 = StabilizerCode(["XIXI", "IZIZ"])

    permutation = are_peq_stab(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace(code1, code2, permutation)


def test_are_peq_stab_witness_handles_row_basis_change() -> None:
    code1 = random_stabilizer_code(5, 2, seed=1234)
    code2 = permutation_equivalent_code(code1, seed=5678)

    permutation = are_peq_stab(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace(code1, code2, permutation)


def test_are_peq_stab_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
                permutation = are_peq_stab(code1, code2)
                assert permutation is not None
                _assert_maps_rowspace(code1, code2, permutation)
            except RandomizeError:
                pass


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    permutation = are_peq_stab(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace(code1, code2, permutation)


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab(code1, code2) is None
