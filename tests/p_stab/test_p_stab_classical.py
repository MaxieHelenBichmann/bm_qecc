"""Focused checks for the classical solution to whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import (
    RandomizeError,
    random_non_permuted_stabilizer_pair,
    random_permuted_stabilizer_pair,
)
from src.algorithms.p_stab.p_stab_classical import (
    GF4,
    ONE,
    W,
    W_BAR,
    ZERO,
    _compute_canonical_form,
    _compute_signatures,
    _gf4_rref,
    _gf4_trace_inner_product,
    _symplectic_to_gf4,
    are_peq_stab_classical,
)
from src.core.stabilizer_code import StabilizerCode


def _gf4_values(matrix: np.ndarray) -> np.ndarray:
    return np.array([[entry.value for entry in row] for row in matrix], dtype=np.uint8)

# ----------------------------------------------------------------------------------------------------
# GF4
# ----------------------------------------------------------------------------------------------------

def test_gf4_arithmetic() -> None:
    elements = [ZERO, ONE, W, W_BAR]

    addition = np.array(
        [[(a + b).value for b in elements] for a in elements],
        dtype=np.uint8,
    )
    multiplication = np.array(
        [[(a * b).value for b in elements] for a in elements],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(
        addition,
        np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(
        multiplication,
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
            ],
            dtype=np.uint8,
        ),
    )
    assert W.conjugate() == W_BAR
    assert W.inverse() == W_BAR
    with pytest.raises(ValueError):
        GF4(4)

# ----------------------------------------------------------------------------------------------------
# _symplectic_to_gf4
# ----------------------------------------------------------------------------------------------------

def test_symplectic_to_gf4() -> None:
    tableau = np.array(
        [
            [0, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    gf4_matrix = _symplectic_to_gf4(tableau)

    np.testing.assert_array_equal(
        _gf4_values(gf4_matrix),
        np.array(
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.uint8,
        ),
    )

# ----------------------------------------------------------------------------------------------------
# _gf4_rref
# ----------------------------------------------------------------------------------------------------

def test_gf4_rref() -> None:
    matrix = np.array(
        [
            [ONE, ZERO],
            [ZERO, ONE],
            [W, ZERO],
        ],
        dtype=object,
    )

    rank, rref, pivot_columns = _gf4_rref(matrix)

    assert rank == 3
    assert pivot_columns == [0, 1, 0]
    np.testing.assert_array_equal(
        _gf4_values(rref),
        np.array(
            [
                [1, 0],
                [0, 1],
                [2, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_gf4_trace_inner_product_matches_symplectic_commutation() -> None:
    commuting1 = np.array([ONE, W, W_BAR, ZERO], dtype=object)
    commuting2 = np.array([ONE, W, W_BAR, ZERO], dtype=object)
    anticommuting = np.array([W, W, W_BAR, ZERO], dtype=object)

    assert _gf4_trace_inner_product(commuting1, commuting2) == ZERO
    assert _gf4_trace_inner_product(commuting1, anticommuting) == ONE

# ----------------------------------------------------------------------------------------------------
# _compute_signatures
# ----------------------------------------------------------------------------------------------------

def test_compute_signatures() -> None:
    generator_matrix = np.array(
        [
            [ONE, ZERO, W, W_BAR, ZERO],
            [ZERO, W, W_BAR, ONE, W],
            [ONE, ONE, ZERO, ZERO, W_BAR],
        ],
        dtype=object,
    )
    permutation = (2, 4, 0, 3, 1)

    signatures = _compute_signatures(generator_matrix)
    permuted_signatures = _compute_signatures(generator_matrix[:, permutation])

    assert permuted_signatures == [signatures[i] for i in permutation]

# ----------------------------------------------------------------------------------------------------
# _compute_canonical_form
# ----------------------------------------------------------------------------------------------------

def test_compute_canonical_form_stable_under_gf2_row_operations() -> None:
    generator_matrix = np.array(
        [
            [ONE, ZERO, W, W_BAR],
            [ZERO, ONE, W_BAR, W],
            [W, W_BAR, ZERO, ONE],
        ],
        dtype=object,
    )
    row_changed = np.array(
        [
            [generator_matrix[0, col] + generator_matrix[1, col] for col in range(4)],
            [generator_matrix[1, col] for col in range(4)],
            [generator_matrix[2, col] for col in range(4)],
        ],
        dtype=object,
    )

    canonical = _compute_canonical_form(generator_matrix, [[0, 1, 2, 3]])
    row_changed_canonical = _compute_canonical_form(row_changed, [[0, 1, 2, 3]])

    assert np.array_equal(canonical, row_changed_canonical)


def test_compute_canonical_form_manual() -> None:
    code = StabilizerCode(["XZ", "IZ"])
    permuted_code = StabilizerCode(["ZX", "ZI"])
    generator_matrix = _symplectic_to_gf4(code.symplectic)
    permuted_generator_matrix = _symplectic_to_gf4(permuted_code.symplectic)

    canonical = _compute_canonical_form(generator_matrix, [[0, 1]])
    permuted_canonical = _compute_canonical_form(
        permuted_generator_matrix,
        [[0, 1]],
    )

    assert np.array_equal(canonical, permuted_canonical)

# ----------------------------------------------------------------------------------------------------
# are_peq_stab_classical
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in [1193, 1074]])
def test_are_peq_stab_classical_failing_bm(seed: int) -> None:
    code1, code2 = random_non_permuted_stabilizer_pair(13, 3, seed=seed)
    assert are_peq_stab_classical(code1, code2) is False

def test_are_peq_stab_classical_failing_bm2() -> None:
    code1, code2 = random_permuted_stabilizer_pair(n=8, k=3, seed=111)
    assert are_peq_stab_classical(code1, code2) is True

def test_are_peq_stab_classical_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_stab_classical(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_classical_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)
    try:
        code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_classical(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_classical_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_classical(code1, code2) is False
