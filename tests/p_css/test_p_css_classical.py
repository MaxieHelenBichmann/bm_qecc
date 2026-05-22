"""Focused checks for the classical solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import (
    RandomizeError,
    random_non_permuted_css_pair,
    random_permuted_css_pair,
)
from src.algorithms.p_css.p_css_classical import (
    _check_permutation_equivalence,
    _compute_canonical_form,
    _compute_signatures,
    _extract_permutations,
    are_peq_css_classical,
)
from src.core.css_code import CSSCode

# ----------------------------------------------------------------------------------------------------
# _compute_signatures
# ----------------------------------------------------------------------------------------------------

def test_compute_signatures_column_permutation() -> None:
    Gx = np.array(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    Gz = np.array(
        [
            [1, 1, 0, 0, 1],
            [0, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    permutation = (2, 4, 0, 3, 1)

    signatures = _compute_signatures(Gx, Gz)
    permuted_signatures = _compute_signatures(Gx[:, permutation], Gz[:, permutation])

    assert permuted_signatures == [signatures[i] for i in permutation]

# ----------------------------------------------------------------------------------------------------
# _compute_canonical_form
# ----------------------------------------------------------------------------------------------------

def test_compute_canonical_form_stable_row_operations() -> None:
    G = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    row_changed = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    canon, perms = _compute_canonical_form(G, [[0, 1, 2, 3]])
    row_changed_canon, row_changed_perms = _compute_canonical_form(
        row_changed, [[0, 1, 2, 3]]
    )

    assert np.array_equal(canon, row_changed_canon)
    assert perms
    assert row_changed_perms

# ----------------------------------------------------------------------------------------------------
# _extract_permutations
# ----------------------------------------------------------------------------------------------------

def test_extract_permutations_matching_convention() -> None:
    canon = np.array([[1, 0, 1, 0],
                      [0, 1, 1, 0]], dtype=np.uint8)
    can_to_g1 = [[1, 0, 3, 2]]
    can_to_g2 = [[2, 1, 0, 3]]
    expected_permutation = (3, 0, 1, 2)
    code1 = CSSCode(Hx=np.array([[0, 1, 0, 1],
                                 [1, 0, 0, 1]], dtype=np.int8), Hz=None)
    code2 = CSSCode(Hx=np.array([[1, 0, 1, 0],
                                 [1, 1, 0, 0]], dtype=np.int8), Hz=None)

    extracted = _extract_permutations(canon, canon, can_to_g1, can_to_g2)

    assert extracted == [expected_permutation]
    assert _check_permutation_equivalence(code1, code2, extracted[0]) is True

# ----------------------------------------------------------------------------------------------------
# are_peq_css_classical
# ----------------------------------------------------------------------------------------------------

def test_are_peq_css_classical_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_css_classical(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_classical_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_classical(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_classical_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_classical(code1, code2) is False
