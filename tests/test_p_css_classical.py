"""Focused checks for the classical solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_permuted_css_pair, random_non_permuted_css_pair
from src.algorithms.p_css_classical import _compute_signatures, _compute_canonical_form, _extract_permutations, are_peq_css_classical

# ----------------------------------------------------------------------------------------------------
# _compute_signatures
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _compute_canonical_form
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _extract_permutations
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# are_peq_css_classical
# ----------------------------------------------------------------------------------------------------

def test_are_peq_css_classical_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
            assert isinstance(are_peq_css_classical(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_classical_random_positive(seed: int) -> None:
    n = 1 + seed % 4
    k = seed % (n + 1)

    code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_css_classical(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_classical_random_negative(seed: int) -> None:
    n = 1 + seed % 4
    k = seed % (n + 1)

    code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_css_classical(code1, code2) is False