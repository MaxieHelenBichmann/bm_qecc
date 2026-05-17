"""Focused checks for the classical solution to whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_permuted_stabilizer_pair, random_non_permuted_stabilizer_pair
from src.algorithms.p_stab_classical import are_peq_stab_classical

# ----------------------------------------------------------------------------------------------------
# GF4
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _symplectic_to_gf4
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _gf4_rref
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _compute_signatures
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _compute_canonical_form
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# are_peq_stab_classical
# ----------------------------------------------------------------------------------------------------

def test_are_peq_stab_classical_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
            assert isinstance(are_peq_stab_classical(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_classical_random_positive(seed: int) -> None:
    n = 1 + seed % 4
    k = seed % (n + 1)

    code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_stab_classical(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_classical_random_negative(seed: int) -> None:
    n = 1 + seed % 4
    k = seed % (n + 1)

    code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_stab_classical(code1, code2) is False