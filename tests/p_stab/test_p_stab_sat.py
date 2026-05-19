"""Focused checks for the SAT solution to whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_permuted_stabilizer_pair, random_non_permuted_stabilizer_pair
from src.algorithms.p_stab_sat import are_peq_stab_sat

# ----------------------------------------------------------------------------------------------------
# are_peq_stab_sat
# ----------------------------------------------------------------------------------------------------

def test_are_peq_stab_sat_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
            assert isinstance(are_peq_stab_sat(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_sat_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_stab_sat(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_sat_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_stab_sat(code1, code2) is False
