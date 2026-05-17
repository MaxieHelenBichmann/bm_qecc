"""Focused checks for the brute-force solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

from numpy.random import seed
import pytest

from benchmarks.utils import random_permuted_css_pair, random_non_permuted_css_pair, RandomizeError
from src.algorithms.p_css_bruteforce import are_peq_css_bruteforce

# ----------------------------------------------------------------------------------------------------
# are_peq_css_bruteforce
# ----------------------------------------------------------------------------------------------------

def test_are_peq_css_bruteforce_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(1, n):
            try:
                code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_css_bruteforce(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_bruteforce_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_bruteforce(code1, code2) is True

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_bruteforce_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_bruteforce(code1, code2) is False