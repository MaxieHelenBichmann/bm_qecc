"""Focused checks for the SAT solution to whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_stabilizer_code, random_css_code, lc_equivalent_code
from src.algorithms.lc_css_sat import is_lceq_css_sat

# ----------------------------------------------------------------------------------------------------
# is_lceq_css_sat
# ----------------------------------------------------------------------------------------------------

def test_is_lceq_css_sat_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_sat(code), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_sat_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)
    assert is_lceq_css_sat(code) is True
