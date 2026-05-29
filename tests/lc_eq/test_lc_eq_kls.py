"""Focused checks for the graph-state machinery to check whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import random_stabilizer_code, lc_equivalent_code
from src.algorithms.lc_eq.lc_eq_kls import (
    are_lceq_kls,
)
from src.core.stabilizer_code import StabilizerCode

# ----------------------------------------------------------------------------------------------------
# are_lceq_kls
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in [3, 28, 35]])
def test_are_lceq_kls_failing_choi(seed: int) -> None:
    """
    3:  < IZI > | < IXI > 
    28: < IXI > | < IYI >
    35: < XZZ > | < ZXX >
    """
    code1 = random_stabilizer_code(3, 2, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)
    assert are_lceq_kls(code1, code2) is True

def test_are_lceq_kls_random_smoke() -> None:
    for n in range(3, 9):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq_kls(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_lceq_kls_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)
    assert are_lceq_kls(code1, code2) is True