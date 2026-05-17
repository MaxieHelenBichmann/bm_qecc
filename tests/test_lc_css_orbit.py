"""Focused checks for the LC-orbit traversal to whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_stabilizer_code, random_css_code, lc_equivalent_code
from src.algorithms.lc_css_orbit import _stab_code_to_stab_state, _stab_state_to_graph_state, _traverse_lc_orbit, is_lceq_css_orbit

# ----------------------------------------------------------------------------------------------------
# _stab_code_to_stab_state
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _stab_code_to_stab_state
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _stab_state_to_graph_state
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _traverse_lc_orbit
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# is_lceq_css_orbit
# ----------------------------------------------------------------------------------------------------

def test_is_lceq_css_orbit_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_orbit(code), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_orbit_random_positive(seed: int) -> None:
    n = 1 + seed % 4
    k = seed % (n + 1)
    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)
    assert is_lceq_css_orbit(code) is True