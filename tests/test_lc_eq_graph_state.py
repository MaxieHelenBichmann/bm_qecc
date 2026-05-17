"""Focused checks for the graph-state machinery to check whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_stabilizer_code, lc_equivalent_code
from src.algorithms.lc_eq_graph_state import _stab_code_to_stab_state, _stab_state_to_graph_state, _lc_equiv_graph_states, are_lceq_graph_state

# ----------------------------------------------------------------------------------------------------
# _stab_code_to_stab_state
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _stab_state_to_graph_state
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _lc_equiv_graph_states
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# are_lceq_graph_state
# ----------------------------------------------------------------------------------------------------

def test_are_lceq_graph_state_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq_graph_state(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_lceq_graph_state_random_positive(seed: int) -> None:
    n = 1 + seed % 4
    k = seed % (n + 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)
    assert are_lceq_graph_state(code1, code2) is True