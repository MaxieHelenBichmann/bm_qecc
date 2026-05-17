"""Focused checks for the graph-state machinery to check whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_permuted_stabilizer_pair, random_non_permuted_stabilizer_pair
from src.algorithms.p_stab_graph_state import _stab_code_to_stab_state, _stab_state_to_graph_state, _traverse_lc_orbit, _extract_qubit_permutations, are_peq_stab_graph_state

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
# _extract_qubit_permutations
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# are_peq_stab_graph_state
# ----------------------------------------------------------------------------------------------------

def test_are_peq_stab_graph_state_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
            assert isinstance(are_peq_stab_graph_state(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_graph_state_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_stab_graph_state(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_graph_state_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    assert are_peq_stab_graph_state(code1, code2) is False