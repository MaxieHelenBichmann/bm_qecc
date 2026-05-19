"""Focused checks for the matroid solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import RandomizeError, random_permuted_css_pair, random_non_permuted_css_pair
from src.algorithms.p_css_matroid import _circuits_binary_matroid, _graph_from_circuits, are_peq_css_matroid

# ----------------------------------------------------------------------------------------------------
# _circuits_binary_matroid
# ----------------------------------------------------------------------------------------------------

def test_circuits_binary_matroid_simple_dependency() -> None:
    matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )

    assert _circuits_binary_matroid(matrix) == [(0, 1, 2)]

# ----------------------------------------------------------------------------------------------------
# _graph_from_circuits
# ----------------------------------------------------------------------------------------------------

def test_graph_from_circuits_small_incidence_graph() -> None:
    graph = _graph_from_circuits(3, circuits_hx=[(0, 2)], circuits_hz=[(1, 2)])

    assert graph.number_of_vertices == 5
    assert graph.directed is False
    assert graph.vertex_coloring == [{0, 1, 2}, {3}, {4}]
    assert {v: set(neighbors) for v, neighbors in graph.adjacency_dict.items()} == {
        0: {3},
        1: {4},
        2: {3, 4},
        3: {0, 2},
        4: {1, 2},
    }

# ----------------------------------------------------------------------------------------------------
# are_peq_css_matroid
# ----------------------------------------------------------------------------------------------------

def test_are_peq_css_matroid_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n):
            try:
                code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_css_matroid(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_matroid_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_matroid(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_matroid_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_matroid(code1, code2) is False
