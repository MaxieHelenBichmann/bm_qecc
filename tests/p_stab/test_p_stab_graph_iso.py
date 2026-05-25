"""Focused checks for the graph-isomorphism solution to whether two Stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import pytest
from pynauty import Graph

from benchmarks.utils import RandomizeError, random_permuted_stabilizer_pair, random_non_permuted_stabilizer_pair
from src.core.stabilizer_code import StabilizerCode
from src.algorithms.p_stab.p_stab_graph_iso import (
    _graph_from_code,
    are_peq_stab_graph_iso,
)


def _adjacency_as_sets(graph: Graph) -> dict[int, set[int]]:
    return {
        int(vertex): {int(neighbor) for neighbor in neighbors}
        for vertex, neighbors in graph.adjacency_dict.items()
    }

# ----------------------------------------------------------------------------------------------------
# _graph_from_code
# ----------------------------------------------------------------------------------------------------

def test_graph_from_trivial_code() -> None:
    graph = _graph_from_code(StabilizerCode.get_trivial_code(3))

    assert graph.number_of_vertices == 4
    assert graph.directed is False
    assert graph.vertex_coloring == [{0, 1, 2}, {3}, set(), set()]
    assert _adjacency_as_sets(graph) == {}


def test_graph_from_code_splits_x_and_z_edges() -> None:
    graph = _graph_from_code(StabilizerCode(["XZ"]))

    assert graph.number_of_vertices == 6
    assert graph.directed is False
    assert graph.vertex_coloring == [{0, 1}, {2, 3}, {5}, {4}]
    assert _adjacency_as_sets(graph) == {
        0: {4},
        1: {5},
        3: {4, 5},
        4: {0, 3},
        5: {1, 3},
    }

# ----------------------------------------------------------------------------------------------------
# are_peq_stab_graph_iso
# ----------------------------------------------------------------------------------------------------

def test_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_stab_graph_iso(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_graph_iso(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_graph_iso(code1, code2) is False
