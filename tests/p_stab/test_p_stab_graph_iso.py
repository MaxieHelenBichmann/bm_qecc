"""Focused checks for the graph-isomorphism solution to whether two Stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import pytest
from pynauty import Graph, certificate

from benchmarks.utils import RandomizeError, random_permuted_stabilizer_pair, random_non_permuted_stabilizer_pair
from src.core.stabilizer_code import StabilizerCode
from src.algorithms.p_stb.p_stab_graph_iso import (
    _graph_from_code,
    are_peq_stab_graph_iso,
)
from src.algorithms.p_stb.p_stab_bruteforce import are_peq_stab_bruteforce


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

    assert graph.number_of_vertices == 6
    assert graph.directed is False
    assert graph.vertex_coloring == [{0, 1, 2}, {3}, {4}, {5}]
    assert _adjacency_as_sets(graph) == {}


def test_graph_from_code_splits_x_and_z_edges() -> None:
    graph = _graph_from_code(StabilizerCode(["XZ"]))

    # Vertex 4 is the x-edge, 5 the z-edge; 6 and 7 are the class anchors.
    assert graph.number_of_vertices == 8
    assert graph.directed is False
    assert graph.vertex_coloring == [{0, 1}, {2, 3}, {5, 6}, {4, 7}]
    assert _adjacency_as_sets(graph) == {
        0: {4},
        1: {5},
        3: {4, 5},
        4: {0, 3},
        5: {1, 3},
    }


def test_anchors_are_isolated() -> None:
    """The anchors hold color positions only; they must not add adjacency."""
    graph = _graph_from_code(StabilizerCode(["XZ"]))
    adjacency = _adjacency_as_sets(graph)
    anchors = {6, 7}

    assert not anchors & adjacency.keys()
    assert not any(anchors & neighbors for neighbors in adjacency.values())

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


def test_pynauty_drops_empty_colour_classes() -> None:
    """The library behaviour the anchors exist to work around."""
    adjacency = {0: [2], 2: [0], 1: [2]}

    def build(colouring: list[set[int]], vertices: int = 3) -> Graph:
        return Graph(
            number_of_vertices=vertices,
            directed=False,
            adjacency_dict=adjacency,
            vertex_coloring=colouring,
        )
    
    assert certificate(build([{0}, {1}, {2}])) != certificate(build([{2}, {1}, {0}]))
    assert certificate(build([{0}, {1}, set(), {2}])) == certificate(
        build([{0}, {1}, {2}, set()])
    )


@pytest.mark.parametrize(
    "first,second",
    [
        (["IXI"], ["IZI"]),
        (["XXI", "IXX"], ["ZZI", "IZZ"]),
        (["XXXX"], ["ZZZZ"]),
        (["XXX"], ["ZZZ"]),
    ],
)
def test_pure_x_and_pure_z_codes_are_not_isomorphic(
    first: list[str], second: list[str]
) -> None:
    """A permutation cannot turn X into Z, and the reduction must agree."""
    code1, code2 = StabilizerCode(first), StabilizerCode(second)
    assert are_peq_stab_bruteforce(code1, code2) is False
    assert are_peq_stab_graph_iso(code1, code2) is False


def test_anchors_do_not_change_isomorphism_of_equivalent_codes() -> None:
    """Adding the anchors must not make equivalent codes look different."""
    for n, k in ((4, 1), (5, 2), (6, 3)):
        for seed in range(6):
            code1, code2 = random_permuted_stabilizer_pair(n, k, seed=seed)
            assert are_peq_stab_graph_iso(code1, code2) is True
