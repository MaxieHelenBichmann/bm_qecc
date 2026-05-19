"""Focused checks for the KLS/HK normal-form conversion."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import random_stabilizer_code, random_css_code, lc_equivalent_code
from src.algorithms.lc_css_kls import (
    ZXGraph,
    _code_to_graph,
    _hk_normal_form,
    _kls_normal_form,
    is_lceq_css_kls,
)
from src.core.stabilizer_code import StabilizerCode

# ----------------------------------------------------------------------------------------------------
# _code_to_graph
# ----------------------------------------------------------------------------------------------------


def _assert_semi_bipartite_encoder_graph(graph: ZXGraph) -> None:
    assert len(graph.vertices) == graph.k + graph.n
    assert graph.get_input_output_adjacency().shape == (graph.k, graph.n)

    for u, v in graph.edges:
        assert 0 <= u < v < graph.k + graph.n
        assert not (u < graph.k and v < graph.k)


@pytest.mark.parametrize(
    ("code", "expected_n", "expected_k", "expected_vertices", "expected_edges"),
    [
        pytest.param(
            StabilizerCode(["Z"]),
            1,
            0,
            [(["H"], False)],
            set(),
            id="one-qubit-z-state",
        ),
        pytest.param(
            StabilizerCode(["X"]),
            1,
            0,
            [([], False)],
            set(),
            id="one-qubit-x-state",
        ),
        pytest.param(
            StabilizerCode(["Y"]),
            1,
            0,
            [(["S"], True)],
            set(),
            id="one-qubit-y-state",
        ),
        pytest.param(
            StabilizerCode.get_trivial_code(1),
            1,
            1,
            [([], False), ([], False)],
            set(),
            id="one-qubit-trivial-code",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            2,
            1,
            [([], False), (["H"], False), ([], False)],
            {(1, 2)},
            id="two-qubit-repetition-code",
        ),
    ],
)
def test_code_to_graph_small_codes(
    code: StabilizerCode,
    expected_n: int,
    expected_k: int,
    expected_vertices: list[tuple[list[str], bool]],
    expected_edges: set[tuple[int, int]],
) -> None:
    graph = _code_to_graph(code)

    assert graph.n == expected_n
    assert graph.k == expected_k
    _assert_graph(graph, expected_vertices, expected_edges)
    _assert_semi_bipartite_encoder_graph(graph)


def test_code_to_graph_rejects_identity_stabilizer_row() -> None:
    with pytest.raises(ValueError, match="identity stabilizer row"):
        _code_to_graph(StabilizerCode(["I"]))



# ----------------------------------------------------------------------------------------------------
# _hk_normal_form
# ----------------------------------------------------------------------------------------------------


def _hk(vertices: list[tuple[list[str], bool]], edges: set[tuple[int, int]]) -> ZXGraph:
    graph = ZXGraph()
    graph.n = len(vertices)
    graph.k = 0
    graph.vertices = [(list(deco), z) for deco, z in vertices]
    graph.edges = set(edges)
    return _hk_normal_form(graph)


def _assert_graph(
    graph: ZXGraph,
    vertices: list[tuple[list[str], bool]],
    edges: set[tuple[int, int]],
) -> None:
    assert graph.vertices == vertices
    assert graph.edges == edges


def _assert_hk_requirements(graph: ZXGraph) -> None:
    for i, (deco, _) in enumerate(graph.vertices):
        assert deco in ([], ["S"], ["H"])
        if deco == ["H"]:
            assert all(i < neighbor for neighbor in graph.neighbors(i))


@pytest.mark.parametrize(
    ("vertices", "expected_vertices"),
    [
        pytest.param([(["H", "H"], False)], [([], False)], id="hh-cancels"),
        pytest.param([(["S", "S"], False)], [([], True)], id="ss-becomes-z-phase"),
        pytest.param([(["Z", "Z"], True)], [([], True)], id="zz-cancels-before-z-bit"),
        pytest.param([(["Z", "S"], False)], [(["S"], True)], id="push-z-through-s"),
        pytest.param([(["H", "Z", "S"], True)], [(["S"], True)], id="internal-z-expands-to-s-square"),
    ],
)
def test_hk_normal_form_reduces_local_clifford_words(
    vertices: list[tuple[list[str], bool]],
    expected_vertices: list[tuple[list[str], bool]],
) -> None:
    graph = _hk(vertices, set())

    _assert_graph(graph, expected_vertices, set())
    _assert_hk_requirements(graph)


@pytest.mark.parametrize(
    ("vertices", "edges", "expected_vertices", "expected_edges"),
    [
        pytest.param(
            [(["H"], False), (["S"], False), ([], False)],
            {(0, 1), (1, 2)},
            [(["H"], False), (["S"], False), ([], False)],
            {(0, 1), (1, 2)},
            id="already-hk-normal-form",
        ),
        pytest.param(
            [(["H", "S"], False), ([], False), ([], False)],
            {(0, 1), (0, 2)},
            [(["S"], True), (["S"], True), (["S"], True)],
            {(0, 1), (0, 2), (1, 2)},
            id="reduce-trailing-hs",
        ),
        pytest.param(
            [(["S", "H"], False), ([], False), ([], False)],
            {(0, 1), (0, 2)},
            [(["H"], False), (["S"], False), (["S"], False)],
            {(0, 1), (0, 2), (1, 2)},
            id="reduce-solo-trailing-sh",
        ),
        pytest.param(
            [(["S", "H"], False), (["H"], False), ([], False)],
            {(0, 1), (0, 2)},
            [(["S"], False), ([], False), ([], False)],
            {(0, 1), (1, 2)},
            id="reduce-shared-trailing-sh",
        ),
        pytest.param(
            [(["S"], False), (["S"], False), (["H"], False), (["H"], False)],
            {(0, 2), (1, 3)},
            [(["H"], False), (["H"], False), ([], True), ([], True)],
            {(0, 2), (1, 3)},
            id="h-slide-and-sh-cleanup",
        ),
        pytest.param(
            [(["H"], False), (["H"], False)],
            {(0, 1)},
            [([], False), ([], False)],
            {(0, 1)},
            id="h-slide-onto-existing-h",
        ),
        pytest.param(
            [(["S"], False), (["H"], False)],
            {(0, 1)},
            [(["H"], False), ([], True)],
            {(0, 1)},
            id="h-slide-creates-and-cleans-sh-with-empty-neighbor",
        ),
        pytest.param(
            [(["S"], False), (["S"], False), (["H"], False)],
            {(0, 2), (1, 2)},
            [(["H"], False), ([], True), ([], True)],
            {(0, 1), (0, 2), (1, 2)},
            id="h-slide-creates-and-cleans-sh-with-s-neighbor",
        ),
        pytest.param(
            [([], False), ([], False), (["H"], False), ([], False)],
            {(0, 2), (1, 2), (2, 3)},
            [(["H"], False), ([], False), ([], False), ([], False)],
            {(0, 1), (0, 2), (0, 3)},
            id="h-slide-through-star",
        ),
    ],
)
def test_hk_normal_form_cases(
    vertices: list[tuple[list[str], bool]],
    edges: set[tuple[int, int]],
    expected_vertices: list[tuple[list[str], bool]],
    expected_edges: set[tuple[int, int]],
) -> None:
    graph = _hk(vertices, edges)
    _assert_graph(graph, expected_vertices, expected_edges)
    _assert_hk_requirements(graph)


def test_hk_normal_form_rejects_unsupported_residual_word() -> None:
    with pytest.raises(ValueError, match="Expected only I, S, or H decorations"):
        _hk([(["H", "S", "H"], False), ([], False)], {(0, 1)})

# ----------------------------------------------------------------------------------------------------
# _kls_normal_form
# ----------------------------------------------------------------------------------------------------


def _graph(
    n: int,
    k: int,
    vertices: list[tuple[list[str], bool]],
    edges: set[tuple[int, int]],
) -> ZXGraph:
    graph = ZXGraph()
    graph.n = n
    graph.k = k
    graph.vertices = [(list(deco), z) for deco, z in vertices]
    graph.edges = set(edges)
    return graph


def _assert_rref(matrix: np.ndarray) -> None:
    pivots : list[int] = []
    zero_row_seen = False

    for row_idx, row in enumerate(matrix.astype(np.uint8)):
        ones = np.flatnonzero(row)
        if len(ones) == 0:
            zero_row_seen = True
            continue

        assert not zero_row_seen

        pivot = int(ones[0])
        assert pivots == [] or pivot > pivots[-1]

        expected_col = np.zeros(matrix.shape[0], dtype=np.uint8)
        expected_col[row_idx] = 1
        np.testing.assert_array_equal(matrix[:, pivot].astype(np.uint8), expected_col)

        pivots.append(pivot)


def _assert_kls_requirements(graph: ZXGraph) -> None:
    _assert_semi_bipartite_encoder_graph(graph)

    for input_vertex in range(graph.k):
        assert graph.vertices[input_vertex] == ([], False)

    _assert_rref(graph.get_input_output_adjacency())


@pytest.mark.parametrize(
    ("graph_hk", "expected_io_adjacency"),
    [
        pytest.param(
            _graph(
                3,
                2,
                [(["H"], True), (["S"], True), (["S"], False), (["H"], False), ([], True)],
                {(0, 1), (0, 2), (1, 4)},
            ),
            np.array([[1, 0, 0], [0, 0, 1]], dtype=bool),
            id="strips-input-decorations-and-input-edge",
        ),
        pytest.param(
            _graph(
                3,
                2,
                [(["S"], True), (["H"], False), ([], False), (["S"], False), (["H"], True)],
                {(0, 1), (2, 3)},
            ),
            np.array([[0, 0, 0], [0, 0, 0]], dtype=bool),
            id="zero-input-output-adjacency",
        ),
        pytest.param(
            _graph(
                4,
                3,
                [([], False), (["S"], False), (["H"], True), ([], False), ([], False), (["S"], False), ([], True)],
                {(0, 3), (1, 5)},
            ),
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=bool),
            id="rank-deficient-already-rref",
        ),
        pytest.param(
            _graph(
                3,
                3,
                [([], False), (["S"], True), (["H"], False), ([], False), ([], False), ([], False)],
                {(0, 3), (1, 4), (2, 5), (3, 4)},
            ),
            np.eye(3, dtype=bool),
            id="full-rank-square-adjacency",
        ),
        pytest.param(
            _graph(
                3,
                0,
                [(["S"], False), (["H"], False), ([], True)],
                {(0, 1)},
            ),
            np.zeros((0, 3), dtype=bool),
            id="no-inputs",
        ),
    ],
)
def test_kls_normal_form_structural_cases(
    graph_hk: ZXGraph,
    expected_io_adjacency: np.ndarray,
) -> None:
    graph_kls = _kls_normal_form(graph_hk)

    _assert_kls_requirements(graph_kls)
    np.testing.assert_array_equal(graph_kls.get_input_output_adjacency(), expected_io_adjacency)


# ----------------------------------------------------------------------------------------------------
# _is_bipartite
# ----------------------------------------------------------------------------------------------------


def _plain_graph(num_vertices: int, edges: set[tuple[int, int]]) -> ZXGraph:
    return _graph(
        n=num_vertices,
        k=0,
        vertices=[([], False) for _ in range(num_vertices)],
        edges=edges,
    )


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        pytest.param(_plain_graph(0, set()), True, id="empty-graph"),
        pytest.param(_plain_graph(1, set()), True, id="single-isolated-vertex"),
        pytest.param(_plain_graph(2, {(0, 1)}), True, id="single-edge"),
        pytest.param(_plain_graph(4, {(0, 1), (1, 2), (2, 3)}), True, id="path"),
        pytest.param(_plain_graph(4, {(0, 1), (1, 2), (2, 3), (0, 3)}), True, id="even-cycle"),
        pytest.param(
            _plain_graph(
                5,
                {
                    (0, 2),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                },
            ),
            True,
            id="complete-bipartite-k23",
        ),
        pytest.param(_plain_graph(3, {(0, 1), (1, 2), (0, 2)}), False, id="triangle"),
        pytest.param(
            _plain_graph(6, {(0, 1), (1, 2), (0, 2), (3, 4)}),
            False,
            id="disconnected-with-odd-cycle",
        ),
    ],
)
def test_is_bipartite_graph_cases(graph: ZXGraph, expected: bool) -> None:
    assert graph.is_bipartite() is expected



# ----------------------------------------------------------------------------------------------------
# is_lceq_css_kls
# ----------------------------------------------------------------------------------------------------


def test_is_lceq_css_kls_accepts_k_zero_state() -> None:
    assert is_lceq_css_kls(StabilizerCode(["XII", "IZI", "IZY"])) is True


def test_is_lceq_css_kls_random_smoke() -> None:
    for n in range(3, 7):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_kls(code), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_kls_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)
    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)
    assert is_lceq_css_kls(code) is True
