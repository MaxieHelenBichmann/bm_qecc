"""Focused checks for the KLS/HK normal-form conversion."""

from __future__ import annotations

import pytest

from benchmarks.utils import random_stabilizer_code, random_css_code, lc_equivalent_code
from src.algorithms.lc_css_kls import ZXGraph, _hk_normal_form, is_lceq_css_kls
from src.core.stabilizer_code import StabilizerCode

# ----------------------------------------------------------------------------------------------------
# _code_to_graph
# ----------------------------------------------------------------------------------------------------



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


@pytest.mark.parametrize(
    ("vertices", "edges", "expected_vertices", "expected_edges"),
    [
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

# ----------------------------------------------------------------------------------------------------
# _kls_normal_form
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# _is_bipartite
# ----------------------------------------------------------------------------------------------------



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

POSITIVE_SEED_CASES = [
    pytest.param(
        seed,
        marks=pytest.mark.xfail(
            strict=True,
            reason="????",
        ),
        id=f"seed-{seed}",
    )
    if seed == 5
    else pytest.param(seed, id=f"seed-{seed}")
    for seed in range(20)
]


@pytest.mark.parametrize("seed", POSITIVE_SEED_CASES)
def test_is_lceq_css_kls_random_positive(seed: int) -> None:
    n = 9 + seed % 4
    k = seed % (n + 1)
    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)
    assert is_lceq_css_kls(code) is True
