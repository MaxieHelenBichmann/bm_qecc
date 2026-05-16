"""Focused checks for the KLS/HK normal-form conversion.

Run as a module from the repository root:

    .venv/bin/python -m tests.test_lc_css_kls_hk
"""

from __future__ import annotations

from benchmarks.utils import random_stabilizer_code
from src.algorithms.lc_css_kls import ZXGraph, _hk_normal_form, is_lceq_css_kls
from src.core.stabilizer_code import StabilizerCode

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


def test_reduce_trailing_hs() -> None:
    graph = _hk(
        [(["H", "S"], False), ([], False), ([], False)],
        {(0, 1), (0, 2)},
    )

    _assert_graph(
        graph,
        [(["S"], True), (["S"], True), (["S"], True)],
        {(0, 1), (0, 2), (1, 2)},
    )


def test_reduce_solo_trailing_sh() -> None:
    graph = _hk(
        [(["S", "H"], False), ([], False), ([], False)],
        {(0, 1), (0, 2)},
    )

    _assert_graph(
        graph,
        [(["H"], False), (["S"], False), (["S"], False)],
        {(0, 1), (0, 2), (1, 2)},
    )


def test_reduce_shared_trailing_sh() -> None:
    graph = _hk(
        [(["S", "H"], False), (["H"], False), ([], False)],
        {(0, 1), (0, 2)},
    )

    _assert_graph(
        graph,
        [(["S"], False), ([], False), ([], False)],
        {(0, 1), (1, 2)},
    )


def test_h_slide_and_sh_cleanup() -> None:
    graph = _hk(
        [(["S"], False), (["S"], False), (["H"], False), (["H"], False)],
        {(0, 2), (1, 3)},
    )

    _assert_graph(
        graph,
        [(["H"], False), (["H"], False), ([], True), ([], True)],
        {(0, 2), (1, 3)},
    )


def test_is_lceq_css_kls_accepts_k_zero_state() -> None:
    assert is_lceq_css_kls(StabilizerCode(["XII", "IZI", "IZY"])) is True


def test_is_lceq_css_kls_random_smoke() -> None:
    for n in range(3, 7):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_kls(code), bool)


if __name__ == "__main__":
    print("- Running HK normal form tests...")
    test_reduce_trailing_hs()
    test_reduce_solo_trailing_sh()
    test_reduce_shared_trailing_sh()
    test_h_slide_and_sh_cleanup()
    print("  + HK normal form tests passed")
    print("- Running full KLS tests...")
    test_is_lceq_css_kls_accepts_k_zero_state()
    test_is_lceq_css_kls_random_smoke()
    print("  + Full KLS tests passed")
