"""Regression tests for pynauty colour-class handling in the graph reductions.

``pynauty`` identifies a colour class by its position in ``vertex_coloring`` and
silently drops empty classes, shifting every later one. Any reduction that
encodes a distinction purely by class position is therefore wrong whenever one of
those classes can be empty.

``are_peq_stab_graph_iso`` hit exactly this: a purely X-type stabilizer group
contributes no z-edges and a purely Z-type one no x-edges, so ``{IXI}`` compared
isomorphic to ``{IZI}``. It now anchors each edge class with an isolated vertex.
"""

from __future__ import annotations

import numpy as np
import pytest
from pynauty import Graph, certificate

from benchmarks.experiments.generators_random import (
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from src.algorithms.lc_stb.lc_stb_graph_iso import _graph_from_code as lc_graph_from_code
from src.algorithms.p_css.p_css_bruteforce import are_peq_css_bruteforce
from src.algorithms.p_css.p_css_graph_iso import are_peq_css_graph_iso
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_stb.p_stab_bruteforce import are_peq_stab_bruteforce
from src.algorithms.p_stb.p_stab_graph_iso import (
    _graph_from_code as stab_graph_from_code,
)
from src.algorithms.p_stb.p_stab_graph_iso import are_peq_stab_graph_iso
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode

#: Pairs that differ only by exchanging X and Z, so one edge class is empty on
#: each side. Every one was accepted before the anchor vertices were added.
PURE_TYPE_PAIRS = (
    (["IXI"], ["IZI"]),
    (["XXI", "IXX"], ["ZZI", "IZZ"]),
    (["XXXX"], ["ZZZZ"]),
    (["XXX"], ["ZZZ"]),
)


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

    # Class order is honoured ...
    assert certificate(build([{0}, {1}, {2}])) != certificate(build([{2}, {1}, {0}]))
    # ... but an empty class is dropped, so these two compare equal.
    assert certificate(build([{0}, {1}, set(), {2}])) == certificate(
        build([{0}, {1}, {2}, set()])
    )


@pytest.mark.parametrize("first,second", PURE_TYPE_PAIRS)
def test_pure_x_and_pure_z_codes_are_not_isomorphic(
    first: list[str], second: list[str]
) -> None:
    """A permutation cannot turn X into Z, and the reduction must agree."""
    code1, code2 = StabilizerCode(first), StabilizerCode(second)
    assert are_peq_stab_bruteforce(code1, code2) is False
    assert are_peq_stab_graph_iso(code1, code2) is False


@pytest.mark.parametrize("generators", [["IXI"], ["ZZI", "IZZ"], ["XXXX"]])
def test_every_edge_class_stays_non_empty(generators: list[str]) -> None:
    """Both edge classes carry at least their anchor, whatever the code type."""
    graph = stab_graph_from_code(StabilizerCode(generators))
    z_class, x_class = graph.vertex_coloring[2], graph.vertex_coloring[3]
    assert z_class and x_class
    # The anchors are isolated, so they add no adjacency.
    for anchor_class in (z_class, x_class):
        for vertex in anchor_class:
            assert vertex < graph.number_of_vertices


def test_anchors_do_not_change_isomorphism_of_equivalent_codes() -> None:
    """Adding the anchors must not make equivalent codes look different."""
    for n, k in ((4, 1), (5, 2), (6, 3)):
        for seed in range(6):
            code1, code2 = PEqCodePairGenerator.stabilizer_codes_permuted(n, k, seed=seed)
            assert are_peq_stab_graph_iso(code1, code2) is True


@pytest.mark.parametrize("n,k", [(3, 1), (4, 2), (5, 2), (6, 3)])
def test_graph_iso_agrees_with_brute_force(n: int, k: int) -> None:
    """The reduction matches exhaustive search on both polarities."""
    for seed in range(6):
        for code1, code2 in (
            PEqCodePairGenerator.stabilizer_codes_permuted(n, k, seed=seed),
            NonPEqCodePairGenerator.stabilizer_codes_independent_candidate(n, k, seed=seed),
        ):
            assert are_peq_stab_graph_iso(code1, code2) == are_peq_stab_bruteforce(
                code1, code2
            )


def test_lc_graph_reduction_has_no_empty_classes() -> None:
    """``lc_stb_graph_iso`` is safe: its classes are fixed-size and always occupied."""
    for generators in (["IXI"], ["ZZI", "IZZ"], ["XXXX"]):
        graph = lc_graph_from_code(StabilizerCode(generators))
        assert all(colour_class for colour_class in graph.vertex_coloring)


def test_css_reductions_handle_an_empty_circuit_class() -> None:
    """The CSS reductions survive a code whose X or Z matroid has no circuits.

    ``are_peq_css_matroid`` is shielded by its circuit-count early exit: it
    returns before comparing certificates whenever the two codes disagree on the
    number of circuits, so the two graphs can never differ in which class is
    empty.
    """
    identity = np.eye(2, dtype=np.int8)
    empty = np.zeros((0, 2), dtype=np.int8)
    all_x = CSSCode(Hx=identity, Hz=empty)
    all_z = CSSCode(Hx=empty, Hz=identity)

    assert are_peq_css_bruteforce(all_x, all_z) is False
    assert are_peq_css_matroid(all_x, all_z) is False
    assert are_peq_css_graph_iso(all_x, all_z) is False
