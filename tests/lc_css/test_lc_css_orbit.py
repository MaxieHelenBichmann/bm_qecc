"""Focused checks for the LC-orbit traversal to whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import lc_equivalent_code, random_stabilizer_code, random_css_code
from src.algorithms.lc_css.lc_css_orbit import (
    RedStabGraph,
    _stab_code_to_stab_state,
    _stab_state_to_graph_state,
    _traverse_cliff_orbit,
    is_lceq_css_orbit,
)
from src.core.stabilizer_code import StabilizerCode


def _assert_same_matrix(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual.astype(np.uint8), expected.astype(np.uint8))


def _red_graph(
    vertices: list[tuple[bool, bool, bool]],
    edges: set[tuple[int, int]],
    k: int = 0,
) -> RedStabGraph:
    graph = RedStabGraph()
    graph.n = len(vertices) - k
    graph.k = k
    graph.vertices = vertices.copy()
    graph.edges = {tuple(sorted(edge)) for edge in edges}
    return graph


# ----------------------------------------------------------------------------------------------------
# RedStabGraph
# ----------------------------------------------------------------------------------------------------


def test_red_stab_graph_rejects_self_edges() -> None:
    graph = _red_graph(
        [(False, True, False), (False, True, False)],
        {(0, 0)},
    )

    assert graph.is_valid() is False
    with pytest.raises(ValueError, match="Self-loops"):
        graph.toggle_edge(0, 0)


def test_apply_cz_mixed_adjacent_pair_does_not_create_self_edge() -> None:
    graph = _red_graph(
        [
            (False, False, False),
            (False, True, False),
            (False, True, False),
        ],
        {(0, 1), (0, 2), (1, 2)},
    )

    result = graph.apply_cz(0, 1)

    assert (1, 1) not in result.edges
    assert result.edges == {(0, 1), (0, 2)}
    assert result.is_valid()

# ----------------------------------------------------------------------------------------------------
# _stab_code_to_stab_state
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(
            StabilizerCode(["Z"]),
            np.array([[0, 1]], dtype=np.uint8),
            id="single-qubit-z-state",
        ),
        pytest.param(
            StabilizerCode(["X"]),
            np.array([[1, 0]], dtype=np.uint8),
            id="single-qubit-x-state",
        ),
        pytest.param(
            StabilizerCode.get_trivial_code(1),
            np.array(
                [
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                ],
                dtype=np.uint8,
            ),
            id="one-qubit-trivial-code",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            np.array(
                [
                    [0, 0, 0, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1],
                ],
                dtype=np.uint8,
            ),
            id="two-qubit-repetition-code",
        ),
        pytest.param(
            StabilizerCode(["ZZI", "IZZ"], z_logicals=["ZII"], x_logicals=["XXX"]),
            np.array(
                [
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                ],
                dtype=np.uint8,
            ),
            id="three-qubit-repetition-code",
        ),
        pytest.param(
            StabilizerCode(["XZYI", "IXXY"]),
            np.array(
                [
                    [1, 0, 1, 0, 0, 0,  0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0,  0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0,  0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1,  1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0,  0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 1],
                ],
                dtype=np.uint8,
            ),
            id="log-op-generators",
        ),
    ],
)
def test_stab_code_to_stab_state_small_codes(code: StabilizerCode, expected: np.ndarray) -> None:
    _assert_same_matrix(_stab_code_to_stab_state(code), expected)


# ----------------------------------------------------------------------------------------------------
# _stab_state_to_graph_state
# ----------------------------------------------------------------------------------------------------


def test_stab_state_to_graph_state_tracks_solid_vertices_through_swaps() -> None:
    tableau = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    graph = _stab_state_to_graph_state(tableau, n=2, k=0)

    assert graph.vertices == [(False, False, False), (False, True, False)]
    assert graph.edges == set()
    assert graph.is_valid()


@pytest.mark.parametrize(
    ("n", "k", "seed"),
    [
        pytest.param(3, 0, 130, id="n3-k0"),
        pytest.param(3, 1, 131, id="n3-k1"),
        pytest.param(4, 1, 141, id="n4-k1"),
        pytest.param(6, 1, 161, id="n6-k1"),
    ],
)
def test_stab_state_to_graph_state_returns_valid_reduced_graph(n: int, k: int, seed: int) -> None:
    code = random_stabilizer_code(n, k, seed=seed)
    tableau = _stab_code_to_stab_state(code)

    graph = _stab_state_to_graph_state(tableau, n=code.n + code.k, k=code.k)

    assert graph.is_valid()


# ----------------------------------------------------------------------------------------------------
# _traverse_cliff_orbit
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# is_lceq_css_orbit
# ----------------------------------------------------------------------------------------------------


def test_is_lceq_css_orbit_random_smoke() -> None:
    for n in range(1, 4):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_orbit(code), bool)


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_orbit_random_positive(seed: int) -> None:
    n = 2 + (2 * seed + 1) % 3
    k = 1 + (seed // 3) % (n - 1)
    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)
    assert is_lceq_css_orbit(code) is True
