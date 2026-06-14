"""Focused checks for the matroid solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import RandomizeError, random_permuted_css_pair, random_non_permuted_css_pair
from src.algorithms.p_css.p_css_matroid import _circuits_binary_matroid, _graph_from_circuits, are_peq_css_matroid

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

    assert _circuits_binary_matroid(matrix) == [0b111]


def test_circuits_binary_matroid_matches_direct_enumeration() -> None:
    def direct_circuits(matrix: np.ndarray) -> set[int]:
        import ldpc.mod2.mod2_numpy as mod2

        def support_as_mask(vector: np.ndarray) -> int:
            support = 0
            for col in np.flatnonzero(vector):
                support |= 1 << int(col)
            return support

        kernel = mod2.nullspace(matrix)
        if hasattr(kernel, "toarray"):
            kernel = kernel.toarray()

        kernel = (np.asarray(kernel) & 1).astype(np.uint8, copy=False)
        if kernel.size == 0:
            return set()

        candidates: list[int] = []
        for mask in range(1, 1 << kernel.shape[0]):
            vector = np.zeros(kernel.shape[1], dtype=np.uint8)
            for i in range(kernel.shape[0]):
                if (mask >> i) & 1:
                    vector ^= kernel[i]

            support = support_as_mask(vector)
            if support:
                candidates.append(support)

        candidates.sort(key=int.bit_count)
        circuits: list[int] = []
        for support in candidates:
            if not any((circuit & support) == circuit for circuit in circuits):
                circuits.append(support)

        return set(circuits)

    rng = np.random.default_rng(1234)
    matrices = [
        np.zeros((2, 4), dtype=np.int8),
        np.eye(4, dtype=np.int8),
        np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.int8),
    ]
    matrices.extend(
        rng.integers(0, 2, size=(rows, cols), dtype=np.int8)
        for rows, cols in [(2, 5), (3, 6), (4, 6)]
    )

    for matrix in matrices:
        assert set(_circuits_binary_matroid(matrix)) == direct_circuits(matrix)

# ----------------------------------------------------------------------------------------------------
# _graph_from_circuits
# ----------------------------------------------------------------------------------------------------

def test_graph_from_circuits_small_incidence_graph() -> None:
    graph = _graph_from_circuits(
        3,
        circuits_hx=[(1 << 0) | (1 << 2)],
        circuits_hz=[(1 << 1) | (1 << 2)],
    )

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
        code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_matroid(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_matroid_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_matroid(code1, code2) is False
