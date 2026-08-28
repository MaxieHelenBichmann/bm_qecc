"""Focused checks for the graph-isomorphism solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import pytest
import ldpc.mod2.mod2_numpy as mod2
from pynauty import certificate, Graph

from benchmarks.experiments.utils import RandomizeError, random_permuted_css_pair, random_non_permuted_css_pair
from src.core.css_code import CSSCode
from src.algorithms.p_css.p_css_graph_iso import (
    _compute_invariant_a,
    _compute_invariant_b,
    _graph_from_invariants,
    _iter_qubit_permutations,
    are_peq_css_graph_iso,
)


def _combined_hull_hash(inv_hx: list[int], inv_hz: list[int]) -> int:
    payload = (
        ",".join(map(str, inv_hx))
        + "|"
        + ",".join(map(str, inv_hz))
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def _positive_pair_from_test_seed(seed: int) -> tuple[CSSCode, CSSCode]:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)
    return random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)


def _is_permutation_equivalent(
    c1: CSSCode,
    c2: CSSCode,
    permutation: tuple[int, ...],
) -> bool:
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return 0
        return mod2.rank(matrix)

    hx_rank = _rank(c1.Hx)
    hz_rank = _rank(c1.Hz)

    if hx_rank != _rank(c2.Hx) or hz_rank != _rank(c2.Hz):
        return False
    if hx_rank and hx_rank != mod2.rank(np.vstack([c1.Hx, c2.Hx[:, permutation]])):
        return False
    if hz_rank and hz_rank != mod2.rank(np.vstack([c1.Hz, c2.Hz[:, permutation]])):
        return False
    return True

# ----------------------------------------------------------------------------------------------------
# _compute_invariant_a
# ----------------------------------------------------------------------------------------------------

def test_invariant_a() -> None:
    code = CSSCode(
        Hx=np.array([[1, 1, 1, 0, 0],
                     [1, 0, 0, 1, 0]], dtype=np.int8),
        Hz=np.array([[0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1]], dtype=np.int8),
    )

    assert _compute_invariant_a(code) == [1, 3, 3, 1, 2]

# ----------------------------------------------------------------------------------------------------
# _compute_invariant_b
# ----------------------------------------------------------------------------------------------------

def test_invariant_b_trivial_code() -> None:
    zero_hull = [1, 0, 0, 0]
    code = CSSCode.get_trivial_code(4)
    expected = [_combined_hull_hash(zero_hull, zero_hull)] * 4

    assert _compute_invariant_b(code) == expected


def test_invariant_b_full_rank() -> None:
    zero_hull = [1, 0, 0, 0]
    code = CSSCode(Hx=np.eye(4, dtype=np.int8), Hz=None)
    expected = [_combined_hull_hash(zero_hull, zero_hull)] * 4

    assert _compute_invariant_b(code) == expected


def test_invariant_b_case1() -> None:
    zero_hull = [1, 0, 0, 0]
    even_pair_hull = [1, 0, 1, 0]
    code = CSSCode(
        Hx=np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.int8,
        ),
        Hz=np.zeros((1, 4), dtype=np.int8),
    )

    assert _compute_invariant_b(code) == [
        _combined_hull_hash(zero_hull, zero_hull),
        _combined_hull_hash(zero_hull, zero_hull),
        _combined_hull_hash(even_pair_hull, zero_hull),
        _combined_hull_hash(even_pair_hull, zero_hull),
    ]


def test_invariant_b_case2() -> None:
    zero_hull = [1, 0, 0, 0, 0]
    even_pair_hull = [1, 0, 1, 0, 0]
    code = CSSCode(
        Hx=np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.int8,
        ),
        Hz=None,
    )

    assert _compute_invariant_b(code) == [
        _combined_hull_hash(zero_hull, zero_hull),
        _combined_hull_hash(zero_hull, zero_hull),
        _combined_hull_hash(even_pair_hull, zero_hull),
        _combined_hull_hash(even_pair_hull, zero_hull),
        _combined_hull_hash(even_pair_hull, zero_hull),
    ]

# ----------------------------------------------------------------------------------------------------
# _graph_from_invariants
# ----------------------------------------------------------------------------------------------------

def test_graph_from_invariants() -> None:
    graph = _graph_from_invariants(3, [[7, 7, 9], [4, 5, 4]])

    assert graph.number_of_vertices == 7
    assert graph.directed is False
    assert graph.vertex_coloring == [{0, 1, 2}, {3, 4}, {5, 6}]
    assert {v: set(neighbors) for v, neighbors in graph.adjacency_dict.items()} == {
        0: {3, 5},
        1: {3, 6},
        2: {4, 5},
        3: {0, 1},
        4: {2},
        5: {0, 2},
        6: {1},
    }

# ----------------------------------------------------------------------------------------------------
# _iter_qubit_permutations
# ----------------------------------------------------------------------------------------------------

def test_iter_qubit_permutations_full_graph() -> None:
    g1 = Graph(
        number_of_vertices=7,
        directed=False,
        adjacency_dict={
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1, 4, 6],
            3: [1],
            4: [2, 5],
            5: [4],
            6: [2],
        },
    )

    g2 = Graph(
        number_of_vertices=7,
        directed=False,
        adjacency_dict={
            0: [5],
            1: [2, 5, 6],
            2: [1, 5],
            3: [4, 5],
            4: [3],
            5: [0, 1, 2, 3],
            6: [1],
        },
    )

    expected_permutations = {(2, 1, 5, 6, 3, 4, 0)}

    assert set(_iter_qubit_permutations(g1, g2, n=7)) == expected_permutations

def test_iter_qubit_permutations_only_qubit() -> None:
    g1 = Graph(
        number_of_vertices=7,
        directed=False,
        adjacency_dict={
            0: [3, 4],
            1: [5, 4],
            2: [4, 5, 6],
            3: [0],
            4: [0, 1, 2],
            5: [1, 2],
            6: [2],
        },
    )

    g2 = Graph(
        number_of_vertices=7,
        directed=False,
        adjacency_dict={
            0: [3, 4],
            1: [3, 4, 5],
            2: [4, 6],
            3: [0, 1],
            4: [0, 1, 2],
            5: [1],
            6: [2],
        },
    )

    expected_permutations = {(2, 0, 1)}

    assert set(_iter_qubit_permutations(g1, g2, n=3)) == expected_permutations

def test_graph_iso_uses_matching_permutation_convention() -> None:
    code1 = CSSCode(
        Hx=np.array([[1, 0, 1, 1, 1],
                     [0, 0, 1, 0, 1],
                     [1, 0, 1, 0, 0]], dtype=np.int8),
        Hz=None,
    )

    code2 = CSSCode(
        Hx=np.array([[0, 1, 1, 1, 1],
                     [0, 1, 0, 0, 1],
                     [0, 0, 1, 0, 1]], dtype=np.int8),
        Hz=None,
    )

    assert are_peq_css_graph_iso(code1, code2) is True

# ----------------------------------------------------------------------------------------------------
# are_peq_css_graph_iso
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [pytest.param(4, id="seed-4"), pytest.param(6, id="seed-6")])
def test_random_positive_seed_4_and_6_extract_checkable_candidates(seed: int) -> None:
    code1, code2 = _positive_pair_from_test_seed(seed)

    invariants_c1 = [_compute_invariant_a(code1), _compute_invariant_b(code1)]
    invariants_c2 = [_compute_invariant_a(code2), _compute_invariant_b(code2)]

    assert Counter(invariants_c1[0]) == Counter(invariants_c2[0])
    assert Counter(invariants_c1[1]) == Counter(invariants_c2[1])

    graph_c1 = _graph_from_invariants(code1.n, invariants_c1)
    graph_c2 = _graph_from_invariants(code2.n, invariants_c2)

    assert certificate(graph_c1) == certificate(graph_c2)

    candidate_permutations = set(
        _iter_qubit_permutations(graph_c1, graph_c2, code1.n)
    )

    assert candidate_permutations
    assert any(
        _is_permutation_equivalent(code1, code2, permutation)
        for permutation in candidate_permutations
    )
    assert are_peq_css_graph_iso(code1, code2) is True


def test_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_css_graph_iso(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_graph_iso(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css_graph_iso(code1, code2) is False
