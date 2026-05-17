"""Focused checks for the graph-state machinery to check whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import RandomizeError, random_permuted_stabilizer_pair_and_log_ops, random_non_permuted_stabilizer_pair
from src.algorithms.p_stab_graph_state import _stab_code_to_stab_state, _stab_state_to_graph_state, _traverse_lc_orbit, _extract_qubit_permutations, are_peq_stab_graph_state

from src.core.stabilizer_code import StabilizerCode

def _assert_same_matrix(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual.astype(np.uint8), expected.astype(np.uint8))


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

@pytest.mark.parametrize(
    ("tableau", "expected"),
    [
        pytest.param(
            np.array([[1, 0]], dtype=np.uint8),
            np.array([[0]], dtype=np.uint8),
            id="one-isolated-vertex",
        ),
        pytest.param(
            np.array([[1, 0, 0, 0, 1, 1],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 1, 1, 1, 0]], dtype=np.uint8),
            np.array([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]], dtype=np.uint8),
            id="triangle",
        ),
        pytest.param(
            np.array(
                [[0, 0, 1, 0],
                 [0, 0, 0, 1]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 0],
                 [0, 0]],
                dtype=np.uint8,
            ),
            id="only-hadamard-improvement",
        ),
        pytest.param(
            np.array(
                [[0, 0, 0, 0, 1, 0],
                 [1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 1, 1]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                dtype=np.uint8,
            ),
            id="mixed",
        ),
    ],
)
def test_stab_state_to_graph_state_small_tableaux(
    tableau: np.ndarray,
    expected: np.ndarray,
) -> None:
    _assert_same_matrix(_stab_state_to_graph_state(tableau.copy(), expected.shape[0]), expected)


# ----------------------------------------------------------------------------------------------------
# _traverse_lc_orbit
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# _extract_qubit_permutations
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# are_peq_stab_graph_state
# ----------------------------------------------------------------------------------------------------

def test_are_peq_stab_graph_state_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_stabilizer_pair_and_log_ops(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_stab_graph_state(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_graph_state_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_stabilizer_pair_and_log_ops(n, k, seed=1000 + 17 * n + k)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_graph_state(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_graph_state_random_negative(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_graph_state(code1, code2) is False