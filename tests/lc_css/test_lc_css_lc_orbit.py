"""Focused checks for the LC-orbit traversal to whether a stabilizer code with k < 2 is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import lc_equivalent_code, random_stabilizer_code, random_css_code, lc_equivalent_code_and_log_ops
from src.algorithms.lc_css.lc_css_lc_orbit import (
    _stab_code_to_stab_state,
    _stab_state_to_graph_state,
    _traverse_lc_orbit,
    is_lceq_css_lc_orbit,
)
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
def test_stab_state_to_graph_state_small_tableau(
    tableau: np.ndarray,
    expected: np.ndarray,
) -> None:
    _assert_same_matrix(_stab_state_to_graph_state(tableau.copy()), expected)



# ----------------------------------------------------------------------------------------------------
# _traverse_lc_orbit
# ----------------------------------------------------------------------------------------------------

def test_traverse_lc_orbit() -> None:
    triangle = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )

    assert _traverse_lc_orbit(triangle) is True

# ----------------------------------------------------------------------------------------------------
# is_lceq_css_lc_orbit
# ----------------------------------------------------------------------------------------------------


def test_is_lceq_css_lc_orbit_random_smoke() -> None:
    for n in range(1, 4):
        for k in [0, 1]:
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_lc_orbit(code), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_lc_orbit_random_log_ops_restricted(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code_and_log_ops(css_code, seed=2000 + seed)

    assert is_lceq_css_lc_orbit(code) is True

@pytest.mark.parametrize("n", [pytest.param(n, id=f"n-{n}") for n in range(1, 5)])
def test_is_lceq_css_lc_orbit_small_k_random_positive(n: int) -> None:
    seed = 69 + n

    css_state = random_css_code(n, 0, seed=1000 + seed)
    css_small = random_css_code(n, 1, seed=4000 + seed)
    state = lc_equivalent_code(css_state, seed=2000 + seed)
    small = lc_equivalent_code(css_small, seed=3000 + seed)

    assert is_lceq_css_lc_orbit(state) is True
    assert is_lceq_css_lc_orbit(small) is True
