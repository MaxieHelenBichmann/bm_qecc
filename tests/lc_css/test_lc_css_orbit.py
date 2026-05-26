"""Focused checks for the LC-orbit traversal to whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import lc_equivalent_code, random_stabilizer_code, random_css_code
from src.algorithms.lc_css.lc_css_orbit import _stab_code_to_stab_state, _stab_state_to_graph_state, _traverse_lc_orbit, is_lceq_css_orbit

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

# ----------------------------------------------------------------------------------------------------
# _traverse_cliff_orbit
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# is_lceq_css_orbit
# ----------------------------------------------------------------------------------------------------

def test_is_lceq_css_orbit_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css_orbit(code), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_orbit_random_positive(seed: int) -> None:
    n = 2 + (5 * seed + 1) % 8
    k = 1 + (3 * seed + 1) % (n - 1)
    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)
    assert is_lceq_css_orbit(code) is True
