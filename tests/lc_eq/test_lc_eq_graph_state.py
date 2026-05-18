"""Focused checks for the graph-state machinery to check whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import lc_equivalent_code, random_stabilizer_code, lc_equivalent_code_and_log_ops
from src.algorithms.lc_eq_graph_state import (
    _lc_equiv_graph_states,
    _stab_code_to_stab_state,
    _stab_state_to_graph_state,
    are_lceq_graph_state,
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
    _assert_same_matrix(_stab_state_to_graph_state(tableau.copy(), expected.shape[0]), expected)

# ----------------------------------------------------------------------------------------------------
# _lc_equiv_graph_states
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("g1", "g2", "expected"),
    [
        pytest.param(
            np.array(
                [[0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0]],
                dtype=np.uint8,
            ),
            True, 
            id="one-empty"
        ),
        pytest.param(
            np.array(
                [[0, 0],
                 [0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 0],
                 [0, 0]],
                dtype=np.uint8,
            ),
            True, 
            id="two-empty"
        ),
        pytest.param(
            np.array(
                [[0, 1],
                 [1, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1],
                 [1, 0]],
                dtype=np.uint8,
            ),
            True,
            id="two-same"
        ),
        pytest.param(
            np.array(
                [[0, 0],
                 [0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1],
                 [1, 0]],
                dtype=np.uint8,
            ),
            False,
            id="two-empty-vs-single-edge",
        ),
        pytest.param(
            np.array(
                [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]],
                dtype=np.uint8,
            ),
            False,
            id="three-vertex-path-vs-star",
        ),
        pytest.param(
            np.array(
                [[0, 1, 1, 1],
                 [1, 0, 0, 0],
                 [1, 0, 0, 0],
                 [1, 0, 0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1, 1, 1],
                 [1, 0, 1, 1],
                 [1, 1, 0, 1],
                 [1, 1, 1, 0]],
                dtype=np.uint8,
            ),
            True,
            id="star-vs-complete-graph",
        ),
        pytest.param(
            np.array(
                [[0, 0, 0, 1],
                 [0, 0, 1, 1],
                 [0, 1, 0, 0],
                 [1, 1, 0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1, 1, 0],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [0, 1, 1, 0]],
                dtype=np.uint8,
            ),
            True,
            id="manual-example-tom",
        ),
        pytest.param(
            np.array(
                [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 0]],
                dtype=np.uint8,
            ),
            True,
            id="unconnected-small-same",
        ),
        pytest.param(
            np.array(
                [[0, 0, 1, 0],
                 [0, 0, 1, 0],
                 [1, 1, 0, 0],
                 [0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1, 1, 0],
                 [1, 0, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            True,
            id="unconnected-small-le-eq",
        ),
        pytest.param(
            np.array(
                [[0, 1, 1, 0, 0],
                 [1, 0, 1, 0, 0],
                 [1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 1, 1, 0, 0],
                 [1, 0, 1, 0, 0],
                 [1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0]],
                dtype=np.uint8,
            ),
            True,
            id="unconnected-same",
        ),
        pytest.param(
            np.array(
                [[0, 1, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1],
                 [1, 1, 0, 0, 1, 0]],
                dtype=np.uint8,
            ),
            np.array(
                [[0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            True,
            id="unconnected-lc-eq",
        ),
    ],
)
def test_lc_equiv_graph_states_small_graphs(
    g1: np.ndarray,
    g2: np.ndarray,
    expected: bool,
) -> None:
    assert _lc_equiv_graph_states(g1, g2, g1.shape[0]) is expected

# ----------------------------------------------------------------------------------------------------
# are_lceq_graph_state
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("code1", "code2", "expected"),
    [
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["X"]), True, id="one-qubit-z-vs-x"),
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["Y"]), True, id="one-qubit-z-vs-y"),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XI", "IX"]),
            True,
            id="two-product-bases",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XX", "ZZ"]),
            False,
            id="product-vs-bell-state",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["XX"], z_logicals=["XI"], x_logicals=["ZZ"]),
            True,
            id="repetition-code-under-hadamards",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            False,
            id="weight-two-vs-weight-one-stabilizer",
        ),
    ],
)
def test_are_lceq_graph_state_small_codes(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: bool,
) -> None:
    assert are_lceq_graph_state(code1, code2) is expected

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in [3, 28, 35]])
def test_are_lceq_graph_state_failing_choi(seed: int) -> None:
    """
    3:  < IZI > | < IXI > 
    28: < IXI > | < IYI >
    35: < XZZ > | < ZXX >
    """
    code1 = random_stabilizer_code(3, 2, seed=1000 + seed)
    code2 = lc_equivalent_code_and_log_ops(code1, seed=2000 + seed)
    assert are_lceq_graph_state(code1, code2) is True

def test_are_lceq_graph_state_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code_and_log_ops(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq_graph_state(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 10)])
def test_are_lceq_graph_state_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code_and_log_ops(code1, seed=2000 + seed)
    assert are_lceq_graph_state(code1, code2) is True
