"""Focused checks for the SAT approach to check whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.experiments.utils import lc_equivalent_code, non_lc_equivalent_code, random_stabilizer_code
from src.core.stabilizer_code import StabilizerCode
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat

# ----------------------------------------------------------------------------------------------------
# are_lceq_sat
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
def test_are_lceq_sat_small_codes(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: bool,
) -> None:
    assert are_lceq_sat(code1, code2) is expected

def test_are_lceq_sat_random_smoke() -> None:
    for n in range(1, 6):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq_sat(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 20)])
def test_are_lceq_sat_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)
    assert are_lceq_sat(code1, code2) is True

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 6)])
def test_are_lceq_sat_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = (2 * seed + 1) % n
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = non_lc_equivalent_code(code1, seed=2000 + seed)

    assert are_lceq_sat(code1, code2) is False
