"""Focused checks for the hybrid solution to whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import (
    RandomizeError,
    lc_equivalent_code,
    non_lc_equivalent_code,
    random_stabilizer_code,
)
from src.core.stabilizer_code import StabilizerCode
from src.hybrids.lc_eq import are_lceq

# ----------------------------------------------------------------------------------------------------
# are_lceq
# ----------------------------------------------------------------------------------------------------

def test_are_lceq_preserves_n() -> None:
    assert are_lceq(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is False


def test_are_lceq_preserves_k() -> None:
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_lceq(code1, code2) is False


@pytest.mark.parametrize(
    ("code1", "code2", "expected"),
    [
        pytest.param(
            StabilizerCode(["Z"]),
            StabilizerCode(["X"]),
            True,
            marks=pytest.mark.skip(reason="src.hybrids.lc_eq.are_lceq is not implemented yet."),
            id="one-qubit-z-vs-x",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XI", "IX"]),
            True,
            marks=pytest.mark.skip(reason="src.hybrids.lc_eq.are_lceq is not implemented yet."),
            id="two-product-bases",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XX", "ZZ"]),
            False,
            marks=pytest.mark.skip(reason="src.hybrids.lc_eq.are_lceq is not implemented yet."),
            id="product-vs-bell-state",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            False,
            marks=pytest.mark.skip(reason="src.hybrids.lc_eq.are_lceq is not implemented yet."),
            id="weight-two-vs-weight-one-stabilizer",
        ),
    ],
)
def test_are_lceq_hardcoded_cases(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: bool,
) -> None:
    assert are_lceq(code1, code2) is expected


def test_are_lceq_random_smoke() -> None:
    for n in range(1, 6):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq(code1, code2), bool)


@pytest.mark.skip(reason="src.hybrids.lc_eq.are_lceq is not implemented yet.")
@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 10)])
def test_are_lceq_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)

    assert are_lceq(code1, code2) is True


@pytest.mark.skip(reason="src.hybrids.lc_eq.are_lceq is not implemented yet.")
@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 6)])
def test_are_lceq_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = (2 * seed + 1) % n
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)

    try:
        code2 = non_lc_equivalent_code(code1, seed=2000 + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_lceq(code1, code2) is False
