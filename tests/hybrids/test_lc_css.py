"""Focused checks for the hybrid solution to whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import pytest
import src.hybrids.lc_css as lc_css

from benchmarks.utils import lc_equivalent_code, random_css_code, random_stabilizer_code
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids.lc_css import _row_basis, _sat, is_lceq_css

# ----------------------------------------------------------------------------------------------------
# is_lceq_css
# ----------------------------------------------------------------------------------------------------


def test_is_lceq_css_accepts_trivial_code() -> None:
    assert is_lceq_css(StabilizerCode.get_trivial_code(3)) is True

def test_is_lceq_css_accepts_css_code() -> None:
    code = CSSCode(
        Hx=np.array([[1, 1, 0, 0]], dtype=np.int8),
        Hz=np.array([[0, 0, 1, 1]], dtype=np.int8),
    )

    assert is_lceq_css(code) is True


def test_hybrid_routes_small_codes_to_bruteforce(monkeypatch: pytest.MonkeyPatch) -> None:
    code = StabilizerCode(["XXX"])
    calls = 0

    def fake_bruteforce(tableau: np.ndarray) -> bool:
        nonlocal calls
        calls += 1
        assert tableau.shape == (1, 2 * code.n)
        return True

    monkeypatch.setattr(lc_css, "_bruteforce", fake_bruteforce)
    monkeypatch.setattr(lc_css, "_sat", lambda *_args: pytest.fail("SAT must not run for n < 4"))

    assert is_lceq_css(code) is True
    assert calls == 1


def test_hybrid_routes_large_codes_to_sat(monkeypatch: pytest.MonkeyPatch) -> None:
    code = StabilizerCode(["XXXX"])
    calls = 0

    def fake_sat(tableau: np.ndarray) -> bool:
        nonlocal calls
        calls += 1
        assert tableau.shape == (1, 2 * code.n)
        return True

    monkeypatch.setattr(lc_css, "_bruteforce", lambda *_args: pytest.fail("brute force must not run for n >= 4"))
    monkeypatch.setattr(lc_css, "_sat", fake_sat)

    assert is_lceq_css(code) is True
    assert calls == 1


def test_is_lceq_css_hardcoded_lc_positive() -> None:
    code = StabilizerCode(["YX"])

    assert is_lceq_css(code) is True


def test_is_lceq_css_hardcoded_negative() -> None:
    code = StabilizerCode(["IZIIII", "IIZZIZ", "ZZIZZZ", "ZIIXIY"])

    assert is_lceq_css(code) is False


def test_is_lceq_css_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css(code), bool)


@pytest.mark.parametrize(
    ("n", "k", "seed"),
    [
        pytest.param(1, 0, 11, id="one-qubit-state"),
        pytest.param(3, 0, 12, id="three-qubit-state"),
        pytest.param(2, 1, 13, id="small-one-logical"),
        pytest.param(5, 1, 14, id="larger-one-logical-bruteforce"),
        pytest.param(6, 2, 15, id="six-qubit-sat-direct"),
    ],
)
def test_is_lceq_css_positive_for_lc_equivalent_css_codes(n: int, k: int, seed: int) -> None:
    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)

    assert is_lceq_css(code) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)

    assert is_lceq_css(code) is True

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in [26]])
def test_is_lceq_css_random_negative(seed: int) -> None:
    code = random_stabilizer_code(6, 2, seed=seed)

    assert is_lceq_css(code) is False

# ----------------------------------------------------------------------------------------------------
# _sat
# ----------------------------------------------------------------------------------------------------


def test_sat_returns_bool() -> None:
    code = CSSCode(
        Hx=np.array([[1, 1, 0, 0]], dtype=np.int8),
        Hz=np.array([[0, 0, 1, 1]], dtype=np.int8),
    )

    assert isinstance(_sat(_row_basis(code.symplectic)), bool)


def test_sat_accepts_css_equivalent_code() -> None:
    css_code = random_css_code(6, 2, seed=1015)
    code = lc_equivalent_code(css_code, seed=2015)

    assert _sat(_row_basis(code.symplectic)) is True


def test_sat_rejects_non_css_equivalent_code() -> None:
    code = random_stabilizer_code(6, 2, seed=26)

    assert _sat(_row_basis(code.symplectic)) is False
