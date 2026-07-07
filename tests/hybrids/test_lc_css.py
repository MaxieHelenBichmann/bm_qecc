"""Focused checks for the hybrid solution to whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import lc_equivalent_code, random_css_code, random_stabilizer_code
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.hybrids.lc_css import is_lceq_css

# ----------------------------------------------------------------------------------------------------
# is_lceq_css
# ----------------------------------------------------------------------------------------------------


@pytest.mark.skip(reason="src.hybrids.lc_css.is_lceq_css is not implemented yet.")
def test_is_lceq_css_accepts_trivial_code() -> None:
    assert is_lceq_css(StabilizerCode.get_trivial_code(3)) is not None


@pytest.mark.skip(reason="src.hybrids.lc_css.is_lceq_css is not implemented yet.")
def test_is_lceq_css_accepts_css_code() -> None:
    code = CSSCode(
        Hx=np.array([[1, 1, 0, 0]], dtype=np.int8),
        Hz=np.array([[0, 0, 1, 1]], dtype=np.int8),
    )

    assert is_lceq_css(code) is not None


@pytest.mark.skip(reason="src.hybrids.lc_css.is_lceq_css is not implemented yet.")
def test_is_lceq_css_hardcoded_lc_positive() -> None:
    code = StabilizerCode(["YX"])

    assert is_lceq_css(code) is not None


@pytest.mark.skip(reason="src.hybrids.lc_css.is_lceq_css is not implemented yet.")
def test_is_lceq_css_hardcoded_negative() -> None:
    code = StabilizerCode(["IZIIII", "IIZZIZ", "ZZIZZZ", "ZIIXIY"])

    assert is_lceq_css(code) is None


def test_is_lceq_css_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            assert isinstance(is_lceq_css(code), (list, type(None)))


@pytest.mark.skip(reason="src.hybrids.lc_css.is_lceq_css is not implemented yet.")
@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_is_lceq_css_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    css_code = random_css_code(n, k, seed=1000 + seed)
    code = lc_equivalent_code(css_code, seed=2000 + seed)

    assert is_lceq_css(code) is not None

@pytest.mark.skip(reason="src.hybrids.lc_css.is_lceq_css is not implemented yet.")
@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in [26]])
def test_is_lceq_css_random_negative(seed: int) -> None:
    code = random_stabilizer_code(6, 2, seed=seed)

    assert is_lceq_css(code) is None
