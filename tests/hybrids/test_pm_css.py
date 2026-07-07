"""Focused checks for the hybrid solution to whether two CSS codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import RandomizeError, random_non_permuted_css_pair, random_permuted_css_pair
from src.core.css_code import CSSCode
from src.hybrids.p_css import are_peq_css

# ----------------------------------------------------------------------------------------------------
# are_peq_css
# ----------------------------------------------------------------------------------------------------

def test_are_peq_css_preserves_n() -> None:
    assert are_peq_css(CSSCode(n=3), CSSCode(n=4)) is None


def test_are_peq_css_preserves_k() -> None:
    code1 = CSSCode(n=4)
    code2 = CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8))

    assert are_peq_css(code1, code2) is None


def test_are_peq_css_preserves_x_and_z_ranks() -> None:
    code1 = CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8))
    code2 = CSSCode(Hz=np.array([[1, 0, 0, 0]], dtype=np.int8))

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_peq_css(code1, code2) is None


def test_are_peq_css_hardcoded_positive() -> None:
    code1 = CSSCode(
        Hx=np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8),
        Hz=np.array([[1, 1, 1, 1]], dtype=np.int8),
    )
    code2 = CSSCode(
        Hx=np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8),
        Hz=np.array([[1, 1, 1, 1]], dtype=np.int8),
    )

    assert are_peq_css(code1, code2) is not None


def test_are_peq_css_random_smoke() -> None:
    for n in range(3, 7):
        for k in range(1, n):
            try:
                code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_css(code1, code2), (list, type(None)))
            except RandomizeError:
                pass


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css(code1, code2) is not None


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_css_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_css_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_css(code1, code2) is None
