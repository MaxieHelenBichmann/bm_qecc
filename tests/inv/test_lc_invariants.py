"""Focused checks for LC-invariant filters for stabilizer codes."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.core.stabilizer_code import StabilizerCode
from src.invariants.lc_eq.lc_invariants import (
    preserved_k,
    preserved_local_weight_distribution,
    preserved_low_degree_local_invariant,
    preserved_n,
)


def _all_subsets(n: int) -> list[tuple[int, ...]]:
    return [subset for size in range(n + 1) for subset in combinations(range(n), size)]


def _row_space_words(code: StabilizerCode) -> list[np.ndarray]:
    matrix = np.asarray(code.symplectic, dtype=np.uint8) & 1
    words = []

    for mask in range(1 << matrix.shape[0]):
        word = np.zeros(matrix.shape[1], dtype=np.uint8)
        for row in range(matrix.shape[0]):
            if (mask >> row) & 1:
                word ^= matrix[row]
        words.append(word)

    return words


def _support(word: np.ndarray) -> set[int]:
    n = word.shape[0] // 2
    return {q for q in range(n) if word[q] or word[q + n]}


def _dim_from_count(count: int) -> int:
    assert count > 0 and count & (count - 1) == 0
    return count.bit_length() - 1


def _brute_support_subcode_dim(code: StabilizerCode, subset: tuple[int, ...]) -> int:
    allowed = set(subset)
    count = sum(_support(word) <= allowed for word in _row_space_words(code))
    return _dim_from_count(count)


def _brute_low_degree_profile(code: StabilizerCode) -> tuple[int, ...]:
    return tuple(_brute_support_subcode_dim(code, subset) for subset in _all_subsets(code.n))


def _brute_local_weight_dim(
    code: StabilizerCode,
    w1: tuple[int, ...],
    w2: tuple[int, ...],
    w12: tuple[int, ...],
) -> int:
    allowed1 = set(w1)
    allowed2 = set(w2)
    allowed12 = set(w12)
    words = _row_space_words(code)

    count = 0
    for v1 in words:
        if not _support(v1) <= allowed1:
            continue
        for v2 in words:
            if _support(v2) <= allowed2 and _support(v1 ^ v2) <= allowed12:
                count += 1

    return _dim_from_count(count)


def _brute_local_weight_profile(code: StabilizerCode) -> tuple[int, ...]:
    subsets = _all_subsets(code.n)
    return tuple(
        _brute_local_weight_dim(code, w1, w2, w12)
        for w1 in subsets
        for w2 in subsets
        for w12 in subsets
    )


def test_preserved_n_and_k_detect_basic_mismatches() -> None:
    assert not preserved_n(StabilizerCode(["Z"]), StabilizerCode(["ZI"]))
    assert not preserved_k(StabilizerCode(["ZZ"]), StabilizerCode(["ZI", "IZ"]))


def test_lc_invariants_are_preserved_by_single_qubit_cliffords() -> None:
    z_product_state = StabilizerCode(["ZI", "IZ"])
    x_product_state = StabilizerCode(["XI", "IX"])

    assert preserved_n(z_product_state, x_product_state)
    assert preserved_k(z_product_state, x_product_state)
    assert preserved_low_degree_local_invariant(z_product_state, x_product_state)
    assert preserved_local_weight_distribution(z_product_state, x_product_state)


def test_low_degree_local_invariant_matches_support_subcode_definition() -> None:
    product_state = StabilizerCode(["ZI", "IZ"])
    bell_state = StabilizerCode(["XX", "ZZ"])

    assert _brute_support_subcode_dim(product_state, (0,)) == 1
    assert _brute_support_subcode_dim(bell_state, (0,)) == 0

    expected = _brute_low_degree_profile(product_state) == _brute_low_degree_profile(bell_state)

    assert expected is False
    assert preserved_low_degree_local_invariant(product_state, bell_state) is expected


def test_local_weight_distribution_matches_paired_support_definition() -> None:
    product_state = StabilizerCode(["ZI", "IZ"])
    bell_state = StabilizerCode(["XX", "ZZ"])

    assert _brute_local_weight_dim(product_state, (0,), (1,), (0, 1)) == 2
    assert _brute_local_weight_dim(bell_state, (0,), (1,), (0, 1)) == 0

    expected = _brute_local_weight_profile(product_state) == _brute_local_weight_profile(bell_state)

    assert expected is False
    assert preserved_local_weight_distribution(product_state, bell_state) is expected


@pytest.mark.parametrize(
    ("code1", "code2"),
    [
        pytest.param(StabilizerCode(["ZZ"]), StabilizerCode(["XX"]), id="repetition-under-hadamards"),
        pytest.param(StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XI", "IX"]), id="product-basis-change"),
    ],
)
def test_reference_profiles_agree_for_simple_lc_equivalent_codes(
    code1: StabilizerCode,
    code2: StabilizerCode,
) -> None:
    assert _brute_low_degree_profile(code1) == _brute_low_degree_profile(code2)
    assert _brute_local_weight_profile(code1) == _brute_local_weight_profile(code2)
