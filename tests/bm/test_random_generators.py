"""Focused tests for the random benchmark generator API."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from benchmarks.experiments.generators_random import (
    LCEqCodeGenerator,
    LCEqCodePairGenerator,
    NonLCEqCodeGenerator,
    NonLCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
    _lc_projection_rank_invariant,
    _projection_rank_invariant,
)
from benchmarks.experiments.utils import _rank_binary, random_stabilizer_code
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode
from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce


GENERATOR_CLASSES = (
    PEqCodePairGenerator,
    NonPEqCodePairGenerator,
    LCEqCodePairGenerator,
    NonLCEqCodePairGenerator,
)


def _matrix(code: StabilizerCode) -> np.ndarray:
    return np.asarray(code.symplectic, dtype=np.int8)


def test_random_stabilizer_initial_support_is_seeded_and_independent() -> None:
    supports = []
    for seed in range(5):
        code = random_stabilizer_code(8, 5, seed=seed, clifford_steps=0)
        matrix = _matrix(code)
        assert _rank_binary(matrix) == 3
        assert np.count_nonzero(matrix[:, :8]) == 0
        occupied = tuple(np.flatnonzero(np.any(matrix[:, 8:], axis=0)))
        assert len(occupied) == 3
        supports.append(occupied)

    assert len(set(supports)) > 1
    assert any(support != (0, 1, 2) for support in supports)


def test_layered_random_stabilizer_code_is_seeded_and_preserves_rank() -> None:
    first = random_stabilizer_code(9, 3, seed=42, clifford_steps=4)
    second = random_stabilizer_code(9, 3, seed=42, clifford_steps=4)
    assert np.array_equal(_matrix(first), _matrix(second))
    assert _rank_binary(_matrix(first)) == 6


def test_lc_css_generators_have_the_common_single_code_signature() -> None:
    methods = (
        LCEqCodeGenerator.stabilizer_code_local_clifford,
        NonLCEqCodeGenerator.stabilizer_code_locally_rank_one,
    )
    for method in methods:
        assert list(inspect.signature(method).parameters)[:3] == ["n", "k", "seed"]
        assert "Sampling bias:" in (method.__doc__ or "")
        assert "NOT USABLE whenever" in (method.__doc__ or "")


def test_positive_lc_css_generator_returns_code_in_a_css_lc_orbit() -> None:
    code = LCEqCodeGenerator.stabilizer_code_local_clifford(3, 1, 17)
    assert isinstance(code, StabilizerCode)
    assert (code.n, code.k) == (3, 1)
    assert is_lceq_css_bruteforce(code)


def test_negative_lc_css_generator_returns_code_outside_every_css_lc_orbit() -> None:
    code = NonLCEqCodeGenerator.stabilizer_code_locally_rank_one(5, 1, 23)
    assert isinstance(code, StabilizerCode)
    assert (code.n, code.k) == (5, 1)
    assert not is_lceq_css_bruteforce(code)


def test_every_public_generator_has_the_common_prefix_signature() -> None:
    for generator_class in GENERATOR_CLASSES:
        methods = [
            member
            for name, member in inspect.getmembers(generator_class, inspect.isfunction)
            if "_codes_" in name
        ]
        assert methods
        for method in methods:
            assert list(inspect.signature(method).parameters)[:3] == ["n", "k", "seed"]


@pytest.mark.parametrize(
    "method",
    [
        PEqCodePairGenerator.stabilizer_codes_permuted,
        PEqCodePairGenerator.stabilizer_codes_basis_changed,
        PEqCodePairGenerator.stabilizer_codes_with_logicals,
        LCEqCodePairGenerator.stabilizer_codes_local_clifford,
        LCEqCodePairGenerator.stabilizer_codes_with_logicals,
    ],
)
def test_seeded_stabilizer_generators_are_deterministic(method) -> None:
    first = method(5, 2, 123)
    second = method(5, 2, 123)
    assert all(np.array_equal(_matrix(a), _matrix(b)) for a, b in zip(first, second))


@pytest.mark.parametrize(
    "method",
    [
        PEqCodePairGenerator.css_codes_permuted,
        PEqCodePairGenerator.css_codes_basis_changed,
    ],
)
def test_positive_css_generators_preserve_parameters(method) -> None:
    left, right = method(6, 2, 321, rx=2)
    assert isinstance(left, CSSCode)
    assert isinstance(right, CSSCode)
    assert (left.n, left.k) == (right.n, right.k) == (6, 2)


def test_anchored_negative_has_a_permutation_certificate() -> None:
    left, right = (
        NonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection_triple_construction(
            6, 2, 7
        )
    )
    source = _projection_rank_invariant(left)
    partner = _projection_rank_invariant(right)
    assert partner not in {source, (source[1], source[0], source[2])}


def test_independent_lc_negative_has_an_lc_certificate() -> None:
    left, right = NonLCEqCodePairGenerator.stabilizer_codes_independent(5, 2, 9)
    assert isinstance(left, StabilizerCode)
    assert isinstance(right, StabilizerCode)
    assert _lc_projection_rank_invariant(left) != _lc_projection_rank_invariant(right)
