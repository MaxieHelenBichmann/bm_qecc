"""Tests for named structured-code generators."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from benchmarks.experiments.generators_structured import (
    LCEqCodeGenerator,
    LCEqCodePairGenerator,
    NAMED_CODE_SPECS,
    NonLCEqCodeGenerator,
    NonLCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
    load_named_code,
    named_code_names,
)
from src.algorithms.lc_css.lc_css_bruteforce import is_lceq_css_bruteforce
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


GENERATOR_CLASSES = (
    PEqCodePairGenerator,
    NonPEqCodePairGenerator,
    LCEqCodePairGenerator,
    NonLCEqCodePairGenerator,
    LCEqCodeGenerator,
    NonLCEqCodeGenerator,
)


def test_named_registry_loads_fresh_codes() -> None:
    assert named_code_names() == tuple(NAMED_CODE_SPECS)
    first = load_named_code("steane")
    second = load_named_code("steane")
    assert isinstance(first, CSSCode)
    assert first is not second
    assert (first.n, first.k) == (7, 1)
    assert np.array_equal(first.symplectic, second.symplectic)


def test_named_loader_rejects_unknown_and_non_css_names() -> None:
    with pytest.raises(ValueError, match="Unknown structured code"):
        load_named_code("not_a_code")
    with pytest.raises(ValueError, match="not stored as a CSS code"):
        PEqCodePairGenerator.css_codes_basis_changed("5q_prf", 1)


def test_every_structured_generator_uses_name_and_seed_and_documents_bias() -> None:
    for generator_class in GENERATOR_CLASSES:
        methods = [
            method
            for method_name, method in inspect.getmembers(
                generator_class, inspect.isfunction
            )
            if "_code_" in method_name or "_codes_" in method_name
        ]
        assert methods
        for method in methods:
            assert list(inspect.signature(method).parameters)[:2] == ["name", "seed"]
            assert "Sampling bias:" in (method.__doc__ or "")
            assert "NOT USABLE whenever" in (method.__doc__ or "")


def test_structured_positive_pair_keeps_named_source() -> None:
    source, partner = PEqCodePairGenerator.css_codes_basis_changed("steane", 12)
    loaded = load_named_code("steane")
    assert isinstance(source, CSSCode)
    assert isinstance(partner, CSSCode)
    assert np.array_equal(source.symplectic, loaded.symplectic)
    assert (partner.n, partner.k) == (source.n, source.k)


def test_structured_lc_pair_keeps_named_source() -> None:
    source, partner = LCEqCodePairGenerator.stabilizer_codes_local_clifford(
        "5q_prf", 13
    )
    assert isinstance(source, StabilizerCode)
    assert np.array_equal(source.symplectic, load_named_code("5q_prf").symplectic)
    assert (partner.n, partner.k) == (source.n, source.k)


def test_structured_positive_lc_css_code_uses_named_css_source() -> None:
    code = LCEqCodeGenerator.stabilizer_code_local_clifford("3q_rep", 14)
    assert is_lceq_css_bruteforce(code)
