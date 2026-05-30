"""Focused checks for the permutation-invariant checks for whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import (
    random_non_permuted_stabilizer_pair,
    random_permuted_stabilizer_pair,
)
from src.invariants.p_eq.pm_invariants import (
    preserved_k,
    preserved_linear_dependencies,
    preserved_n,
    preserved_number_duplicate_columns,
    preserved_number_zero_columns,
    preserved_pauli_weight_enumerator,
    preserved_rank,
    preserved_weight_enumerator,
)
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode


INVARIANTS = [
    preserved_n,
    preserved_k,
    preserved_rank,
    preserved_number_zero_columns,
    preserved_number_duplicate_columns,
    preserved_weight_enumerator,
    preserved_pauli_weight_enumerator,
    preserved_linear_dependencies,
]


def _code_from_matrix(matrix: np.ndarray) -> StabilizerCode:
    matrix = np.asarray(matrix, dtype=np.int8)
    phases = np.zeros(matrix.shape[0], dtype=np.int8)
    return StabilizerCode(StabilizerTableau(matrix, phases))


def test_invariants_are_preserved_by_qubit_permutation() -> None:
    code, permuted_code = random_permuted_stabilizer_pair(5, 2, seed=0, clifford_steps=4)

    assert all(invariant(code, permuted_code) for invariant in INVARIANTS)


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_some_invariant_rejects_generated_non_permuted_pair(seed: int) -> None:
    code, non_permuted_code = random_non_permuted_stabilizer_pair(5, 2, seed=seed, clifford_steps=4)

    assert not all(invariant(code, non_permuted_code) for invariant in INVARIANTS)


def test_number_zero_columns_counts_complete_zero_columns() -> None:
    assert not preserved_number_zero_columns(
        StabilizerCode(["XI"]),
        StabilizerCode(["XX"]),
    )


def test_number_duplicate_columns_compares_multiplicity_multiset() -> None:
    assert not preserved_number_duplicate_columns(
        StabilizerCode(["XI"]),
        StabilizerCode(["XX"]),
    )


def test_weight_enumerators_ignore_redundant_generators() -> None:
    base_code = StabilizerCode(["XI", "IX"])
    redundant_generator = base_code.symplectic[0] ^ base_code.symplectic[1]
    redundant_code = _code_from_matrix(np.vstack([base_code.symplectic, redundant_generator]))

    assert preserved_weight_enumerator(base_code, redundant_code)
    assert preserved_pauli_weight_enumerator(base_code, redundant_code)
