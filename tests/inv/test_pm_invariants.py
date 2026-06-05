"""Focused checks for the permutation-invariant checks for whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils import (
    random_non_permuted_stabilizer_pair,
    random_permuted_stabilizer_pair,
)
from src.core.css_code import CSSCode
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode
from src.invariants.p_eq.pm_invariants import (
    preserved_d,
    preserved_k,
    preserved_linear_dependencies,
    preserved_n,
    preserved_number_duplicate_columns,
    preserved_number_zero_columns,
    preserved_pauli_weight_enumerator,
    preserved_rank,
    preserved_weight_enumerator,
)


INVARIANTS = [
    preserved_n,
    preserved_k,
    preserved_d,
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


def test_distance_is_preserved_when_stabilizer_constructor_generates_default_distance() -> None:
    code1 = StabilizerCode(["ZZ"])
    code2 = StabilizerCode(["XX"])

    assert code1.distance == 1
    assert code2.distance == 1
    assert preserved_d(code1, code2)


def test_distance_rejects_different_stabilizer_distances() -> None:
    assert not preserved_d(
        StabilizerCode(["ZZ"], distance=2),
        StabilizerCode(["XX"], distance=3),
    )


def test_distance_is_preserved_when_css_constructor_generates_default_distances() -> None:
    code1 = CSSCode(
        Hx=np.array([[1, 1, 0]], dtype=np.int8),
        Hz=np.array([[0, 0, 1]], dtype=np.int8),
    )
    code2 = CSSCode(
        Hx=np.array([[1, 0, 1]], dtype=np.int8),
        Hz=np.array([[0, 1, 0]], dtype=np.int8),
    )

    assert (code1.distance, code1.x_distance, code1.z_distance) == (1, 1, 1)
    assert (code2.distance, code2.x_distance, code2.z_distance) == (1, 1, 1)
    assert preserved_d(code1, code2)


def test_css_distance_compares_x_and_z_distances_separately() -> None:
    hx = np.array([[1, 1, 0]], dtype=np.int8)
    hz = np.array([[0, 0, 1]], dtype=np.int8)

    assert not preserved_d(
        CSSCode(hx, hz, distance=3, x_distance=5, z_distance=3),
        CSSCode(hx, hz, distance=3, x_distance=3, z_distance=5),
    )


def test_rank_is_preserved_by_qubit_permutation() -> None:
    assert preserved_rank(
        StabilizerCode(["XI", "IZ"]),
        StabilizerCode(["IX", "ZI"]),
    )


def test_rank_rejects_different_x_and_z_projection_ranks() -> None:
    assert not preserved_rank(
        StabilizerCode(["XI", "IX"]),
        StabilizerCode(["XX", "ZZ"]),
    )


def test_css_rank_compares_x_and_z_check_ranks_separately() -> None:
    assert not preserved_rank(
        CSSCode(Hx=np.eye(2, dtype=np.int8), Hz=None),
        CSSCode(
            Hx=np.array([[1, 1]], dtype=np.int8),
            Hz=np.array([[1, 1]], dtype=np.int8),
        ),
    )


def test_number_zero_columns_is_preserved_by_qubit_permutation() -> None:
    assert preserved_number_zero_columns(
        StabilizerCode(["XI"]),
        StabilizerCode(["IX"]),
    )


def test_number_zero_columns_rejects_different_zero_column_counts() -> None:
    assert not preserved_number_zero_columns(
        StabilizerCode(["XII"]),
        StabilizerCode(["XXX"]),
    )


def test_number_duplicate_columns_accepts_same_multiplicity_profile() -> None:
    assert preserved_number_duplicate_columns(
        StabilizerCode(["XX"]),
        StabilizerCode(["ZZ"]),
    )


def test_number_duplicate_columns_rejects_different_multiplicity_profiles() -> None:
    assert not preserved_number_duplicate_columns(
        StabilizerCode(["XXX"]),
        StabilizerCode(["XXI"]),
    )


def test_weight_enumerator_ignores_pauli_type_when_binary_weights_match() -> None:
    assert preserved_weight_enumerator(
        StabilizerCode(["XX"]),
        StabilizerCode(["ZZ"]),
    )


def test_weight_enumerator_rejects_different_binary_row_space_weights() -> None:
    assert not preserved_weight_enumerator(
        StabilizerCode(["XI"]),
        StabilizerCode(["XX"]),
    )


def test_css_weight_enumerator_compares_x_and_z_parts_separately() -> None:
    assert not preserved_weight_enumerator(
        CSSCode(
            Hx=np.array([[1, 0, 0]], dtype=np.int8),
            Hz=np.array([[0, 1, 1]], dtype=np.int8),
        ),
        CSSCode(
            Hx=np.array([[1, 1, 0]], dtype=np.int8),
            Hz=np.array([[0, 0, 1]], dtype=np.int8),
        ),
    )


def test_pauli_weight_enumerator_is_preserved_by_qubit_permutation() -> None:
    assert preserved_pauli_weight_enumerator(
        StabilizerCode(["XYI"]),
        StabilizerCode(["IXY"]),
    )


def test_pauli_weight_enumerator_rejects_different_pauli_type_counts() -> None:
    assert not preserved_pauli_weight_enumerator(
        StabilizerCode(["XX"]),
        StabilizerCode(["ZZ"]),
    )


def test_linear_dependencies_are_preserved_by_qubit_permutation() -> None:
    assert preserved_linear_dependencies(
        StabilizerCode(["XII"]),
        StabilizerCode(["IXI"]),
    )


def test_linear_dependencies_reject_different_single_qubit_column_ranks() -> None:
    assert not preserved_linear_dependencies(
        StabilizerCode(["XII"]),
        StabilizerCode(["XXI"]),
    )


def test_weight_enumerators_ignore_redundant_generators() -> None:
    base_code = StabilizerCode(["XI", "IX"])
    redundant_generator = base_code.symplectic[0] ^ base_code.symplectic[1]
    redundant_code = _code_from_matrix(np.vstack([base_code.symplectic, redundant_generator]))

    assert preserved_weight_enumerator(base_code, redundant_code)
    assert preserved_pauli_weight_enumerator(base_code, redundant_code)
