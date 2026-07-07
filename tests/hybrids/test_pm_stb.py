"""Focused checks for the hybrid solution to whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

import pytest

from benchmarks.utils import RandomizeError, random_non_permuted_stabilizer_pair, random_permuted_stabilizer_pair
from src.core.stabilizer_code import StabilizerCode
from src.hybrids.p_stab import are_peq_stab

# ----------------------------------------------------------------------------------------------------
# are_peq_stab
# ----------------------------------------------------------------------------------------------------

def test_are_peq_stab_preserves_n() -> None:
    assert are_peq_stab(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_are_peq_stab_preserves_k() -> None:
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_peq_stab(code1, code2) is None


def test_are_peq_stab_does_not_swap_x_and_z() -> None:
    code1 = StabilizerCode(["XII"])
    code2 = StabilizerCode(["ZII"])

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_peq_stab(code1, code2) is None


def test_are_peq_stab_hardcoded_positive() -> None:
    code1 = StabilizerCode(["XXII", "IIZZ"])
    code2 = StabilizerCode(["XIXI", "IZIZ"])

    assert are_peq_stab(code1, code2) is not None


def test_are_peq_stab_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_stab(code1, code2), (list, type(None)))
            except RandomizeError:
                pass


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab(code1, code2) is not None


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab(code1, code2) is None
