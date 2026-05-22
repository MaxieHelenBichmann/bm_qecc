"""Focused checks for the automorphism solution to whether two stabilizer codes are permutation-equivalent."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.utils import RandomizeError, random_permuted_stabilizer_pair, random_non_permuted_stabilizer_pair
from src.algorithms.p_stab import p_stab_aut
from src.algorithms.p_stab.p_stab_aut import are_peq_stab_aut

# ----------------------------------------------------------------------------------------------------
# are_peq_stab_aut
# ----------------------------------------------------------------------------------------------------

def test_gap_package_root_survives_module_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    moved_module = repo_root / "src" / "algorithms" / "p_stab" / "nested" / "p_stab_aut.py"

    monkeypatch.setattr(p_stab_aut, "__file__", str(moved_module))

    assert p_stab_aut._gap_package_root() == repo_root / ".gap"


def test_are_peq_stab_aut_random_smoke() -> None:
    for n in range(3, 5):
        for k in range(n + 1):
            try:
                code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k)
                assert isinstance(are_peq_stab_aut(code1, code2), bool)
            except RandomizeError:
                pass

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_aut_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_positive: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_aut(code1, code2) is True


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_peq_stab_aut_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = 1 + (2 * seed + 1) % (n - 1)

    try:
        code1, code2 = random_non_permuted_stabilizer_pair(n, k, seed=1000 + 17 * n + k + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_peq_stab_aut(code1, code2) is False
