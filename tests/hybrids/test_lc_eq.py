"""Focused checks for the hybrid solution to whether two stabilizer codes are LC-equivalent."""

from __future__ import annotations

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import pytest
import src.hybrids.lc_stb as lc_stb

from benchmarks.experiments.utils import (
    RandomizeError,
    lc_equivalent_code,
    non_lc_equivalent_code,
    random_stabilizer_code,
)
from src.core.stabilizer_code import StabilizerCode
from src.hybrids.lc_stb import LOCAL_CLIFFORDS, are_lceq


def _apply_lc_witness(symplectic: np.ndarray, witness: list[str]) -> np.ndarray:
    """Apply matrix-ordered LC words to a symplectic tableau."""
    transformed = symplectic.copy()
    n = transformed.shape[1] // 2

    assert len(witness) == n
    assert all(operation in LOCAL_CLIFFORDS for operation in witness)

    for qubit, operation in enumerate(witness):
        # The strings denote matrix products, so the rightmost gate acts first.
        for gate in reversed(operation):
            if gate == "H":
                transformed[:, [qubit, qubit + n]] = transformed[:, [qubit + n, qubit]]
            elif gate == "S":
                transformed[:, qubit + n] ^= transformed[:, qubit]

    return transformed


def _assert_maps_rowspace(
    code1: StabilizerCode,
    code2: StabilizerCode,
    witness: list[str],
) -> None:
    transformed = _apply_lc_witness(code1.symplectic, witness)
    rank = mod2.rank(code1.symplectic)

    assert mod2.rank(code2.symplectic) == rank
    assert mod2.rank(np.vstack([transformed, code2.symplectic])) == rank

# ----------------------------------------------------------------------------------------------------
# are_lceq
# ----------------------------------------------------------------------------------------------------

def test_are_lceq_preserves_n() -> None:
    assert are_lceq(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_are_lceq_preserves_k() -> None:
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_lceq(code1, code2) is None


def test_hybrid_routes_small_k_to_lse(monkeypatch: pytest.MonkeyPatch) -> None:
    code = StabilizerCode(["ZIII", "IZII", "IIZI", "IIIZ"])
    assert code.k < 2
    sentinel = ["I"] * code.n
    calls = 0

    def fake_lse(*_args: object) -> list[str]:
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(lc_stb, "_lse", fake_lse)
    monkeypatch.setattr(lc_stb, "_sat", lambda *_args: pytest.fail("SAT must not run for k < 2"))

    assert are_lceq(code, code) == sentinel
    assert calls == 1


def test_hybrid_routes_large_k_to_sat(monkeypatch: pytest.MonkeyPatch) -> None:
    code = StabilizerCode(["XIII"])
    assert code.k >= 2
    sentinel = ["I"] * code.n
    calls = 0

    monkeypatch.setattr(lc_stb, "_lse", lambda *_args: pytest.fail("LSE must not run for k >= 2"))
    monkeypatch.setattr(lc_stb, "preserved_low_degree_local_invariant", lambda *_args: True)

    def fake_sat(c1: StabilizerCode, c2: StabilizerCode) -> list[str]:
        nonlocal calls
        calls += 1
        assert c1 is code and c2 is code
        return sentinel

    monkeypatch.setattr(lc_stb, "_sat", fake_sat)

    assert are_lceq(code, code) == sentinel
    assert calls == 1


@pytest.mark.parametrize(
    ("code1", "code2", "expected"),
    [
        pytest.param(
            StabilizerCode(["Z"]),
            StabilizerCode(["X"]),
            ["H"],
            id="one-qubit-z-vs-x",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XI", "IX"]),
            ["H", "H"],
            id="two-product-bases",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XX", "ZZ"]),
            None,
            id="product-vs-bell-state",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            None,
            id="weight-two-vs-weight-one-stabilizer",
        ),
    ],
)
def test_are_lceq_hardcoded_cases(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: list[str] | None,
) -> None:
    assert are_lceq(code1, code2) == expected


@pytest.mark.parametrize(
    ("n", "k", "seed"),
    [
        pytest.param(1, 0, 11, id="one-qubit-state"),
        pytest.param(4, 0, 12, id="four-qubit-state"),
        pytest.param(2, 1, 13, id="small-one-logical"),
        pytest.param(6, 1, 14, id="larger-one-logical"),
    ],
)
def test_are_lceq_lse_returns_valid_witness(n: int, k: int, seed: int) -> None:
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)

    witness = are_lceq(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)


def test_are_lceq_random_smoke() -> None:
    for n in range(1, 6):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq(code1, code2), (list, type(None)))


@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 10)])
def test_are_lceq_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)

    assert are_lceq(code1, code2) is not None

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(2, 6)])
def test_are_lceq_random_negative(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 4
    k = (2 * seed + 1) % n
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)

    try:
        code2 = non_lc_equivalent_code(code1, seed=2000 + seed)
    except RandomizeError as re:
        pytest.skip(f"Skip test random_negative: [[{n}, {k}]] (seed {seed}) - randomization error: {re}")

    assert are_lceq(code1, code2) is None
