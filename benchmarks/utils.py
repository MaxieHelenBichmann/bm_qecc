"""Small helpers for constructing benchmark inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import ldpc.mod2.mod2_numpy as mod2

from src.core.pauli import StabilizerTableau
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


def random_stabilizer_code(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    clifford_steps: int | None = None,
) -> StabilizerCode:
    """Return a seeded random-looking stabilizer code with parameters ``[[n, k]]``.

    The construction starts with ``n-k`` single-qubit Z stabilizers and applies a
    seeded random Clifford circuit to the stabilizer tableau. This is meant for
    benchmark instances, not for uniform sampling from all stabilizer codes.
    """
    if n < 1:
        msg = "n must be at least 1."
        raise ValueError(msg)
    if not 0 <= k <= n:
        msg = f"k must satisfy 0 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)

    num_stabilizers = n - k
    generators = ["I" * q + "Z" + "I" * (n - q - 1) for q in range(num_stabilizers)]
    tableau = StabilizerTableau.from_pauli_strings(generators) if generators else StabilizerTableau.empty(n)

    rng = np.random.default_rng(seed)
    steps = clifford_steps if clifford_steps is not None else 4 * n
    if steps < 0:
        msg = "clifford_steps must be non-negative."
        raise ValueError(msg)

    for _ in range(steps):
        _apply_random_clifford_gate(tableau, rng)

    return StabilizerCode(_without_phases(tableau))


def random_css_code(
    n: int,
    k: int,
    rx: int | None = None,
    seed: int | None = None,
) -> CSSCode:
    """
    Generate random CSS parity-check matrices Hx, Hz over GF(2).
    """
    if not 0 <= k <= n:
        raise ValueError("Require 0 <= k <= n.")

    total_checks = n - k

    if rx is None:
        rx = total_checks // 2
    rz = total_checks - rx

    if not 0 <= rx <= n or not 0 <= rz <= n:
        raise ValueError("Require 0 <= rx <= n and 0 <= rz <= n.")

    rng = np.random.default_rng(seed)

    if rx == 0:
        Hx = np.zeros((0, n), dtype=np.int8)
    else:
        while True:
            Hx = rng.integers(0, 2, size=(rx, n), dtype=np.int8)
            if mod2.rank(Hx) == rx:
                break

    ker_hx = mod2.nullspace(Hx)

    if hasattr(ker_hx, "toarray"):
        ker_hx = ker_hx.toarray()

    ker_hx = np.asarray(ker_hx, dtype=np.int8) % 2

    if rz > ker_hx.shape[0]:
        raise ValueError(
            f"Cannot construct {rz} independent Z checks: "
            f"ker(Hx) has dimension {ker_hx.shape[0]}."
        )

    if rz == 0:
        Hz = np.zeros((0, n), dtype=np.int8)
    else:
        while True:
            coeffs = rng.integers(0, 2, size=(rz, ker_hx.shape[0]), dtype=np.int8)
            if mod2.rank(coeffs) == rz:
                break
        Hz = np.asarray(coeffs @ ker_hx, dtype=np.int8) % 2

    return CSSCode(Hx=Hx, Hz=Hz)

def permutation_equivalent_code(code: StabilizerCode, seed: int | None = None) -> StabilizerCode:
    """Return a permuted equivalent code to the given code."""
    rng = np.random.default_rng(seed)
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    row_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    permutation = _random_permutation(code.n, seed=permutation_seed)
    base_changed_tableau = _random_tableau_row_space_base_change(code.generators, seed=row_seed)
    return StabilizerCode(_permute_tableau(base_changed_tableau, permutation), distance=code.distance)


def permutation_equivalent_css_code(code: CSSCode, seed: int | None = None) -> CSSCode:
    """Return a permuted equivalent CSS code to the given code."""
    rng = np.random.default_rng(seed)
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    x_row_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    z_row_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    permutation = _random_permutation(code.n, seed=permutation_seed)
    hx = _random_row_space_base_change(code.Hx, seed=x_row_seed)[:, permutation]
    hz = _random_row_space_base_change(code.Hz, seed=z_row_seed)[:, permutation]
    return CSSCode(
        hx,
        hz,
        distance=code.distance,
        x_distance=code.x_distance,
        z_distance=code.z_distance,
    )

def non_permutation_equivalent_css_code(code: CSSCode, seed: int | None = None) -> CSSCode:
    """Return a same-invariant CSS code certified non-equivalent by joint row-space weights."""
    rng = np.random.default_rng(seed)
    rx = _rank_binary(code.Hx)
    rz = _rank_binary(code.Hz)
    if (rx, rz) in {(0, 0), (code.n, 0), (0, code.n)}:
        raise ValueError("No non-equivalent CSS code exists with these small invariants.")
    invariant = _stabilizer_weight_enumerator(code)
    for _ in range(10_000):
        candidate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        candidate = random_css_code(code.n, code.k, rx=rx, seed=candidate_seed)
        if _stabilizer_weight_enumerator(candidate) != invariant:
            return candidate

    raise ValueError("Could not find a candidate with a different stabilizer weight enumerator.")


def non_permutation_equivalent_stabilizer_code(code: StabilizerCode, seed: int | None = None) -> StabilizerCode:
    """Return a same-[[n,k]] stabilizer code certified non-equivalent by stabilizer weights."""
    rng = np.random.default_rng(seed)
    if code.k == code.n:
        raise ValueError("No non-equivalent stabilizer code exists with these small invariants.")
    invariant = _stabilizer_weight_enumerator(code)
    for _ in range(10_000):
        candidate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        candidate = random_stabilizer_code(code.n, code.k, seed=candidate_seed)
        if _stabilizer_weight_enumerator(candidate) != invariant:
            return candidate

    raise ValueError("Could not find a candidate with a different stabilizer weight enumerator.")

def random_permuted_stabilizer_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
    clifford_steps: int | None = None,
) -> tuple[StabilizerCode, StabilizerCode]:
    """Return a seeded random code together with a permuted equivalent copy."""
    rng = np.random.default_rng(seed)
    code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    code = random_stabilizer_code(n, k, seed=code_seed, clifford_steps=clifford_steps)
    permutation = _random_permutation(n, seed=permutation_seed)
    return code, _permute_stabilizer_code(code, permutation)

def random_permuted_css_pair(
    n: int,
    k: int,
    *,
    seed: int | None = None,
) -> tuple[CSSCode, CSSCode]:
    """Return a seeded random css code together with a permuted equivalent copy."""
    rng = np.random.default_rng(seed)
    code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    permutation_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    rx = int(rng.integers(0, n - k + 1))

    code = random_css_code(n, k, rx, seed=code_seed)
    permutation = _random_permutation(n, seed=permutation_seed)
    return code, permutation_equivalent_css_code(code, permutation)

def random_non_permuted_stabilizer_pair(    
    n: int,
    k: int,
    *,
    seed: int | None = None,
) -> tuple[StabilizerCode, StabilizerCode]:
    """Return a seeded random stabilizer code together with another random stabilizer code that is guaranteed to be non-permuted."""
    rng = np.random.default_rng(seed)
    code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    other_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    code = random_stabilizer_code(n, k, seed=code_seed)
    return code, non_permutation_equivalent_stabilizer_code(code, seed=other_seed)

def lc_equivalent_code(
    code: StabilizerCode,
    seed: int | None = None,
    *,
    row_steps: int | None = None,
) -> StabilizerCode:
    """Return an LC-equivalent code with randomized stabilizer generators."""
    rng = np.random.default_rng(seed)
    tableau = code.generators.copy()
    local_cliffords = [str(rng.choice(_LOCAL_CLIFFORDS)) for _ in range(tableau.n)]

    if tableau.n > 0 and all(local_clifford == "I" for local_clifford in local_cliffords):
        local_cliffords[int(rng.integers(0, tableau.n))] = str(rng.choice(_LOCAL_CLIFFORDS[1:]))

    for qubit, local_clifford in enumerate(local_cliffords):
        _apply_local_clifford(tableau, local_clifford, qubit)

    base_changed_tableau = _random_tableau_row_space_base_change(tableau, rng=rng, steps=row_steps)
    return StabilizerCode(base_changed_tableau, distance=code.distance)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------

_LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")


def _random_permutation(n: int, *, seed: int | None = None) -> tuple[int, ...]:
    """Return a seeded random permutation of ``range(n)``."""
    if n < 0:
        msg = "n must be non-negative."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    return tuple(int(q) for q in rng.permutation(n))


def _rank_binary(matrix: np.ndarray) -> int:
    matrix = np.asarray(matrix, dtype=np.int8) % 2
    return 0 if matrix.size == 0 or matrix.shape[0] == 0 else int(mod2.rank(matrix))


def _stabilizer_weight_enumerator(code: StabilizerCode) -> tuple[tuple[tuple[int, int, int, int], int], ...]:
    enumerator: dict[tuple[int, int, int, int], int] = {}
    for word in _row_space_words(code.symplectic):
        x_word, z_word = word[: code.n], word[code.n :]
        both = int(np.count_nonzero(x_word & z_word))
        x_only = int(np.count_nonzero(x_word)) - both
        z_only = int(np.count_nonzero(z_word)) - both
        key = (code.n - x_only - z_only - both, x_only, z_only, both)
        enumerator[key] = enumerator.get(key, 0) + 1
    return tuple(sorted(enumerator.items()))


def _row_space_words(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    rank = _rank_binary(matrix)
    if rank == 0:
        return np.zeros((1, matrix.shape[1]), dtype=np.uint8)
    basis = np.asarray(mod2.row_basis(matrix), dtype=np.uint8) % 2
    words = np.zeros((1 << rank, matrix.shape[1]), dtype=np.uint8)
    num_words = 1
    for row in basis:
        words[num_words : 2 * num_words] = words[:num_words] ^ row
        num_words *= 2
    return words


def _random_row_space_base_change(
    matrix: np.ndarray,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    steps: int | None = None,
) -> np.ndarray:
    """Return ``matrix`` after seeded invertible row operations over GF(2).

    This keeps the generated row space unchanged, but may replace the rows by
    different linear combinations. For stabilizer generator matrices, this gives
    an equivalent generating set for the same stabilizer group.
    """
    changed = np.asarray(matrix, dtype=np.int8).copy() % 2
    if changed.ndim != 2:
        msg = f"Expected a 2D matrix, got shape {changed.shape}."
        raise ValueError(msg)

    n_rows = changed.shape[0]
    num_steps = steps if steps is not None else 4 * n_rows
    if num_steps < 0:
        msg = "steps must be non-negative."
        raise ValueError(msg)

    if n_rows < 2:
        return changed

    if rng is None:
        rng = np.random.default_rng(seed)

    for _ in range(num_steps):
        row_a, row_b = rng.choice(n_rows, size=2, replace=False)
        row_a = int(row_a)
        row_b = int(row_b)
        if bool(rng.integers(0, 2)):
            changed[[row_a, row_b]] = changed[[row_b, row_a]]
        else:
            changed[row_b] ^= changed[row_a]

    return changed


def _random_tableau_row_space_base_change(
    tableau: StabilizerTableau,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    steps: int | None = None,
) -> StabilizerTableau:
    """Return ``tableau`` after seeded invertible row operations on its symplectic rows."""
    if rng is None:
        rng = np.random.default_rng(seed)

    changed = _random_row_space_base_change(tableau.tableau.matrix, rng=rng, steps=steps)
    return StabilizerTableau(changed)


def _without_phases(tableau: StabilizerTableau) -> StabilizerTableau:
    """Return a copy of ``tableau`` with zero phases."""
    return StabilizerTableau(tableau.tableau.matrix.copy())


def _permute_tableau(tableau: StabilizerTableau, permutation: Sequence[int]) -> StabilizerTableau:
    """Return a copy of ``tableau`` with physical qubits permuted."""
    permutation = _checked_permutation(tableau.n, permutation)
    columns = list(permutation) + [q + tableau.n for q in permutation]
    return StabilizerTableau(tableau.tableau.matrix[:, columns].copy())


def _permute_stabilizer_code(code: StabilizerCode, permutation: Sequence[int]) -> StabilizerCode:
    """Return a copy of ``code`` with physical qubits permuted."""
    return StabilizerCode(_permute_tableau(code.generators, permutation), distance=code.distance)


def _apply_local_clifford(tableau: StabilizerTableau, local_clifford: str, qubit: int) -> None:
    """Apply one single-qubit Clifford representative to ``qubit`` in-place."""
    if local_clifford not in _LOCAL_CLIFFORDS:
        msg = f"Unknown local Clifford {local_clifford!r}."
        raise ValueError(msg)

    for gate in reversed(local_clifford):
        if gate == "H":
            tableau.apply_h(qubit)
        elif gate == "S":
            tableau.apply_s(qubit)


def _apply_random_clifford_gate(tableau: StabilizerTableau, rng: np.random.Generator) -> None:
    """Apply one seeded random Clifford generator to a tableau in-place."""
    if tableau.n == 1:
        gate = rng.choice(("h", "s"))
    else:
        gate = rng.choice(("h", "s", "cx"))

    if gate == "h":
        tableau.apply_h(int(rng.integers(0, tableau.n)))
    elif gate == "s":
        tableau.apply_s(int(rng.integers(0, tableau.n)))
    else:
        ctrl, target = rng.choice(tableau.n, size=2, replace=False)
        tableau.apply_cx(int(ctrl), int(target))


def _checked_permutation(n: int, permutation: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a physical-qubit permutation."""
    normalized = tuple(int(q) for q in permutation)
    if sorted(normalized) != list(range(n)):
        msg = f"Expected a permutation of 0..{n - 1}, got {list(permutation)}."
        raise ValueError(msg)
    return normalized
