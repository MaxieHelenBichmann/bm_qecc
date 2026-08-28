"""Direct deterministic generation of the few code-pair populations we need."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from benchmarks.experiments.generators_random import (
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from benchmarks.experiments.run import run
from paper.benchmarks.utils.config import (
    CERTIFICATION_TIMEOUT_SECONDS,
    MEMORY_LIMIT_BYTES,
)
from paper.benchmarks.utils.invariants import evaluate_signature
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode

CodePair = tuple[StabilizerCode, StabilizerCode]
CERTIFIERS: dict[str, Callable[..., bool]] = {
    "pm_stb": are_peq_stab_sat,
    "pm_css": are_peq_css_sat,
    "lc_stb": are_lceq_sat,
}


def _attempt_seed(population: str, n: int, k: int, seed: int, attempt: int) -> int:
    value = f"{population}|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _independent_candidate(problem: str, n: int, k: int, seed: int) -> CodePair:
    if problem == "pm_css":
        return NonPEqCodePairGenerator.css_codes_independent_candidate(n, k, seed)
    return NonPEqCodePairGenerator.stabilizer_codes_independent_candidate(n, k, seed)


def _css_rank_mismatch(left: StabilizerCode, right: StabilizerCode) -> bool:
    return (
        isinstance(left, CSSCode)
        and isinstance(right, CSSCode)
        and (left.Hx.shape[0], left.Hz.shape[0])
        != (right.Hx.shape[0], right.Hz.shape[0])
    )


def _certified_inequivalent(problem: str, pair: CodePair) -> bool:
    left, right = pair
    if problem == "pm_css" and _css_rank_mismatch(left, right):
        return True
    result = run(
        CERTIFIERS[problem],
        pair,
        False,
        timeout=CERTIFICATION_TIMEOUT_SECONDS,
        max_memory_bytes=MEMORY_LIMIT_BYTES,
    )
    if result.timed_out:
        raise RuntimeError("inequivalence certification timed out")
    if result.memory_exceeded:
        raise RuntimeError("inequivalence certification exceeded memory limit")
    if result.error is not None:
        raise RuntimeError(f"inequivalence certification failed: {result.error}")
    return result.result is False


def certified_negative_pair(
    problem: str,
    n: int,
    k: int,
    seed: int,
    *,
    require_matching_signature: bool = False,
    max_attempts: int = 1_000,
) -> CodePair:
    """Return independent codes certified outside the requested orbit."""
    population = f"{problem}_negative_matching={require_matching_signature}"
    for attempt in range(max_attempts):
        pair = _independent_candidate(
            problem,
            n,
            k,
            _attempt_seed(population, n, k, seed, attempt),
        )
        if require_matching_signature and not evaluate_signature(problem, *pair)[0]:
            continue
        if _certified_inequivalent(problem, pair):
            return pair
    raise RuntimeError(
        f"could not generate a certified {problem} negative for [[{n},{k}]], seed {seed}"
    )


def signature_pair(
    problem: str,
    n: int,
    k: int,
    seed: int,
    positive: bool,
) -> CodePair:
    """Return a positive or certified-negative pair with matching signatures."""
    if not positive:
        return certified_negative_pair(
            problem,
            n,
            k,
            seed,
            require_matching_signature=True,
        )
    if problem == "pm_css":
        return PEqCodePairGenerator.css_codes_basis_changed(n, k, seed)
    return PEqCodePairGenerator.stabilizer_codes_permuted(n, k, seed)
