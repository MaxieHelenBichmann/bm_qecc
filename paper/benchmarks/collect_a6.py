"""Collect the additional PM-STB SAT measurements on CSS inputs.

The script uses the same positive and negative CSS populations as the ordinary 
PM-CSS random suite and appends statistics to
``paper/data/collected/pm_stb_sat_on_css.csv``.
The A6 experiment combines this file with the normal ``pm_stb_sat.csv`` and
``pm_css_sat.csv`` files. This collector appends one summary row per seeded
batch and does not skip completed keys; re-running recomputes every batch, and
extraction keeps the latest row per ``(algorithm, n, k, positive, seed)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from benchmarks.experiments.generators_random import (
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import BenchmarkCase, run_statistics
from benchmarks.thesis.thesis_prototypes import DecisionAlgorithm, measurement_dimensions
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat

ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 42
NUM_SEEDS = 10
TIMEOUT_SECONDS = 5_400.0
CERTIFICATION_TIMEOUT_SECONDS = 600.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
CSS_SAT_MAX_R = 9
CSS_MATROID_MAX_N = 28
VERBOSE = True
N_RANGE = (3, 47)
OUTPUT_FILE = ROOT / "paper" / "data" / "collected" / "pm_stb_sat_on_css.csv"


def _attempt_seed(n: int, k: int, seed: int, attempt: int) -> int:
    population = "pm_css_negative_matching=False"
    value = f"{population}|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _css_certifier(n: int, k: int) -> Callable[..., bool] | None:
    if n - k <= CSS_SAT_MAX_R:
        return are_peq_css_sat
    if n <= CSS_MATROID_MAX_N:
        return are_peq_css_matroid
    return None


def _certified_negative_pair(
    n: int,
    k: int,
    seed: int,
    *,
    max_attempts: int = 1_000,
) -> tuple[Any, Any]:
    certifier = _css_certifier(n, k)
    use_fallback = certifier is None
    for attempt in range(max_attempts):
        attempt_seed = _attempt_seed(n, k, seed, attempt)
        if use_fallback:
            return NonPEqCodePairGenerator.css_codes_cascaded(n, k, attempt_seed)

        assert certifier is not None
        rx = attempt_seed % (n - k + 1)
        pair = NonPEqCodePairGenerator.css_codes_independent_candidate(
            n, k, attempt_seed, rx=rx
        )
        result = run(
            certifier,
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
        if result.result is False:
            return pair
    raise RuntimeError(
        f"could not generate a certified pm_css negative for [[{n},{k}]], "
        f"seed {seed}"
    )


@dataclass(frozen=True)
class CSSCaseGenerator:
    n: int
    k: int
    positive: bool

    seed_upper_bound = 1_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "negative"
        return f"random_pm_stb_sat_on_css_{self.n}_{self.k}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "pm_stb_sat_on_css",
            "generator": self.__name__,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        pair = (
            PEqCodePairGenerator.css_codes_basis_changed(self.n, self.k, seed)
            if self.positive
            else _certified_negative_pair(self.n, self.k, seed)
        )
        return BenchmarkCase(tuple(pair), self.positive, self.metadata)


def collect() -> None:
    nmin, nmax = N_RANGE
    algorithm = DecisionAlgorithm("pm_stb_sat_on_css", are_peq_stab_sat)
    for n, k in measurement_dimensions(nmin, nmax):
        for positive in (True, False):
            if VERBOSE:
                label = "positive" if positive else "negative"
                print(f"pm_stb_sat_on_css [[{n},{k}]] {label}", flush=True)
            run_statistics(
                algorithm,
                CSSCaseGenerator(n, k, positive),
                MASTER_SEED,
                NUM_SEEDS,
                OUTPUT_FILE,
                timeout=TIMEOUT_SECONDS,
                max_memory_bytes=MEMORY_LIMIT_BYTES,
                verbose=VERBOSE,
            )


if __name__ == "__main__":
    collect()
