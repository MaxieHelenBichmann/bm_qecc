"""Collect the additional PM-STB SAT measurements on CSS inputs.

Usage::

    python3 -m paper.benchmarks.collect_css_sat_encoding

There are no CLI arguments. Edit the constants below to change the inclusive
``N_RANGE``, seed schedule, timeout, memory limit, or verbosity. The script uses
the same positive and negative CSS populations as the ordinary PM-CSS random
suite and appends statistics to
``paper/data/collected/pm_stb_sat_on_css.csv``. The A6 experiment combines this
file with the normal ``pm_stb_sat.csv`` and ``pm_css_sat.csv`` files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.experiments.generators_random import (
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from benchmarks.experiments.statistics import BenchmarkCase, run_statistics
from benchmarks.thesis.thesis_prototypes import DecisionAlgorithm, measurement_dimensions
from paper.benchmarks.utils.config import COLLECTED_DATA_DIR
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat

MASTER_SEED = 42
NUM_SEEDS = 10
TIMEOUT_SECONDS = 5_400.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
VERBOSE = True
N_RANGE = (3, 47)
OUTPUT_FILE = COLLECTED_DATA_DIR / "pm_stb_sat_on_css.csv"


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
            else NonPEqCodePairGenerator.css_codes_cascaded(self.n, self.k, seed)
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
