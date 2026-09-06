"""Collect runtimes and deciding stages of the paper hybrids on the named codes.

For every problem, named code, label, and seed the collector

1. generates one instance and caches it in
   ``paper/data/collected/hybrids/<problem>_instances.csv`` (an existing cached
   instance is read back instead of regenerated),
2. runs the matching paper hybrid on it under timeout/memory supervision, and
3. appends runtime, status, and the stage that decided (or, for a killed call,
   the stage it was stuck in) to ``paper/data/collected/hybrids/<problem>_raw.csv``.

Positive instances are the named code and an equivalent presentation of it:
a random qubit permutation plus generator-basis change for PM, a random local
Clifford plus generator-basis change for LC. Negative instances follow the A1
invariant-rejection benchmark: two random Clifford gates (PM-STB, LC-STB) or two
random CNOTs (PM-CSS) applied to the named code, kept only once an exact
backend certifies inequivalence (SAT; matroid isomorphism for PM-CSS with
``n - k > 9``; a sound invariant mismatch for PM-CSS with ``n > 28``). Instance
generation, including certification, runs under its own time limit.

Restarting skips ``(code, positive, seed)`` keys already present in the raw
file. Delete a raw file, or single rows of it, to remeasure; delete an instance
file to regenerate its instances.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.experiments.generators_random import _certificate as css_certificate
from benchmarks.experiments.generators_structured import (
    LCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from benchmarks.experiments.run import RunResult, run
from benchmarks.experiments.statistics import deterministic_seeds
from paper.hybrids.lc_stb import are_lceq
from paper.hybrids.pm_css import are_peq_css
from paper.hybrids.pm_stb import are_peq_stab
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = ROOT / "paper" / "data" / "collected" / "hybrids"

CODES = (
    "bell", "3q_rep", "5q_prf", "steane", "shor", "carbon", "hamming_15",
    "15q_optimal", "tetrahedral", "golay", "rot_surf_d5", "hamming_31",
    "coco_488", "coco_666", "bb_72", "bb_90", "bb_108", "bb_144",
)
NON_CSS_CODES = frozenset({"5q_prf", "15q_optimal"})
HYBRIDS: dict[str, Callable[..., tuple[bool, str]]] = {
    "pm_stb": are_peq_stab,
    "pm_css": are_peq_css,
    "lc_stb": are_lceq,
}

MASTER_SEED = 42
NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
GATE_STEPS = 2
NEGATIVE_MAX_ATTEMPTS = 1_000
CSS_SAT_MAX_R = 9
CSS_MATROID_MAX_N = 28
GENERATION_TIMEOUT_SECONDS = 900.0
TIMEOUT_SECONDS = 5_400.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
VERBOSE = True

#: Stage tags the hybrids print on entry, in pipeline order.
STAGES = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE")
TRIVIAL = "trivial"
UNREACHED = "start"
DECIDED_MARKER = "#decided_by "

KEY_FIELDS = ("code", "positive", "seed")
INSTANCE_FIELDS = (*KEY_FIELDS, "n", "k", "status", "left", "right", "error")
RAW_FIELDS = (
    "problem", *KEY_FIELDS, "n", "k", "status", "runtime_seconds",
    "decided_by", "stuck_at", "timeout_seconds", "error",
)


# CSV persistence -------------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def encode_code(code: StabilizerCode) -> str:
    """Serialize check matrices as ``css:<Hx>|<Hz>`` or ``stb:<symplectic>``."""
    def rows(matrix: np.ndarray) -> str:
        return "/".join("".join(str(int(bit)) for bit in row) for row in np.asarray(matrix) % 2)

    if isinstance(code, CSSCode):
        return f"css:{rows(code.Hx)}|{rows(code.Hz)}"
    return f"stb:{rows(code.symplectic)}"


def decode_code(text: str, n: int) -> StabilizerCode:
    def matrix(part: str, columns: int) -> np.ndarray:
        rows = [[int(bit) for bit in row] for row in part.split("/") if row]
        return np.array(rows, dtype=np.int8).reshape(-1, columns)

    kind, body = text.split(":", 1)
    if kind == "css":
        hx, hz = body.split("|")
        return CSSCode(matrix(hx, n), matrix(hz, n), n=n)
    return StabilizerCode(StabilizerTableau(matrix(body, 2 * n)), n=n)


# Instance generation ---------------------------------------------------------------------------

def certified_inequivalent(problem: str, left: StabilizerCode, right: StabilizerCode) -> bool:
    """Decide inequivalence with the exact backend policy of the A1 benchmark."""
    if problem == "pm_stb":
        return not are_peq_stab_sat(left, right)
    if problem == "lc_stb":
        return not are_lceq_sat(left, right)
    assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
    if left.n - left.k <= CSS_SAT_MAX_R:
        return not are_peq_css_sat(left, right)
    if left.n <= CSS_MATROID_MAX_N:
        return not are_peq_css_matroid(left, right)
    # Sound one-sided witness: differing permutation invariants imply inequivalence.
    return css_certificate(left) != css_certificate(right)


def generate_pair(problem: str, code_name: str, positive: bool,
                  seed: int) -> tuple[StabilizerCode, StabilizerCode]:
    if positive:
        if problem == "lc_stb":
            return LCEqCodePairGenerator.stabilizer_codes_local_clifford(code_name, seed)
        if code_name in NON_CSS_CODES:
            return PEqCodePairGenerator.stabilizer_codes_basis_changed(code_name, seed)
        # PM-STB and PM-CSS receive the identical pair for a CSS code.
        return PEqCodePairGenerator.css_codes_basis_changed(code_name, seed)

    for attempt in range(NEGATIVE_MAX_ATTEMPTS):
        attempt_seed = seed * NEGATIVE_MAX_ATTEMPTS + attempt
        code: StabilizerCode
        candidate: StabilizerCode
        if problem == "pm_css":
            code, candidate = NonPEqCodePairGenerator.css_codes_cnot_candidate(
                code_name, attempt_seed, gate_steps=GATE_STEPS
            )
        else:
            code, candidate = NonPEqCodePairGenerator.stabilizer_codes_clifford_candidate(
                code_name, attempt_seed, gate_steps=GATE_STEPS
            )
        if certified_inequivalent(problem, code, candidate):
            return code, candidate
    raise RuntimeError(f"no certified {problem} negative for {code_name}, seed {seed}")


def generate_instance(problem: str, code_name: str, positive: bool, seed: int) -> dict[str, Any]:
    """Worker entry point: returns the already serialized instance row."""
    left, right = generate_pair(problem, code_name, positive, seed)
    return {
        "code": code_name, "positive": positive, "seed": seed, "n": left.n, "k": left.k,
        "status": "success", "left": encode_code(left), "right": encode_code(right), "error": "",
    }


# Supervised hybrid execution -------------------------------------------------------------------

@dataclass(frozen=True)
class TracedHybrid:
    """Run a hybrid with its printed stage tags redirected to a line-buffered log.

    The hybrids print the tag of every stage they enter. Supervision kills a
    timed-out call, so the tags must already be on disk by then: that is the
    only way to learn which stage a timeout was stuck in.
    """

    function: Callable[..., tuple[bool, str]]
    log_path: str

    def __call__(self, left: StabilizerCode, right: StabilizerCode) -> bool:
        with open(self.log_path, "w", buffering=1, encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                decision, stage = self.function(left, right)
                print(f"{DECIDED_MARKER}{stage or TRIVIAL}")
        return bool(decision)


def read_trace(log_path: Path) -> tuple[list[str], str]:
    """Return the stages a call entered and the stage that decided (empty if killed)."""
    trace, decided_by = [], ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line in STAGES:
            trace.append(line)
        elif line.startswith(DECIDED_MARKER):
            decided_by = line[len(DECIDED_MARKER):]
    return trace, decided_by


def execution_status(result: RunResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.memory_exceeded:
        return "memory_limited"
    if result.error is not None:
        return "error"
    return "success" if result.result_is_expected else "unexpected"


def run_instance(problem: str, instance: Mapping[str, str], log_path: Path, *,
                 timeout: float, memory_limit_bytes: int) -> dict[str, Any]:
    n = int(instance["n"])
    left, right = decode_code(instance["left"], n), decode_code(instance["right"], n)
    positive = instance["positive"] == "True"
    log_path.write_text("", encoding="utf-8")
    result = run(TracedHybrid(HYBRIDS[problem], str(log_path)), (left, right), positive,
                 timeout=timeout, max_memory_bytes=memory_limit_bytes)
    status = execution_status(result)
    trace, decided_by = read_trace(log_path)
    killed = status in {"timeout", "memory_limited"}
    return {
        "problem": problem, "code": instance["code"], "positive": positive,
        "seed": instance["seed"], "n": n, "k": instance["k"], "status": status,
        "runtime_seconds": f"{result.runtime:.9f}", "decided_by": decided_by,
        "stuck_at": (trace[-1] if trace else UNREACHED) if killed else "",
        "timeout_seconds": timeout, "error": result.error or "",
    }


# Collection ------------------------------------------------------------------------------------

def collect(
    problems: Sequence[str] = tuple(HYBRIDS),
    *,
    codes: Sequence[str] = CODES,
    seeds: Sequence[int] = SEEDS,
    output_directory: Path = OUTPUT_DIRECTORY,
    generation_timeout: float = GENERATION_TIMEOUT_SECONDS,
    timeout: float = TIMEOUT_SECONDS,
    memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
    verbose: bool = VERBOSE,
) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        log_path = Path(scratch) / "trace.log"
        for problem in problems:
            instances_file = output_directory / f"{problem}_instances.csv"
            raw_file = output_directory / f"{problem}_raw.csv"
            instances = {row_key(row): row for row in read_rows(instances_file)}
            measured = {row_key(row) for row in read_rows(raw_file)}
            for code_name in codes:
                if problem == "pm_css" and code_name in NON_CSS_CODES:
                    continue
                for positive in (True, False):
                    for seed in seeds:
                        key = (code_name, str(positive), str(seed))
                        if key in measured:
                            continue
                        instance = instances.get(key)
                        if instance is None:
                            outcome = run(generate_instance, (problem, code_name, positive, seed),
                                          None, timeout=generation_timeout,
                                          max_memory_bytes=memory_limit_bytes)
                            instance = outcome.result or {
                                "code": code_name, "positive": positive, "seed": seed,
                                "n": "", "k": "", "status": "generation_error",
                                "left": "", "right": "",
                                "error": outcome.error or f"generation {execution_status(outcome)}",
                            }
                            instance = {field: str(instance[field]) for field in INSTANCE_FIELDS}
                            append_row(instances_file, instance, INSTANCE_FIELDS)
                            instances[key] = instance
                        if instance["status"] != "success":
                            row: dict[str, Any] = {
                                "problem": problem, "code": code_name, "positive": positive,
                                "seed": seed, "n": instance["n"], "k": instance["k"],
                                "status": "generation_error", "runtime_seconds": "",
                                "decided_by": "", "stuck_at": "", "timeout_seconds": timeout,
                                "error": instance["error"],
                            }
                        else:
                            row = run_instance(problem, instance, log_path, timeout=timeout,
                                               memory_limit_bytes=memory_limit_bytes)
                        append_row(raw_file, row, RAW_FIELDS)
                        measured.add(key)
                        if verbose:
                            detail = row["decided_by"] or row["stuck_at"] or row["error"]
                            runtime = f"{row['runtime_seconds']}s " if row["runtime_seconds"] else ""
                            print(f"{problem} {code_name} positive={positive} seed={seed}: "
                                  f"{row['status']} {runtime}{detail}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(HYBRIDS), action="append",
                        help="collect only this problem (repeatable); default: all")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    collect(arguments.problem or tuple(HYBRIDS))
