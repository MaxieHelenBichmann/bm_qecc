"""Collect mean runtimes and deciding components of the diagnostic hybrids.

Usage::

    python3 -m paper.benchmarks.collect_a7

There is no command line: every run measures all three hybrids on every
compatible named structured code. The three diagnostic hybrids in
``paper/hybrids`` (``pm_stb``, ``pm_css``, ``lc_stb``) return a
``(decision, component)`` pair and print the tag of every stage they enter, so a
run yields two things the plain algorithm collector cannot express: which
component actually decided a case, and, for a case that never finishes, which
component it was stuck in. Every call is supervised by
``benchmarks/experiments/run.py`` exactly like the other collectors; what is
added here is only the callable it runs, a :class:`TracedHybrid` wrapper that
redirects standard output line-buffered into a log file inside the supervised
process, so the trace survives the kill that ends a timed-out call.

Component tags: ``CI`` cheap invariants, ``EI`` expensive invariants,
``S`` signatures, and the decision procedures ``BF`` (brute force), ``MI``
(matroid isomorphism), ``GI`` (graph isomorphism), ``SAT``, and ``LSE``.

Two files are written per hybrid, both append-only, under
``paper/data/collected/hybrids/``:

``<algorithm>.csv``
    One row per (named code, label) cell: mean/stddev/maximum runtime over the
    completed instances, the distribution of deciding components, and the
    distribution of stages that timed-out instances were stuck in.
``<algorithm>_instances.csv``
    One row per instance: its runtime, status, deciding component, and the full
    stage trace it printed.

``mean_seconds`` averages only instances that ran to completion; instances that
timed out, exceeded memory, or raised are excluded and counted separately.
``mean_seconds_capped`` is the alternative convention used by
``benchmarks/experiments/statistics.py``: completed instances plus timed-out
ones contributing their capped runtime.

Edit ``HYBRID_N_RANGES`` and the constants below to change which named codes a
hybrid sees, the master seed, instances per cell, timeout, memory limit, or
verbosity. Every instance row is persisted as soon as its call returns and the
summary row as soon as its cell finishes, so a crash loses at most the instance
that was running. Restarting skips cells already present in the summary CSV and
reuses individual instances already present in the instance CSV, so an
interrupted server run continues without duplicating work.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from benchmarks.experiments.generators_structured import (
    LCEqCodePairGenerator,
    NonLCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
    load_named_code,
    named_code_names,
)
from benchmarks.experiments.run import RunResult, run
from benchmarks.experiments.statistics import deterministic_seeds
from paper.hybrids.lc_stb import are_lceq
from paper.hybrids.pm_css import are_peq_css
from paper.hybrids.pm_stb import are_peq_stab
from src.core.css_code import CSSCode

ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 42
NUM_SEEDS = 10
SEED_UPPER_BOUND = 1_000
TIMEOUT_SECONDS = 5_400.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
VERBOSE = True
OUTPUT_DIRECTORY = ROOT / "paper" / "data" / "collected" / "hybrids"

#: Stage tags the hybrids print on entry, in pipeline order.
COMPONENTS = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE")
#: Placeholder for a case decided before the first instrumented stage.
TRIVIAL = "trivial"
#: Placeholder for a case killed before the first instrumented stage.
UNREACHED = "start"
#: Prefix under which the traced call records the component that decided.
DECIDED_MARKER = "#decided_by "


def execution_status(result: RunResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.memory_exceeded:
        return "memory_limited"
    if result.error is not None:
        return "error"
    return "success"


def csv_key(*values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def completed_csv_keys(
    path: Path,
    key_fields: Sequence[str],
) -> set[tuple[str, ...]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(key_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} has an incompatible header; missing {sorted(missing)}"
            )
        return {
            tuple(row[field] for field in key_fields)
            for row in reader
            if all(row.get(field) is not None for field in key_fields)
        }


def append_csv_row(
    path: Path,
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@dataclass(frozen=True)
class Hybrid:
    """One diagnostic hybrid and the populations it is measured on."""

    name: str
    function: Callable[..., tuple[bool, str]]
    css_only: bool


HYBRIDS: dict[str, Hybrid] = {
    hybrid.name: hybrid
    for hybrid in (
        Hybrid("pm_stb_hybrid", are_peq_stab, css_only=False),
        Hybrid("pm_css_hybrid", are_peq_css, css_only=True),
        Hybrid("lc_stb_hybrid", are_lceq, css_only=False),
    )
}

# Inclusive block-length ranges, intentionally centralized so server runs can be
# tuned by editing one table without adding more command-line configuration.
HYBRID_N_RANGES: dict[str, tuple[int, int]] = {
    "pm_stb_hybrid": (2, 144),
    "pm_css_hybrid": (2, 144),
    "lc_stb_hybrid": (2, 144),
}

SUMMARY_FIELDS = (
    "algorithm",
    "name",
    "n",
    "k",
    "positive",
    "seed",
    "nr_seeds",
    "timeout_seconds",
    "memory_limit_bytes",
    "num_cases",
    "num_completed",
    "num_expected",
    "num_unexpected",
    "num_timeouts",
    "num_memory_limited",
    "num_errors",
    "num_generation_errors",
    "mean_seconds",
    "stddev_seconds",
    "maximum_seconds",
    "mean_seconds_capped",
    "deciders",
    "stuck_at",
)
SUMMARY_KEY_FIELDS = ("algorithm", "name", "positive")
INSTANCE_FIELDS = (
    "algorithm",
    "name",
    "n",
    "k",
    "positive",
    "seed",
    "status",
    "runtime_seconds",
    "decided_by",
    "stuck_at",
    "decision",
    "trace",
    "error",
)


@dataclass(frozen=True)
class InstanceResult:
    """Outcome of one supervised hybrid call, including its printed trace."""

    seed: int
    runtime: float
    status: str
    decided_by: str
    decision: bool | None
    trace: tuple[str, ...]
    error: str | None

    @property
    def completed(self) -> bool:
        return self.status in {"success", "unexpected"}

    @property
    def stuck_at(self) -> str:
        if self.status not in {"timeout", "memory_limited"}:
            return ""
        return self.trace[-1] if self.trace else UNREACHED


def validate_configuration() -> None:
    missing = set(HYBRID_N_RANGES) - set(HYBRIDS)
    if missing:
        raise ValueError(f"unknown hybrids in HYBRID_N_RANGES: {sorted(missing)}")
    for name, (nmin, nmax) in HYBRID_N_RANGES.items():
        if nmin < 1 or nmax < nmin:
            raise ValueError(f"invalid n range for {name}: {(nmin, nmax)}")
    if NUM_SEEDS <= 0 or TIMEOUT_SECONDS <= 0 or MEMORY_LIMIT_BYTES <= 0:
        raise ValueError("seed count and resource limits must be positive")


# ----------------------------------------------------------------------------------------------------
# case generation
# ----------------------------------------------------------------------------------------------------

def generate_inputs(
    algorithm_name: str, code_name: str, positive: bool, seed: int
) -> tuple[Any, ...]:
    """Return one labeled presentation of a named code for the given hybrid."""
    if algorithm_name == "pm_stb_hybrid":
        return (
            PEqCodePairGenerator.stabilizer_codes_basis_changed(code_name, seed)
            if positive
            else NonPEqCodePairGenerator.stabilizer_codes_x_z_rank_projection(
                code_name, seed
            )
        )
    if algorithm_name == "pm_css_hybrid":
        return (
            PEqCodePairGenerator.css_codes_basis_changed(code_name, seed)
            if positive
            else NonPEqCodePairGenerator.css_codes_cascaded(code_name, seed)
        )
    if algorithm_name == "lc_stb_hybrid":
        return (
            LCEqCodePairGenerator.stabilizer_codes_local_clifford(code_name, seed)
            if positive
            else NonLCEqCodePairGenerator.stabilizer_codes_independent(code_name, seed)
        )
    raise ValueError(f"Unknown hybrid: {algorithm_name}")


def selected_codes(algorithm_name: str) -> list[tuple[str, Any]]:
    """Return the named codes a hybrid is measured on, in registry order."""
    hybrid = HYBRIDS[algorithm_name]
    nmin, nmax = HYBRID_N_RANGES[algorithm_name]
    codes = []
    for name in named_code_names():
        code = load_named_code(name)
        if hybrid.css_only and not isinstance(code, CSSCode):
            continue
        if nmin <= code.n <= nmax:
            codes.append((name, code))
    return codes


# ----------------------------------------------------------------------------------------------------
# supervised execution with captured stage traces
# ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TracedHybrid:
    """A hybrid that streams its stage tags to a log file while it runs.

    Wrapping the call instead of the runner keeps every supervision concern in
    :func:`benchmarks.experiments.run.run`: this object is simply the callable
    that ``run`` executes, so the redirection happens inside the very process
    ``run`` forks, isolates, limits, and kills. Line buffering plus a duplicated
    file descriptor means each tag is already on disk when a timed-out call is
    killed, and the component that decided is appended as a final marker line so
    the return value stays the plain Boolean ``run`` compares against.
    """

    function: Callable[..., Any]
    log_path: str

    def __call__(self, *inputs: Any) -> bool | None:
        log = open(self.log_path, "w", buffering=1, encoding="utf-8")
        saved_stdout, saved_fd = sys.stdout, os.dup(1)
        try:
            os.dup2(log.fileno(), 1)
            sys.stdout = log
            decision, component = _split_decision(self.function(*inputs))
            print(f"{DECIDED_MARKER}{component}")
            return decision
        finally:
            sys.stdout = saved_stdout
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            log.close()


def _split_decision(payload: Any) -> tuple[bool | None, str]:
    """Split a hybrid's ``(decision, component)`` return value."""
    if isinstance(payload, tuple) and len(payload) == 2:
        decision, component = payload
        return bool(decision), (str(component) or TRIVIAL)
    return (bool(payload) if payload is not None else None), ""


def _read_log(log_path: Path) -> tuple[tuple[str, ...], str]:
    """Return the stage tags a call reached and the component that decided.

    A killed call never writes its marker line, so the component is empty and
    the last stage tag is the one it was stuck in.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (), ""
    trace: list[str] = []
    decided_by = ""
    for line in text.splitlines():
        line = line.strip()
        if line in COMPONENTS:
            trace.append(line)
        elif line.startswith(DECIDED_MARKER):
            decided_by = line[len(DECIDED_MARKER):]
    return tuple(trace), decided_by


def run_instance(
    function: Callable[..., Any],
    inputs: tuple[Any, ...],
    expected: bool,
    seed: int,
    log_path: Path,
    *,
    timeout: float | None,
    max_memory_bytes: int | None,
) -> InstanceResult:
    """Run one hybrid call under ``run``'s supervision, keeping its stage trace."""
    result = run(
        TracedHybrid(function, str(log_path)),
        inputs,
        expected,
        timeout=timeout,
        max_memory_bytes=max_memory_bytes,
    )
    trace, decided_by = _read_log(log_path)
    status = execution_status(result)
    if status == "success" and not result.result_is_expected:
        status = "unexpected"
    return InstanceResult(
        seed=seed,
        runtime=result.runtime,
        status=status,
        decided_by=decided_by,
        decision=result.result if isinstance(result.result, bool) else None,
        trace=trace,
        error=result.error,
    )


# ----------------------------------------------------------------------------------------------------
# collection
# ----------------------------------------------------------------------------------------------------

def _distribution(values: Sequence[str]) -> str:
    """Encode a tag distribution as ``TAG:count`` pairs, most frequent first."""
    counts = Counter(value for value in values if value)
    return ";".join(
        f"{tag}:{count}"
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _number(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.9f}"


def summary_row(
    algorithm_name: str,
    code_name: str,
    code: Any,
    positive: bool,
    results: Sequence[InstanceResult],
) -> dict[str, Any]:
    """Aggregate one (named code, label) cell into a single summary row."""
    completed = [result.runtime for result in results if result.completed]
    capped = completed + [
        result.runtime for result in results if result.status == "timeout"
    ]
    return {
        "algorithm": algorithm_name,
        "name": code_name,
        "n": code.n,
        "k": code.k,
        "positive": positive,
        "seed": MASTER_SEED,
        "nr_seeds": NUM_SEEDS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "num_cases": sum(result.status != "generation_error" for result in results),
        "num_completed": len(completed),
        "num_expected": sum(result.status == "success" for result in results),
        "num_unexpected": sum(result.status == "unexpected" for result in results),
        "num_timeouts": sum(result.status == "timeout" for result in results),
        "num_memory_limited": sum(
            result.status == "memory_limited" for result in results
        ),
        "num_errors": sum(result.status == "error" for result in results),
        "num_generation_errors": sum(
            result.status == "generation_error" for result in results
        ),
        "mean_seconds": _number(mean(completed)) if completed else "",
        "stddev_seconds": _number(stdev(completed)) if len(completed) > 1 else "0.0",
        "maximum_seconds": _number(max(completed)) if completed else "",
        "mean_seconds_capped": _number(mean(capped)) if capped else "",
        "deciders": _distribution([result.decided_by for result in results]),
        "stuck_at": _distribution([result.stuck_at for result in results]),
    }


def instance_row(
    algorithm_name: str,
    code_name: str,
    code: Any,
    positive: bool,
    result: InstanceResult,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm_name,
        "name": code_name,
        "n": code.n,
        "k": code.k,
        "positive": positive,
        "seed": result.seed,
        "status": result.status,
        "runtime_seconds": _number(result.runtime),
        "decided_by": result.decided_by,
        "stuck_at": result.stuck_at,
        "decision": "" if result.decision is None else result.decision,
        "trace": ">".join(result.trace),
        "error": result.error or "",
    }


def stored_instance_results(
    path: Path,
    algorithm_name: str,
    code_name: str,
    positive: bool,
) -> list[InstanceResult]:
    """Load already persisted instances for one resumable cell."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    by_seed: dict[int, InstanceResult] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["algorithm"] != algorithm_name
                or row["name"] != code_name
                or row["positive"] != str(positive)
            ):
                continue
            decision_text = row["decision"].strip().lower()
            decision = (
                None
                if not decision_text
                else decision_text in {"true", "1", "yes"}
            )
            result = InstanceResult(
                seed=int(row["seed"]),
                runtime=float(row["runtime_seconds"] or 0.0),
                status=row["status"],
                decided_by=row["decided_by"],
                decision=decision,
                trace=tuple(tag for tag in row["trace"].split(">") if tag),
                error=row["error"] or None,
            )
            by_seed[result.seed] = result
    return list(by_seed.values())


def collect_cell(
    algorithm_name: str,
    code_name: str,
    code: Any,
    positive: bool,
    log_directory: Path,
    seeds: Sequence[int],
    persist: Callable[[InstanceResult], None],
) -> list[InstanceResult]:
    """Run every seeded instance of one (named code, label) cell."""
    hybrid = HYBRIDS[algorithm_name]
    results: list[InstanceResult] = []

    for seed in seeds:
        try:
            inputs = generate_inputs(algorithm_name, code_name, positive, seed)
        except Exception as exc:  # noqa: BLE001 - generation errors are statistics
            result = InstanceResult(
                seed=seed,
                runtime=0.0,
                status="generation_error",
                decided_by="",
                decision=None,
                trace=(),
                error=f"{type(exc).__name__}: {exc}",
            )
            results.append(result)
            persist(result)
            if VERBOSE:
                print(
                    f"        seed={seed}: {result.error}",
                    flush=True,
                )
            continue

        log_path = log_directory / f"{algorithm_name}_{code_name}_{positive}_{seed}.log"
        result = run_instance(
            hybrid.function,
            tuple(inputs),
            positive,
            seed,
            log_path,
            timeout=TIMEOUT_SECONDS,
            max_memory_bytes=MEMORY_LIMIT_BYTES,
        )
        log_path.unlink(missing_ok=True)
        results.append(result)
        persist(result)
        if VERBOSE:
            detail = result.decided_by or result.stuck_at or result.error or ""
            print(
                f"        seed={seed}: {result.status} "
                f"{result.runtime:.3f}s [{'>'.join(result.trace)}] {detail}",
                flush=True,
            )
    return results


def _persister(
    instance_file: Path,
    algorithm_name: str,
    code_name: str,
    code: Any,
    positive: bool,
) -> Callable[[InstanceResult], None]:
    """Return a sink that appends one finished instance to its CSV immediately."""

    def persist(result: InstanceResult) -> None:
        append_csv_row(
            instance_file,
            instance_row(algorithm_name, code_name, code, positive, result),
            INSTANCE_FIELDS,
        )

    return persist


def collect() -> None:
    """Append runtime and component statistics for every hybrid and named code."""
    validate_configuration()
    with tempfile.TemporaryDirectory(prefix="hybrid_traces_") as directory:
        log_directory = Path(directory)
        for algorithm_name in HYBRIDS:
            summary_file = OUTPUT_DIRECTORY / f"{algorithm_name}.csv"
            instance_file = OUTPUT_DIRECTORY / f"{algorithm_name}_instances.csv"
            done = completed_csv_keys(summary_file, SUMMARY_KEY_FIELDS)
            codes = selected_codes(algorithm_name)
            if VERBOSE:
                print(
                    f"{algorithm_name}: {len(codes)} named codes, "
                    f"{NUM_SEEDS} instances/cell -> {summary_file}",
                    flush=True,
                )
            for code_name, code in codes:
                for positive in (True, False):
                    label = "positive" if positive else "negative"
                    if csv_key(algorithm_name, code_name, positive) in done:
                        if VERBOSE:
                            print(f"    {code_name} {label}: already collected")
                        continue
                    if VERBOSE:
                        print(f"    {code_name} {label}", flush=True)
                    seeds = deterministic_seeds(
                        MASTER_SEED,
                        NUM_SEEDS,
                        upper_bound=SEED_UPPER_BOUND,
                    )
                    stored = stored_instance_results(
                        instance_file, algorithm_name, code_name, positive
                    )
                    stored_seeds = {result.seed for result in stored}

                    new_results = collect_cell(
                        algorithm_name,
                        code_name,
                        code,
                        positive,
                        log_directory,
                        [seed for seed in seeds if seed not in stored_seeds],
                        _persister(
                            instance_file, algorithm_name, code_name, code, positive
                        ),
                    )
                    append_csv_row(
                        summary_file,
                        summary_row(
                            algorithm_name,
                            code_name,
                            code,
                            positive,
                            [*stored, *new_results],
                        ),
                        SUMMARY_FIELDS,
                    )


def main() -> int:
    collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
