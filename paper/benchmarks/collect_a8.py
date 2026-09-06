"""Collect the resumable A8 hybrid campaign on named-code perturbations.

The A8 target population has two deliberately separate strata:

* ``positive_control``: a named code with a reproducibly randomized equivalent
  presentation. PM-STB and PM-CSS share the exact same applicable CSS pair.
* ``certified_negative``: seeded short-circuit proposals obtained from the
  named code and retained only after the same independent certification used
  by the invariant-rejection benchmark. Large PM-CSS instances keep the same
  fixed-depth CNOT proposal and use a scalable invariant as a one-sided
  inequivalence witness.

Every proposal and certification attempt is persisted. The retained negative
population is therefore conditional on certification and is not uniform over
codes, circuits, or equivalence classes. Named-code selection, fixed depth,
backend feasibility thresholds, and certificate-dependent missingness remain
sources of bias.

Positive labels are witnessed by the recorded transformations and bypass the
certification stage. Candidate negatives use SAT for PM-STB and LC-STB; PM-CSS
uses SAT for rank at most 9, matroid isomorphism through n=28, and a scalable
invariant certificate beyond that region. A matching large-code invariant,
certification timeout, or failure is unresolved: it does not become a negative,
does not trigger a replacement proposal, and blocks timed execution.
Certification and hybrid execution occur in different supervised processes;
there is no solver-object reuse, although earlier stages can warm OS file/page
caches and imported module pages.

Each generated pair and its complete provenance is atomically persisted before
certification or hybrid execution. Per-stage JSON records are the durable
source of truth; CSV files are replaceable materialized views. A campaign
manifest binds resume to the exact configuration and implementation identity,
and an advisory file lock rejects concurrent collectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import z3

from benchmarks.experiments.generators_random import (
    _certificate as _css_certificate,
    _uses_additive_invariant,
)
from benchmarks.experiments.generators_structured import (
    NAMED_CODE_SPECS,
    load_named_code,
)
from benchmarks.experiments.run import RunResult, run
from benchmarks.experiments.statistics import deterministic_seeds
from paper.hybrids.lc_stb import are_lceq
from paper.hybrids.pm_css import are_peq_css
from paper.hybrids.pm_stb import are_peq_stab
from src.algorithms.lc_stb.lc_stb_sat import _build_lceq_stab_sat_solver
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import _build_peq_css_sat_solver
from src.algorithms.p_stb.p_stab_sat import _build_peq_stab_sat_solver
from src.core.css_code import CSSCode
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode

try:
    import fcntl
except ImportError:  # pragma: no cover - production target is Linux
    fcntl = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 4

REQUESTED_CODES = (
    "bell", "3q_rep", "5q_prf", "steane", "shor", "carbon",
    "hamming_15", "15q_optimal", "tetrahedral", "golay", "rot_surf_d5",
    "hamming_31", "coco_488", "coco_666", "bb_72", "bb_90", "bb_108",
    "bb_144",
)
NON_CSS_CODES = frozenset({"5q_prf", "15q_optimal"})

MASTER_SEED = 42
NUM_SEEDS = 10
SEED_UPPER_BOUND = 1_000
PERTURBATION_DEPTH = 2
PRESENTATION_DEPTH = 4
NEGATIVE_MAX_ATTEMPTS = 1_000
CSS_SAT_MAX_R = 9
CSS_MATROID_MAX_N = 28

GENERATION_TIMEOUT_SECONDS = 300.0
CERTIFICATION_TIMEOUT_SECONDS = 900.0
EXECUTION_TIMEOUT_SECONDS = 5_400.0
GENERATION_MEMORY_BYTES = 4 * 1024**3
CERTIFICATION_MEMORY_BYTES = 13 * 1024**3
EXECUTION_MEMORY_BYTES = 13 * 1024**3

OUTPUT_DIRECTORY = ROOT / "paper" / "data" / "collected" / "hybrids_a8_v4"
PREFLIGHT_OUTPUT_DIRECTORY = ROOT / "paper" / "data" / "preflight" / "a8"

COMPONENTS = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE")
TRIVIAL = "trivial"
UNREACHED = "start"
DECIDED_MARKER = "#decided_by "


@dataclass(frozen=True)
class Hybrid:
    name: str
    problem: str
    function: Callable[..., tuple[bool, str]]
    css_only: bool


HYBRIDS: dict[str, Hybrid] = {
    item.name: item
    for item in (
        Hybrid("pm_stb_hybrid", "pm_stb", are_peq_stab, False),
        Hybrid("pm_css_hybrid", "pm_css", are_peq_css, True),
        Hybrid("lc_stb_hybrid", "lc_stb", are_lceq, False),
    )
}
ALGORITHM_ALIASES = {
    "pm-stb": "pm_stb_hybrid",
    "pm-css": "pm_css_hybrid",
    "lc-stb": "lc_stb_hybrid",
    **{name: name for name in HYBRIDS},
}
POPULATIONS = ("positive_control", "certified_negative")

IMPLEMENTATION_FILES = (
    "paper/benchmarks/collect_a8.py",
    "paper/hybrids/pm_stb.py", "paper/hybrids/pm_css.py", "paper/hybrids/lc_stb.py",
    "benchmarks/experiments/run.py", "benchmarks/experiments/generators_structured.py",
    "benchmarks/experiments/generators_random.py", "benchmarks/experiments/utils.py",
    "src/algorithms/p_stb/p_stab_sat.py", "src/algorithms/p_css/p_css_sat.py",
    "src/algorithms/p_css/p_css_matroid.py",
    "src/algorithms/lc_stb/lc_stb_sat.py",
    "src/core/stabilizer_code.py", "src/core/css_code.py", "src/core/pauli.py",
)

INSTANCE_FIELDS = (
    "campaign_id", "algorithm", "problem", "code", "n", "k",
    "population", "seed", "applicable", "input_id", "input_status",
    "certification_status", "certification_label", "certification_method",
    "num_certified_equivalent_proposals",
    "certification_runtime_seconds", "certification_timeout_seconds",
    "certification_memory_limit_bytes", "execution_status", "runtime_seconds",
    "execution_timeout_seconds", "execution_memory_limit_bytes",
    "expected_decision", "decision", "correct", "decided_by", "stuck_at",
    "trace", "error", "execution_attempt",
)
SUMMARY_FIELDS = (
    "campaign_id", "algorithm", "problem", "code", "n", "k", "population",
    "applicable", "status", "num_requested", "num_generated",
    "num_generation_failures", "num_certified_equivalent",
    "num_certified_inequivalent", "num_unresolved_labels",
    "num_certification_failures", "num_execution_attempted", "num_successful",
    "num_correct", "num_incorrect", "num_timeouts", "num_memory_limited",
    "num_errors", "num_blocked", "coverage_fraction", "mean_success_seconds",
    "restricted_mean_seconds", "stddev_success_seconds", "maximum_seconds",
    "deciders", "stuck_at", "execution_timeout_seconds",
    "execution_memory_limit_bytes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".csv", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CampaignLock(AbstractContextManager["CampaignLock"]):
    """Non-blocking advisory lock held for the entire collection process."""

    def __init__(self, output_directory: Path) -> None:
        self.path = output_directory / ".collector.lock"
        self.handle: Any = None

    def __enter__(self) -> "CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        if fcntl is None:  # pragma: no cover
            self.handle.close()
            raise RuntimeError("A8 collection requires advisory file locking")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.seek(0)
            owner = self.handle.read().strip() or "unknown owner"
            self.handle.close()
            raise RuntimeError(f"campaign is already locked: {owner}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} host={socket.gethostname()} started={_utc_now()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "ldpc", "z3-solver", "pynauty", "stim"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _implementation_identity() -> dict[str, Any]:
    hashes = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_FILES
    }
    revision = "unknown"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "git_revision": revision,
        "source_sha256": hashes,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": _package_versions(),
    }


def _named_source_hash(name: str) -> str:
    stem, is_css = NAMED_CODE_SPECS[name]
    if stem is None:
        return _digest({"builtin": "bell", "is_css": is_css,
                        "Hx": [[1, 1]], "Hz": [[1, 1]]})
    return hashlib.sha256((ROOT / "data" / stem).read_bytes()).hexdigest()


def _canonical_algorithms(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        try:
            canonical = ALGORITHM_ALIASES[value]
        except KeyError as exc:
            raise ValueError(f"unknown hybrid {value!r}") from exc
        if canonical not in result:
            result.append(canonical)
    return tuple(result)


def _validate_codes(codes: Sequence[str]) -> tuple[str, ...]:
    unknown = set(codes) - set(REQUESTED_CODES)
    if unknown:
        raise ValueError(f"codes outside the A8 population: {sorted(unknown)}")
    if not codes:
        raise ValueError("at least one code is required")
    return tuple(dict.fromkeys(codes))


def _build_configuration(args: argparse.Namespace) -> dict[str, Any]:
    preflight = bool(args.preflight)
    num_seeds = args.num_seeds if args.num_seeds is not None else (1 if preflight else NUM_SEEDS)
    if num_seeds <= 0:
        raise ValueError("num-seeds must be positive")
    if args.seeds:
        seeds = tuple(dict.fromkeys(args.seeds))
    else:
        seeds = deterministic_seeds(args.master_seed, num_seeds, upper_bound=args.seed_upper_bound)
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")

    def value(name: str, normal: float, smoke: float) -> float:
        explicit = getattr(args, name)
        return float(explicit if explicit is not None else (smoke if preflight else normal))

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "mode": "preflight" if preflight else "production",
        "codes": list(_validate_codes(args.codes)),
        "algorithms": list(_canonical_algorithms(args.algorithms)),
        "master_seed": args.master_seed, "seed_upper_bound": args.seed_upper_bound,
        "seeds": list(seeds),
        "certified_negative": {
            "depth": args.perturbation_depth,
            "css_gate_distribution": {"CX": 1.0},
            "stabilizer_gate_distribution": {
                gate: 1 / 9 for gate in ("H", "S", "Sdg", "X", "Y", "Z", "CX", "CZ", "SWAP")
            },
            "single_qubit_coordinates": "uniform over 0..n-1 per gate",
            "two_qubit_coordinates": "uniform ordered distinct pair per gate",
            "sampling_between_gates": "independent with replacement",
            "max_attempts": args.negative_max_attempts,
            "selection": "first proposal independently certified inequivalent",
            "pm_css_backend_policy": {
                "sat_max_check_rank": CSS_SAT_MAX_R,
                "matroid_max_n": CSS_MATROID_MAX_N,
                "larger": "fixed-depth named-code CNOT proposal; one-sided invariant witness",
            },
        },
        "positive_control": {
            "presentation_row_operations_per_sector": args.presentation_depth,
            "permutation": "uniform NumPy permutation",
            "lc_local_cliffords": "independent uniform over I,H,S,HS,SH,HSH",
        },
        "budgets": {
            "generation": {
                "timeout_seconds": value("generation_timeout", GENERATION_TIMEOUT_SECONDS, 4.0),
                "memory_limit_bytes": int(value("generation_memory_gib", GENERATION_MEMORY_BYTES / 1024**3, 2.0) * 1024**3),
            },
            "certification": {
                "timeout_seconds": value("certification_timeout", CERTIFICATION_TIMEOUT_SECONDS, 4.0),
                "memory_limit_bytes": int(value("certification_memory_gib", CERTIFICATION_MEMORY_BYTES / 1024**3, 2.0) * 1024**3),
            },
            "execution": {
                "timeout_seconds": value("execution_timeout", EXECUTION_TIMEOUT_SECONDS, 4.0),
                "memory_limit_bytes": int(value("execution_memory_gib", EXECUTION_MEMORY_BYTES / 1024**3, 2.0) * 1024**3),
            },
        },
        "implementation": _implementation_identity(),
        "named_source_sha256": {
            name: _named_source_hash(name)
            for name in _validate_codes(args.codes)
        },
    }
    if (args.perturbation_depth < 1 or args.presentation_depth < 0
            or args.negative_max_attempts < 1):
        raise ValueError(
            "perturbation depth and negative attempts must be positive; "
            "presentation depth must be non-negative"
        )
    for stage, budget in configuration["budgets"].items():
        if budget["timeout_seconds"] <= 0 or budget["memory_limit_bytes"] <= 0:
            raise ValueError(f"{stage} budgets must be positive")
    return configuration


def _campaign_manifest(configuration: Mapping[str, Any]) -> dict[str, Any]:
    campaign_id = _digest(configuration)
    return {"schema_version": SCHEMA_VERSION, "campaign_id": campaign_id,
            "created_at": _utc_now(), "configuration": configuration}


def _prepare_manifest(output_directory: Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    path = output_directory / "campaign.json"
    expected = _campaign_manifest(configuration)
    if path.exists():
        actual = _read_json(path)
        if actual.get("campaign_id") != expected["campaign_id"]:
            raise ValueError(
                "incompatible resume configuration or implementation; use a new output "
                f"directory (existing={actual.get('campaign_id')}, requested={expected['campaign_id']})"
            )
        return actual
    leftovers = [item for item in output_directory.iterdir() if item.name != ".collector.lock"]
    if leftovers:
        raise ValueError(
            f"refusing to mix a new campaign with existing files in {output_directory}: "
            f"{', '.join(item.name for item in leftovers[:5])}"
        )
    _atomic_json(path, expected)
    return expected


def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return int(mod2.rank(np.asarray(matrix, dtype=np.int8) % 2))


def _row_basis(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.int8) % 2
    if matrix.shape[0] == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.int8)
    basis = mod2.row_basis(matrix)
    if hasattr(basis, "toarray"):
        basis = basis.toarray()
    basis = np.asarray(basis, dtype=np.int8) % 2
    return basis.reshape((-1, matrix.shape[1]))


def _code_record(code: StabilizerCode) -> dict[str, Any]:
    common = {"n": int(code.n), "k": int(code.k), "distance": None,
              "distance_status": "unavailable"}
    if isinstance(code, CSSCode):
        return {**common, "type": "css",
                "Hx": (np.asarray(code.Hx, dtype=np.int8) % 2).tolist(),
                "Hz": (np.asarray(code.Hz, dtype=np.int8) % 2).tolist(),
                "x_distance": None, "z_distance": None}
    return {**common, "type": "stabilizer",
            "symplectic": (np.asarray(code.symplectic, dtype=np.int8) % 2).tolist()}


def _record_matrix(record: Mapping[str, Any], field: str, columns: int) -> np.ndarray:
    raw = record[field]
    if not raw:
        return np.zeros((0, columns), dtype=np.int8)
    matrix = np.asarray(raw, dtype=np.int8) % 2
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"invalid persisted {field} shape {matrix.shape}")
    return matrix


def _code_from_record(record: Mapping[str, Any]) -> StabilizerCode:
    n = int(record["n"])
    if record["type"] == "css":
        css_code = CSSCode(_record_matrix(record, "Hx", n), _record_matrix(record, "Hz", n), n=n)
        setattr(css_code, "distance", None)
        setattr(css_code, "x_distance", None)
        setattr(css_code, "z_distance", None)
        return css_code
    if record["type"] != "stabilizer":
        raise ValueError(f"unknown persisted code type {record['type']!r}")
    stabilizer_code = StabilizerCode(
        StabilizerTableau(_record_matrix(record, "symplectic", 2 * n)), n=n
    )
    setattr(stabilizer_code, "distance", None)
    return stabilizer_code


def _pair_from_payload(payload: Mapping[str, Any]) -> tuple[StabilizerCode, StabilizerCode]:
    return _code_from_record(payload["left"]), _code_from_record(payload["right"])


def _read_input(path: Path, expected_input_id: str) -> dict[str, Any]:
    payload = _read_json(path)
    stored_digest = payload.pop("payload_sha256", None)
    actual_digest = _digest(payload)
    payload["payload_sha256"] = stored_digest
    if stored_digest != actual_digest:
        raise ValueError(f"persisted input checksum mismatch: {path}")
    if payload.get("input_id") != expected_input_id:
        raise ValueError(
            f"persisted input id mismatch in {path}: {payload.get('input_id')!r}"
        )
    return payload


def _derived_seed(seed: int, code_name: str, mode: str) -> int:
    data = f"a8-v{SCHEMA_VERSION}:{seed}:{code_name}:{mode}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _attempt_seed(problem: str, n: int, k: int, seed: int, attempt: int) -> int:
    """Match the seeded proposal sequence used by the A1 rejection benchmark."""
    population = f"{problem}_negative_matching=False"
    value = f"{population}|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _css_certifier_kind(n: int, k: int) -> str:
    if n - k <= CSS_SAT_MAX_R:
        return "sat"
    if n <= CSS_MATROID_MAX_N:
        return "matroid"
    return "invariant_witness"


def _row_randomize(matrix: np.ndarray, rng: np.random.Generator, steps: int,
                   sector: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    changed = np.asarray(matrix, dtype=np.int8).copy() % 2
    operations: list[dict[str, Any]] = []
    if changed.shape[0] < 2:
        return changed, operations
    for _ in range(steps):
        source, target = (int(value) for value in rng.choice(changed.shape[0], 2, replace=False))
        changed[target] ^= changed[source]
        operations.append({"gate": "row_add", "sector": sector,
                           "source": source, "target": target})
    return changed, operations


def _positive_pm(code: StabilizerCode, rng: np.random.Generator,
                 presentation_depth: int) -> tuple[StabilizerCode, list[dict[str, Any]]]:
    permutation = [int(value) for value in rng.permutation(code.n)]
    operations: list[dict[str, Any]] = [{"gate": "qubit_permutation", "permutation": permutation}]
    partner: StabilizerCode
    if isinstance(code, CSSCode):
        hx, x_ops = _row_randomize(code.Hx, rng, presentation_depth, "X")
        hz, z_ops = _row_randomize(code.Hz, rng, presentation_depth, "Z")
        operations.extend(x_ops + z_ops)
        partner = CSSCode(hx[:, permutation], hz[:, permutation], n=code.n)
    else:
        matrix, row_ops = _row_randomize(code.symplectic, rng, presentation_depth, "stabilizer")
        operations.extend(row_ops)
        columns = permutation + [qubit + code.n for qubit in permutation]
        partner = StabilizerCode(StabilizerTableau(matrix[:, columns]), n=code.n)
    return partner, operations


def _apply_local_clifford(tableau: StabilizerTableau, operation: str, qubit: int) -> None:
    for gate in reversed(operation):
        if gate == "H":
            tableau.apply_h(qubit)
        elif gate == "S":
            tableau.apply_s(qubit)


def _positive_lc(code: StabilizerCode, rng: np.random.Generator,
                 presentation_depth: int) -> tuple[StabilizerCode, list[dict[str, Any]]]:
    tableau = code.generators.copy()
    operations: list[dict[str, Any]] = []
    choices = ("I", "H", "S", "HS", "SH", "HSH")
    for qubit in range(code.n):
        operation = str(rng.choice(choices))
        _apply_local_clifford(tableau, operation, qubit)
        operations.append({"gate": "local_clifford", "operation": operation, "qubit": qubit})
    matrix, row_ops = _row_randomize(tableau.tableau.matrix, rng, presentation_depth, "stabilizer")
    operations.extend(row_ops)
    return StabilizerCode(StabilizerTableau(matrix), n=code.n), operations


def _perturb(code: StabilizerCode, rng: np.random.Generator, depth: int, *,
             css_preserving: bool = False) -> tuple[StabilizerCode, list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []
    if css_preserving:
        if not isinstance(code, CSSCode):
            raise TypeError("CSS-preserving perturbation requires a CSS source")
        hx = np.asarray(code.Hx, dtype=np.int8).copy() % 2
        hz = np.asarray(code.Hz, dtype=np.int8).copy() % 2
        for _ in range(depth):
            control, target = (int(value) for value in rng.choice(code.n, 2, replace=False))
            hx[:, target] ^= hx[:, control]
            hz[:, control] ^= hz[:, target]
            operations.append({"gate": "CX", "control": control, "target": target})
        return CSSCode(hx, hz, n=code.n), operations

    tableau = code.generators.copy()
    for _ in range(depth):
        choices = ("H", "S", "Sdg", "X", "Y", "Z", "CX", "CZ", "SWAP")
        gate = str(rng.choice(choices if code.n > 1 else choices[:6]))
        if gate in {"CX", "CZ", "SWAP"}:
            left, right = (int(value) for value in rng.choice(code.n, 2, replace=False))
            getattr(tableau, f"apply_{gate.lower()}")(left, right)
            operations.append({"gate": gate, "left": left, "right": right})
        else:
            qubit = int(rng.integers(0, code.n))
            getattr(tableau, f"apply_{gate.lower()}")(qubit)
            operations.append({"gate": gate, "qubit": qubit})
    return StabilizerCode(StabilizerTableau(tableau.tableau.matrix.copy()), n=code.n), operations


def _validate_generated_pair(source: StabilizerCode, partner: StabilizerCode, mode: str) -> None:
    if (source.n, source.k) != (partner.n, partner.k):
        raise ValueError(f"{mode} changed dimensions from [[{source.n},{source.k}]] to [[{partner.n},{partner.k}]]")
    if isinstance(source, CSSCode) and (mode == "positive_pm" or mode == "negative_pm_css"):
        if not isinstance(partner, CSSCode):
            raise ValueError(f"{mode} did not preserve CSS representation")
        if np.any((partner.Hx @ partner.Hz.T) % 2):
            raise ValueError(f"{mode} violated CSS orthogonality")
        if (_rank(source.Hx), _rank(source.Hz)) != (_rank(partner.Hx), _rank(partner.Hz)):
            raise ValueError(f"{mode} changed CSS check ranks")


def generate_input_payload(spec: Mapping[str, Any], perturbation_depth: int,
                           presentation_depth: int) -> dict[str, Any]:
    """Worker entry point for one deterministic control or negative proposal."""
    code_name = str(spec["code"])
    seed = int(spec["seed"])
    mode = str(spec["mode"])
    source = load_named_code(code_name)
    attempt = int(spec.get("attempt", 0))
    relation = str(spec.get("relation", ""))
    derived = (
        _attempt_seed(relation, source.n, source.k, seed, attempt)
        if mode.startswith("negative_")
        else _derived_seed(seed, code_name, mode)
    )
    rng = np.random.default_rng(derived)
    inequivalence_witness: dict[str, Any] | None = None
    witness: str | None
    if mode == "positive_pm":
        partner, operations = _positive_pm(source, rng, presentation_depth)
        population = "positive_control"
        witness = "recorded qubit permutation and invertible row additions"
    elif mode == "positive_lc":
        partner, operations = _positive_lc(source, rng, presentation_depth)
        population = "positive_control"
        witness = "recorded local Cliffords and invertible row additions"
    elif mode.startswith("negative_"):
        if relation not in {"pm_stb", "pm_css", "lc_stb"}:
            raise ValueError(f"unknown negative relation {relation!r}")
        partner, operations = _perturb(
            source, rng, perturbation_depth, css_preserving=relation == "pm_css"
        )
        if relation == "pm_css" and _css_certifier_kind(source.n, source.k) == "invariant_witness":
            assert isinstance(source, CSSCode) and isinstance(partner, CSSCode)
            source_certificate = _css_certificate(source)
            partner_certificate = _css_certificate(partner)
            total_rank = _rank(source.Hx) + _rank(source.Hz)
            certificate_kind = (
                "additive_collision_rank"
                if _uses_additive_invariant(source)
                else "support_rank" if total_rank > 20
                else "stabilizer_weight_enumerator"
            )
            inequivalence_witness = {
                "method": certificate_kind,
                "source_sha256": hashlib.sha256(repr(source_certificate).encode()).hexdigest(),
                "partner_sha256": hashlib.sha256(repr(partner_certificate).encode()).hexdigest(),
                "separates": source_certificate != partner_certificate,
            }
            witness = (
                "mismatched sound CSS permutation invariant"
                if source_certificate != partner_certificate else None
            )
        else:
            witness = None
        population = "certified_negative"
    else:
        raise ValueError(f"unknown input mode {mode!r}")
    _validate_generated_pair(source, partner, mode)
    left = _code_record(source)
    right = _code_record(partner)
    payload = {
        "schema_version": SCHEMA_VERSION, "input_id": spec["input_id"],
        "campaign_id": spec.get("campaign_id", "standalone"),
        "source": {"registry_name": code_name,
                   "registry_spec": list(NAMED_CODE_SPECS[code_name]),
                   "matrix_sha256": _digest(left)},
        "seed": seed, "derived_seed": derived, "mode": mode,
        "relation": relation or None, "attempt": attempt,
        "population": population, "operations": operations,
        "construction_witness": witness,
        "inequivalence_witness": inequivalence_witness,
        "distance_metadata": "unavailable; not copied to either persisted code",
        "left": left, "right": right,
    }
    payload["payload_sha256"] = _digest(payload)
    return payload


def generate_input_file(spec: Mapping[str, Any], perturbation_depth: int,
                        presentation_depth: int, output_path: str) -> dict[str, str]:
    """Generate and atomically persist an input, returning only small metadata."""
    payload = generate_input_payload(spec, perturbation_depth, presentation_depth)
    _atomic_json(Path(output_path), payload)
    return {
        "input_id": str(payload["input_id"]),
        "payload_sha256": str(payload["payload_sha256"]),
    }


def _minimal_pair(payload: Mapping[str, Any], relation: str) -> tuple[StabilizerCode, StabilizerCode]:
    left, right = _pair_from_payload(payload)
    if relation == "pm_css":
        if not isinstance(left, CSSCode) or not isinstance(right, CSSCode):
            raise ValueError("PM-CSS certification requires two CSS inputs")
        return (CSSCode(_row_basis(left.Hx), _row_basis(left.Hz), n=left.n),
                CSSCode(_row_basis(right.Hx), _row_basis(right.Hz), n=right.n))
    return (StabilizerCode(StabilizerTableau(_row_basis(left.symplectic)), n=left.n),
            StabilizerCode(StabilizerTableau(_row_basis(right.symplectic)), n=right.n))


def certify_payload(relation: str, payload: Mapping[str, Any]) -> dict[str, str]:
    """Certify one proposed negative with the A1 backend policy."""
    left, right = _minimal_pair(payload, relation)
    if relation == "pm_stb":
        solver = _build_peq_stab_sat_solver(left, right)
    elif relation == "pm_css":
        assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
        backend = _css_certifier_kind(left.n, left.k)
        if backend == "matroid":
            equivalent = bool(are_peq_css_matroid(left, right))
            return {
                "label": "equivalent" if equivalent else "inequivalent",
                "solver_result": str(equivalent).lower(),
                "reason_unknown": "",
                "method": "exact_pm_css_matroid",
            }
        if backend == "invariant_witness":
            raise ValueError("large PM-CSS inputs must carry a one-sided invariant result")
        solver = _build_peq_css_sat_solver(left, right)
    elif relation == "lc_stb":
        solver = _build_lceq_stab_sat_solver(left, right)
    else:
        raise ValueError(f"unknown certification relation {relation!r}")
    result = solver.check()
    if result == z3.sat:
        label = "equivalent"
    elif result == z3.unsat:
        label = "inequivalent"
    else:
        label = "unresolved"
    return {
        "label": label,
        "solver_result": str(result),
        "reason_unknown": solver.reason_unknown() if label == "unresolved" else "",
        "method": f"exact_{relation}_sat",
    }


def certify_input_file(relation: str, input_path: str,
                       expected_input_id: str) -> dict[str, str]:
    """Load one durable input inside the certification worker and certify it."""
    return certify_payload(relation, _read_input(Path(input_path), expected_input_id))


@dataclass(frozen=True)
class TracedHybrid:
    function: Callable[..., Any]
    log_path: str
    input_path: str
    expected_input_id: str

    def __call__(self) -> bool | None:
        with open(self.log_path, "w", buffering=1, encoding="utf-8") as log:
            saved_stdout, saved_stderr = sys.stdout, sys.stderr
            saved_fd, saved_error_fd = os.dup(1), os.dup(2)
            try:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                sys.stdout = log
                sys.stderr = log
                inputs = _pair_from_payload(
                    _read_input(Path(self.input_path), self.expected_input_id)
                )
                payload = self.function(*inputs)
                if isinstance(payload, tuple) and len(payload) == 2:
                    decision, component = payload
                else:
                    decision, component = payload, ""
                print(f"{DECIDED_MARKER}{component or TRIVIAL}")
                return bool(decision) if decision is not None else None
            finally:
                sys.stdout = saved_stdout
                sys.stderr = saved_stderr
                os.dup2(saved_fd, 1)
                os.dup2(saved_error_fd, 2)
                os.close(saved_fd)
                os.close(saved_error_fd)


def _read_trace(path: Path) -> tuple[list[str], str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], ""
    trace: list[str] = []
    decided_by = ""
    for line in lines:
        line = line.strip()
        if line in COMPONENTS:
            trace.append(line)
        elif line.startswith(DECIDED_MARKER):
            decided_by = line[len(DECIDED_MARKER):]
    return trace, decided_by


def _supervision_status(result: RunResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.memory_exceeded:
        return "memory_limited"
    if result.error is not None:
        return "error"
    return "success"


def _stage_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    history: list[dict[str, Any]] = []
    attempt = 1
    if path.exists():
        previous = _read_json(path)
        history = list(previous.get("history", []))
        history.append({key: value for key, value in previous.items() if key != "history"})
        attempt = int(previous.get("attempt", 1)) + 1
    payload["attempt"] = attempt
    payload["recorded_at"] = _utc_now()
    if history:
        payload["history"] = history
    _atomic_json(path, payload)
    return payload


def _input_id(mode: str, code: str, seed: int) -> str:
    return f"{mode}__{code}__{seed}"


def _case_id(algorithm: str, code: str, population: str, seed: int) -> str:
    return f"{algorithm}__{code}__{population}__{seed}"


def _proposal_path(output: Path, input_id: str, attempt: int) -> Path:
    return output / "proposals" / input_id / f"attempt-{attempt:04d}.json"


def _input_mode(algorithm: str, population: str) -> str:
    if population == "certified_negative":
        return f"negative_{HYBRIDS[algorithm].problem}"
    return "positive_lc" if algorithm == "lc_stb_hybrid" else "positive_pm"


def _applicable(algorithm: str, code: str) -> bool:
    return not (HYBRIDS[algorithm].css_only and code in NON_CSS_CODES)


def _input_specs(configuration: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    campaign_id = _digest(configuration)
    for task in _execution_tasks(configuration):
        if not task["applicable"] or task["input_id"] in seen:
            continue
        seen.add(task["input_id"])
        mode = _input_mode(task["algorithm"], task["population"])
        specs.append({
            "input_id": task["input_id"], "campaign_id": campaign_id,
            "mode": mode, "relation": HYBRIDS[task["algorithm"]].problem
            if task["population"] == "certified_negative" else "",
            "code": task["code"], "seed": task["seed"], "attempt": 0,
        })
    return specs


def _execution_tasks(configuration: Mapping[str, Any]) -> list[dict[str, Any]]:
    algorithms = list(configuration["algorithms"])
    tasks: list[dict[str, Any]] = []
    for seed in configuration["seeds"]:
        for code in configuration["codes"]:
            for population in POPULATIONS:
                for algorithm in algorithms:
                    tasks.append({
                        "case_id": _case_id(algorithm, code, population, seed),
                        "algorithm": algorithm, "code": code, "population": population,
                        "seed": seed, "applicable": _applicable(algorithm, code),
                        "input_id": _input_id(_input_mode(algorithm, population), code, seed),
                    })
    return tasks


def _certification_tasks(configuration: Mapping[str, Any]) -> list[dict[str, Any]]:
    needed: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for task in _execution_tasks(configuration):
        if task["applicable"] and task["population"] == "certified_negative":
            relation = HYBRIDS[task["algorithm"]].problem
            needed[(task["seed"], task["population"], task["code"], relation)] = {
                "input_id": task["input_id"], "code": task["code"],
                "seed": task["seed"], "population": task["population"],
                "relation": relation,
            }
    relations = list(dict.fromkeys(
        HYBRIDS[algorithm].problem for algorithm in configuration["algorithms"]
    ))
    tasks: list[dict[str, Any]] = []
    for seed in configuration["seeds"]:
        for code in configuration["codes"]:
            for population in POPULATIONS:
                for relation in relations:
                    candidate = needed.get((seed, population, code, relation))
                    if candidate is not None:
                        tasks.append(candidate)
    return tasks


class EventLog:
    def __init__(self, path: Path, campaign_id: str) -> None:
        self.path = path
        self.campaign_id = campaign_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, key: str, record: Mapping[str, Any]) -> None:
        event = {"campaign_id": self.campaign_id, "time": _utc_now(),
                 "stage": stage, "key": key, **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class FailureGuard:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.counts: Counter[tuple[str, str]] = Counter()

    def observe(self, stage: str, status: str, error: str) -> None:
        if self.maximum <= 0 or status != "error":
            return
        signature = error.splitlines()[0][:240]
        key = (stage, signature)
        self.counts[key] += 1
        if self.counts[key] >= self.maximum:
            raise RuntimeError(f"systematic {stage} failure repeated {self.counts[key]} times: {signature}")


def _needs_retry(record: Mapping[str, Any], stage: str, retries: set[str]) -> bool:
    if stage not in retries:
        return False
    if record.get("status") != "success":
        return True
    return stage == "certification" and record.get("label") == "unresolved"


def _collect_generation(output: Path, configuration: Mapping[str, Any],
                        retries: set[str], events: EventLog, guard: FailureGuard,
                        verbose: bool) -> None:
    budget = configuration["budgets"]["generation"]
    for spec in _input_specs(configuration):
        input_path = (
            _proposal_path(output, str(spec["input_id"]), 0)
            if spec["mode"].startswith("negative_")
            else output / "inputs" / f"{spec['input_id']}.json"
        )
        stage_path = output / "stages" / "generation" / f"{spec['input_id']}.json"
        previous = _read_json(stage_path) if stage_path.exists() else None
        if previous is not None and not _needs_retry(previous, "generation", retries):
            continue
        result = run(
            generate_input_file,
            (spec, configuration["certified_negative"]["depth"],
             configuration["positive_control"]["presentation_row_operations_per_sector"],
             str(input_path)),
            None, timeout=budget["timeout_seconds"],
            max_memory_bytes=budget["memory_limit_bytes"],
        )
        status = _supervision_status(result)
        error = result.error or ""
        if status == "success":
            metadata = result.result if isinstance(result.result, dict) else {}
            try:
                payload = _read_input(input_path, str(spec["input_id"]))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                status, error = "error", f"invalid generation output: {type(exc).__name__}: {exc}"
            else:
                if metadata != {
                    "input_id": payload["input_id"],
                    "payload_sha256": payload["payload_sha256"],
                }:
                    status, error = "error", "generation worker returned mismatched metadata"
        record = _stage_record(stage_path, {
            "stage": "generation", "input_id": spec["input_id"], "status": status,
            "runtime_seconds": result.runtime, "timeout_seconds": budget["timeout_seconds"],
            "memory_limit_bytes": budget["memory_limit_bytes"], "error": error,
            "input_path": str(input_path.relative_to(output)) if status == "success" else "",
        })
        events.write("generation", spec["input_id"], record)
        guard.observe("generation", status, error)
        if verbose:
            print(f"generation {spec['input_id']}: {status} {result.runtime:.3f}s", flush=True)


def _collect_certification(output: Path, configuration: Mapping[str, Any],
                           retries: set[str], events: EventLog, guard: FailureGuard,
                           verbose: bool) -> None:
    cert_budget = configuration["budgets"]["certification"]
    generation_budget = configuration["budgets"]["generation"]
    max_attempts = int(configuration["certified_negative"]["max_attempts"])
    presentation_depth = int(
        configuration["positive_control"]["presentation_row_operations_per_sector"]
    )
    perturbation_depth = int(configuration["certified_negative"]["depth"])
    for task in _certification_tasks(configuration):
        key = f"{task['input_id']}__{task['relation']}"
        stage_path = output / "stages" / "certification" / f"{key}.json"
        previous = _read_json(stage_path) if stage_path.exists() else None
        generation_path = output / "stages" / "generation" / f"{task['input_id']}.json"
        generation = _read_json(generation_path) if generation_path.exists() else {}
        dependency_recovered = (
            previous is not None
            and previous.get("status") == "blocked_generation"
            and generation.get("status") == "success"
            and _proposal_path(output, str(task["input_id"]), 0).exists()
        )
        if (
            previous is not None
            and not dependency_recovered
            and not _needs_retry(previous, "certification", retries)
        ):
            continue
        first_proposal = _proposal_path(output, str(task["input_id"]), 0)
        if generation.get("status") != "success" or not first_proposal.exists():
            record = _stage_record(stage_path, {
                "stage": "certification", "input_id": task["input_id"],
                "relation": task["relation"], "status": "blocked_generation",
                "label": "unresolved", "method": "", "runtime_seconds": 0.0,
                "timeout_seconds": cert_budget["timeout_seconds"],
                "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                "num_proposals": 0, "num_certified_equivalent_proposals": 0,
                "error": "input generation did not succeed",
            })
        else:
            record = {}
            certification_runtime = 0.0
            equivalent_proposals = 0
            for attempt in range(max_attempts):
                proposal_path = _proposal_path(output, str(task["input_id"]), attempt)
                if attempt > 0 and not proposal_path.exists():
                    proposal_spec = {
                        "input_id": task["input_id"],
                        "campaign_id": _digest(configuration),
                        "mode": f"negative_{task['relation']}",
                        "relation": task["relation"], "code": task["code"],
                        "seed": task["seed"], "attempt": attempt,
                    }
                    generation_attempt_path = (
                        output / "stages" / "generation_attempts"
                        / f"{task['input_id']}__{attempt:04d}.json"
                    )
                    generated = run(
                        generate_input_file,
                        (proposal_spec, perturbation_depth, presentation_depth,
                         str(proposal_path)),
                        None, timeout=generation_budget["timeout_seconds"],
                        max_memory_bytes=generation_budget["memory_limit_bytes"],
                    )
                    generation_status = _supervision_status(generated)
                    generation_error = generated.error or ""
                    generation_attempt = _stage_record(generation_attempt_path, {
                        "stage": "generation_attempt", "input_id": task["input_id"],
                        "relation": task["relation"], "attempt_index": attempt,
                        "status": generation_status, "runtime_seconds": generated.runtime,
                        "timeout_seconds": generation_budget["timeout_seconds"],
                        "memory_limit_bytes": generation_budget["memory_limit_bytes"],
                        "proposal_path": str(proposal_path.relative_to(output)),
                        "error": generation_error,
                    })
                    events.write("generation_attempt", f"{task['input_id']}:{attempt}", generation_attempt)
                    guard.observe("generation", generation_status, generation_error)
                    if generation_status != "success" or not proposal_path.exists():
                        record = {
                            "stage": "certification", "input_id": task["input_id"],
                            "relation": task["relation"], "status": "error",
                            "label": "unresolved", "method": "",
                            "runtime_seconds": certification_runtime,
                            "timeout_seconds": cert_budget["timeout_seconds"],
                            "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                            "num_proposals": attempt + 1,
                            "num_certified_equivalent_proposals": equivalent_proposals,
                            "error": f"negative proposal generation failed: {generation_error}",
                        }
                        break

                try:
                    proposal = _read_input(proposal_path, str(task["input_id"]))
                except (OSError, ValueError) as exc:
                    record = {
                        "stage": "certification", "input_id": task["input_id"],
                        "relation": task["relation"], "status": "error",
                        "label": "unresolved", "method": "",
                        "runtime_seconds": certification_runtime,
                        "timeout_seconds": cert_budget["timeout_seconds"],
                        "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                        "num_proposals": attempt + 1,
                        "num_certified_equivalent_proposals": equivalent_proposals,
                        "error": f"invalid proposal: {type(exc).__name__}: {exc}",
                    }
                    break

                inequivalence_witness = proposal.get("inequivalence_witness")
                if inequivalence_witness:
                    separates = bool(inequivalence_witness.get("separates"))
                    record = {
                        "stage": "certification", "input_id": task["input_id"],
                        "relation": task["relation"],
                        "status": "success" if separates else "unresolved",
                        "label": "inequivalent" if separates else "unresolved",
                        "method": f"invariant_{inequivalence_witness['method']}",
                        "solver_result": "not_run", "reason_unknown": (
                            "sound invariant matched; equivalence is undetermined"
                            if not separates else ""
                        ),
                        "runtime_seconds": certification_runtime,
                        "timeout_seconds": cert_budget["timeout_seconds"],
                        "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                        "num_proposals": attempt + 1,
                        "num_certified_equivalent_proposals": equivalent_proposals,
                        "selected_attempt": attempt if separates else None,
                        "selected_input_path": "",
                        "error": "",
                    }
                    if separates:
                        final_path = output / "inputs" / f"{task['input_id']}.json"
                        _atomic_bytes(final_path, proposal_path.read_bytes())
                        record["selected_input_path"] = str(final_path.relative_to(output))
                    break

                attempt_path = (
                    output / "stages" / "certification_attempts"
                    / f"{key}__{attempt:04d}.json"
                )
                saved_attempt = _read_json(attempt_path) if attempt_path.exists() else None
                rerun_attempt = (
                    saved_attempt is None
                    or ("certification" in retries and saved_attempt.get("status") != "success")
                )
                if rerun_attempt:
                    checked = run(
                        certify_input_file,
                        (task["relation"], str(proposal_path), task["input_id"]),
                        None, timeout=cert_budget["timeout_seconds"],
                        max_memory_bytes=cert_budget["memory_limit_bytes"],
                    )
                    status = _supervision_status(checked)
                    detail = (
                        checked.result
                        if status == "success" and isinstance(checked.result, dict)
                        else {}
                    )
                    label = str(detail.get("label", "unresolved"))
                    if label not in {"equivalent", "inequivalent", "unresolved"}:
                        status, label = "error", "unresolved"
                    saved_attempt = _stage_record(attempt_path, {
                        "stage": "certification_attempt", "input_id": task["input_id"],
                        "relation": task["relation"], "attempt_index": attempt,
                        "status": status, "label": label,
                        "method": detail.get("method", ""),
                        "solver_result": detail.get("solver_result", ""),
                        "reason_unknown": detail.get("reason_unknown", ""),
                        "runtime_seconds": checked.runtime,
                        "timeout_seconds": cert_budget["timeout_seconds"],
                        "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                        "proposal_path": str(proposal_path.relative_to(output)),
                        "error": checked.error or "",
                    })
                    events.write("certification_attempt", f"{key}:{attempt}", saved_attempt)
                assert saved_attempt is not None
                certification_runtime += float(saved_attempt.get("runtime_seconds", 0.0))
                label = str(saved_attempt.get("label", "unresolved"))
                status = str(saved_attempt.get("status", "error"))
                if status == "success" and label == "equivalent":
                    equivalent_proposals += 1
                    continue
                if status == "success" and label == "inequivalent":
                    final_path = output / "inputs" / f"{task['input_id']}.json"
                    _atomic_bytes(final_path, proposal_path.read_bytes())
                    record = {
                        "stage": "certification", "input_id": task["input_id"],
                        "relation": task["relation"], "status": "success",
                        "label": "inequivalent", "method": saved_attempt.get("method", ""),
                        "solver_result": saved_attempt.get("solver_result", ""),
                        "reason_unknown": "", "runtime_seconds": certification_runtime,
                        "timeout_seconds": cert_budget["timeout_seconds"],
                        "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                        "num_proposals": attempt + 1, "selected_attempt": attempt,
                        "num_certified_equivalent_proposals": equivalent_proposals,
                        "selected_input_path": str(final_path.relative_to(output)),
                        "error": "",
                    }
                    break
                record = {
                    "stage": "certification", "input_id": task["input_id"],
                    "relation": task["relation"], "status": status,
                    "label": "unresolved", "method": saved_attempt.get("method", ""),
                    "solver_result": saved_attempt.get("solver_result", ""),
                    "reason_unknown": saved_attempt.get("reason_unknown", ""),
                    "runtime_seconds": certification_runtime,
                    "timeout_seconds": cert_budget["timeout_seconds"],
                    "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                    "num_proposals": attempt + 1,
                    "num_certified_equivalent_proposals": equivalent_proposals,
                    "error": saved_attempt.get("error", ""),
                }
                break
            if not record:
                record = {
                    "stage": "certification", "input_id": task["input_id"],
                    "relation": task["relation"], "status": "error",
                    "label": "unresolved", "method": "",
                    "runtime_seconds": certification_runtime,
                    "timeout_seconds": cert_budget["timeout_seconds"],
                    "memory_limit_bytes": cert_budget["memory_limit_bytes"],
                    "num_proposals": max_attempts,
                    "num_certified_equivalent_proposals": equivalent_proposals,
                    "error": f"no inequivalent proposal in {max_attempts} attempts",
                }
            record = _stage_record(stage_path, record)
        events.write("certification", key, record)
        guard.observe("certification", str(record["status"]), str(record.get("error", "")))
        if verbose:
            print(f"certification {key}: {record['status']} {record.get('label', '')} {float(record['runtime_seconds']):.3f}s", flush=True)
        if record.get("status") == "success" and record.get("label") == "inequivalent":
            expected = False
            for execution_task in _execution_tasks(configuration):
                if (execution_task["input_id"] != task["input_id"] or
                        HYBRIDS[execution_task["algorithm"]].problem != task["relation"]):
                    continue
                execution_path = output / "stages" / "execution" / f"{execution_task['case_id']}.json"
                if execution_path.exists():
                    saved = _read_json(execution_path)
                    if saved.get("status") == "success" and isinstance(saved.get("decision"), bool) and saved["decision"] != expected:
                        raise RuntimeError(
                            f"saved hybrid answer disagrees with newly resolved certification for {execution_task['case_id']}"
                        )


def _collect_execution(output: Path, configuration: Mapping[str, Any],
                       retries: set[str], events: EventLog, guard: FailureGuard,
                       verbose: bool) -> None:
    budget = configuration["budgets"]["execution"]
    for task in _execution_tasks(configuration):
        stage_path = output / "stages" / "execution" / f"{task['case_id']}.json"
        previous = _read_json(stage_path) if stage_path.exists() else None
        relation = HYBRIDS[task["algorithm"]].problem
        base = {
            "stage": "execution", "case_id": task["case_id"],
            "algorithm": task["algorithm"], "code": task["code"],
            "population": task["population"], "seed": task["seed"],
            "input_id": task["input_id"], "timeout_seconds": budget["timeout_seconds"],
            "memory_limit_bytes": budget["memory_limit_bytes"],
        }
        generation_path = output / "stages" / "generation" / f"{task['input_id']}.json"
        generation = _read_json(generation_path) if generation_path.exists() else {}
        input_path = output / "inputs" / f"{task['input_id']}.json"
        cert_path = output / "stages" / "certification" / f"{task['input_id']}__{relation}.json"
        certification = _read_json(cert_path) if cert_path.exists() else {}
        dependency_recovered = (
            previous is not None
            and previous.get("status") in {"blocked_generation", "blocked_certification"}
            and generation.get("status") == "success"
            and input_path.exists()
            and (
                task["population"] == "positive_control"
                or (
                    certification.get("status") == "success"
                    and certification.get("label") == "inequivalent"
                )
            )
        )
        if (
            previous is not None
            and not dependency_recovered
            and not _needs_retry(previous, "execution", retries)
        ):
            continue
        if not task["applicable"]:
            record = _stage_record(stage_path, {
                **base, "status": "not_applicable", "runtime_seconds": 0.0,
                "decision": None, "expected": None, "correct": None, "trace": [],
                "decided_by": "", "stuck_at": "",
                "error": "PM-CSS requires CSS inputs",
            })
        else:
            if generation.get("status") != "success":
                record = _stage_record(stage_path, {
                    **base, "status": "blocked_generation", "runtime_seconds": 0.0,
                    "decision": None, "expected": None, "correct": None,
                    "trace": [], "decided_by": "", "stuck_at": "",
                    "error": "input generation did not succeed",
                })
            elif task["population"] == "certified_negative" and (
                certification.get("status") != "success"
                or certification.get("label") != "inequivalent"
                or not input_path.exists()
            ):
                record = _stage_record(stage_path, {
                    **base, "status": "blocked_certification", "runtime_seconds": 0.0,
                    "decision": None, "expected": False, "correct": None,
                    "trace": [], "decided_by": "", "stuck_at": "",
                    "certification_label": certification.get("label", "unresolved"),
                    "error": "negative input was not certified inequivalent",
                })
            elif not input_path.exists():
                record = _stage_record(stage_path, {
                    **base, "status": "blocked_generation", "runtime_seconds": 0.0,
                    "decision": None, "expected": None, "correct": None,
                    "trace": [], "decided_by": "", "stuck_at": "",
                    "error": "persisted input is missing",
                })
            else:
                expected = task["population"] == "positive_control"
                label = "equivalent_by_construction" if expected else "inequivalent"
                next_attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
                trace_path = output / "traces" / f"{task['case_id']}.attempt-{next_attempt}.log"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                result = run(
                    TracedHybrid(
                        HYBRIDS[task["algorithm"]].function,
                        str(trace_path), str(input_path), task["input_id"],
                    ),
                    (), expected, timeout=budget["timeout_seconds"],
                    max_memory_bytes=budget["memory_limit_bytes"],
                )
                status = _supervision_status(result)
                decision = result.result if isinstance(result.result, bool) else None
                correct = decision == expected if status == "success" and expected is not None else None
                if correct is False:
                    status = "incorrect"
                trace, decided_by = _read_trace(trace_path)
                stuck_at = (trace[-1] if trace else UNREACHED) if status in {"timeout", "memory_limited"} else ""
                record = _stage_record(stage_path, {
                    **base, "status": status, "runtime_seconds": result.runtime,
                    "decision": decision, "expected": expected, "correct": correct,
                    "certification_label": label, "trace": trace,
                    "decided_by": decided_by, "stuck_at": stuck_at,
                    "trace_path": str(trace_path.relative_to(output)),
                    "error": result.error or "",
                })
        events.write("execution", task["case_id"], record)
        guard.observe("execution", str(record["status"]), str(record.get("error", "")))
        if verbose:
            detail = record.get("decided_by") or record.get("stuck_at") or record.get("error", "")
            print(f"execution {task['case_id']}: {record['status']} {float(record['runtime_seconds']):.3f}s {detail}", flush=True)
        if record["status"] == "incorrect":
            raise RuntimeError(f"incorrect hybrid answer for {task['case_id']}: expected={record['expected']} decision={record['decision']}")


def _distribution(values: Iterable[str]) -> str:
    counts = Counter(value for value in values if value)
    return ";".join(f"{key}:{count}" for key, count in
                    sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _number(value: float | None) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.9f}"


def _dimensions(output: Path, configuration: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    result: dict[str, tuple[Any, Any]] = {}
    for name in configuration["codes"]:
        result[name] = ("", "")
        for spec in _input_specs(configuration):
            if spec["code"] != name:
                continue
            stage_path = output / "stages" / "generation" / f"{spec['input_id']}.json"
            if not stage_path.exists():
                continue
            stage = _read_json(stage_path)
            relative = stage.get("input_path")
            if stage.get("status") != "success" or not relative:
                continue
            try:
                payload = _read_input(output / str(relative), str(spec["input_id"]))
            except (OSError, ValueError):
                continue
            result[name] = (int(payload["left"]["n"]), int(payload["left"]["k"]))
            break
    return result


def materialize(output: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Atomically rebuild instance and summary CSV views from durable JSON state."""
    configuration = manifest["configuration"]
    campaign_id = str(manifest["campaign_id"])
    dimensions = _dimensions(output, configuration)
    instances: list[dict[str, Any]] = []
    for task in _execution_tasks(configuration):
        algorithm, code = task["algorithm"], task["code"]
        relation = HYBRIDS[algorithm].problem
        n, k = dimensions[code]
        generation_path = output / "stages" / "generation" / f"{task['input_id']}.json"
        certification_path = output / "stages" / "certification" / f"{task['input_id']}__{relation}.json"
        execution_path = output / "stages" / "execution" / f"{task['case_id']}.json"
        generation = _read_json(generation_path) if generation_path.exists() else {}
        certification = _read_json(certification_path) if certification_path.exists() else {}
        execution = _read_json(execution_path) if execution_path.exists() else {}
        if not task["applicable"]:
            certification_status, certification_label = "not_applicable", "not_applicable"
            certification_method = ""
            current_expected = None
        elif task["population"] == "positive_control":
            certification_status = "not_required"
            certification_label = "equivalent_by_construction"
            certification_method = "recorded_construction_witness"
            current_expected = True
        else:
            certification_status = str(certification.get("status", "pending"))
            certification_label = str(certification.get("label", "unresolved"))
            certification_method = str(certification.get("method", ""))
            current_expected = (
                False
                if certification_status == "success"
                and certification_label == "inequivalent"
                else None
            )
        decision = execution.get("decision")
        current_correct = (
            decision == current_expected
            if execution.get("status") == "success"
            and isinstance(decision, bool)
            and current_expected is not None
            else execution.get("correct", "")
        )
        instances.append({
            "campaign_id": campaign_id, "algorithm": algorithm, "problem": relation,
            "code": code, "n": n, "k": k, "population": task["population"],
            "seed": task["seed"], "applicable": task["applicable"],
            "input_id": task["input_id"], "input_status": generation.get("status", "pending"),
            "certification_status": certification_status,
            "certification_label": certification_label,
            "certification_method": certification_method,
            "num_certified_equivalent_proposals": int(
                certification.get("num_certified_equivalent_proposals", 0)
            ),
            "certification_runtime_seconds": _number(certification.get("runtime_seconds")),
            "certification_timeout_seconds": certification.get("timeout_seconds", configuration["budgets"]["certification"]["timeout_seconds"]),
            "certification_memory_limit_bytes": certification.get("memory_limit_bytes", configuration["budgets"]["certification"]["memory_limit_bytes"]),
            "execution_status": execution.get("status", "not_applicable" if not task["applicable"] else "pending"),
            "runtime_seconds": _number(execution.get("runtime_seconds")),
            "execution_timeout_seconds": execution.get("timeout_seconds", configuration["budgets"]["execution"]["timeout_seconds"]),
            "execution_memory_limit_bytes": execution.get("memory_limit_bytes", configuration["budgets"]["execution"]["memory_limit_bytes"]),
            "expected_decision": current_expected if current_expected is not None else "",
            "decision": decision if decision is not None else "", "correct": current_correct,
            "decided_by": execution.get("decided_by", ""),
            "stuck_at": execution.get("stuck_at", ""),
            "trace": ">".join(execution.get("trace", [])),
            "error": execution.get("error", generation.get("error", certification.get("error", ""))),
            "execution_attempt": execution.get("attempt", 0),
        })

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        groups[(row["algorithm"], row["code"], row["population"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (algorithm, code, population), rows in groups.items():
        applicable = rows[0]["applicable"]
        successful_times = [float(row["runtime_seconds"]) for row in rows
                            if row["execution_status"] == "success" and row["runtime_seconds"] != ""]
        observed_times = [float(row["runtime_seconds"]) for row in rows
                          if row["runtime_seconds"] != "" and row["execution_status"]
                          not in {"pending", "blocked_generation", "blocked_certification", "not_applicable"}]
        attempted = sum(row["execution_status"] not in
                        {"pending", "blocked_generation", "blocked_certification", "not_applicable"} for row in rows)
        failures = sum(row["execution_status"] in
                       {"timeout", "memory_limited", "error", "incorrect"} for row in rows)
        certification_issues = sum(
            row["population"] == "certified_negative"
            and (
                row["certification_label"] != "inequivalent"
                or row["certification_status"] != "success"
            )
            for row in rows
        )
        pending = sum(row["execution_status"] in
                      {"pending", "blocked_generation", "blocked_certification"} for row in rows)
        status = ("not_applicable" if not applicable else "incomplete" if pending
                  else "issues" if failures or certification_issues else "complete")
        summaries.append({
            "campaign_id": campaign_id, "algorithm": algorithm,
            "problem": HYBRIDS[algorithm].problem, "code": code,
            "n": rows[0]["n"], "k": rows[0]["k"], "population": population,
            "applicable": applicable, "status": status,
            "num_requested": len(rows) if applicable else 0,
            "num_generated": sum(row["input_status"] == "success" for row in rows),
            "num_generation_failures": sum(row["input_status"] in
                                             {"timeout", "memory_limited", "error"} for row in rows),
            "num_certified_equivalent": sum(
                int(row["num_certified_equivalent_proposals"]) for row in rows
            ),
            "num_certified_inequivalent": sum(row["certification_label"] == "inequivalent" for row in rows),
            "num_unresolved_labels": sum(bool(applicable) and row["certification_label"] == "unresolved" for row in rows),
            "num_certification_failures": sum(row["certification_status"] in
                                                {"timeout", "memory_limited", "error", "blocked_generation"} for row in rows),
            "num_execution_attempted": attempted,
            "num_successful": sum(row["execution_status"] == "success" for row in rows),
            "num_correct": sum(row["correct"] is True for row in rows),
            "num_incorrect": sum(row["execution_status"] == "incorrect" for row in rows),
            "num_timeouts": sum(row["execution_status"] == "timeout" for row in rows),
            "num_memory_limited": sum(row["execution_status"] == "memory_limited" for row in rows),
            "num_errors": sum(row["execution_status"] == "error" for row in rows),
            "num_blocked": pending,
            "coverage_fraction": _number(attempted / len(rows)) if applicable and rows else "",
            "mean_success_seconds": _number(mean(successful_times)) if successful_times else "",
            "restricted_mean_seconds": _number(mean(observed_times)) if observed_times else "",
            "stddev_success_seconds": (_number(stdev(successful_times)) if len(successful_times) > 1
                                       else "0.000000000" if successful_times else ""),
            "maximum_seconds": _number(max(observed_times)) if observed_times else "",
            "deciders": _distribution(str(row["decided_by"]) for row in rows),
            "stuck_at": _distribution(str(row["stuck_at"]) for row in rows),
            "execution_timeout_seconds": configuration["budgets"]["execution"]["timeout_seconds"],
            "execution_memory_limit_bytes": configuration["budgets"]["execution"]["memory_limit_bytes"],
        })
    code_order = {name: index for index, name in enumerate(configuration["codes"])}
    algorithm_order = {name: index for index, name in enumerate(configuration["algorithms"])}
    population_order = {name: index for index, name in enumerate(POPULATIONS)}
    instances.sort(key=lambda row: (int(row["seed"]), population_order[row["population"]],
                                    code_order[row["code"]], algorithm_order[row["algorithm"]]))
    summaries.sort(key=lambda row: (code_order[row["code"]], algorithm_order[row["algorithm"]],
                                    population_order[row["population"]]))
    _write_csv(output / "instances.csv", instances, INSTANCE_FIELDS)
    _write_csv(output / "summary.csv", summaries, SUMMARY_FIELDS)
    return instances, summaries


def collect_campaign(output_directory: Path, configuration: Mapping[str, Any], *,
                     retries: set[str], max_systematic_errors: int = 3,
                     verbose: bool = True) -> dict[str, Any]:
    output_directory = output_directory.resolve()
    with CampaignLock(output_directory):
        manifest = _prepare_manifest(output_directory, configuration)
        events = EventLog(output_directory / "events.jsonl", str(manifest["campaign_id"]))
        guard = FailureGuard(max_systematic_errors)
        try:
            _collect_generation(output_directory, configuration, retries, events, guard, verbose)
            materialize(output_directory, manifest)
            _collect_certification(output_directory, configuration, retries, events, guard, verbose)
            materialize(output_directory, manifest)
            _collect_execution(output_directory, configuration, retries, events, guard, verbose)
        finally:
            materialize(output_directory, manifest)
        return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", nargs="+", default=list(REQUESTED_CODES), choices=REQUESTED_CODES)
    parser.add_argument("--algorithms", nargs="+", default=list(HYBRIDS), choices=tuple(ALGORITHM_ALIASES))
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--seed-upper-bound", type=int, default=SEED_UPPER_BOUND)
    parser.add_argument("--perturbation-depth", type=int, default=PERTURBATION_DEPTH)
    parser.add_argument("--negative-max-attempts", type=int, default=NEGATIVE_MAX_ATTEMPTS)
    parser.add_argument("--presentation-depth", type=int, default=PRESENTATION_DEPTH)
    parser.add_argument("--generation-timeout", type=float)
    parser.add_argument("--certification-timeout", type=float)
    parser.add_argument("--execution-timeout", type=float)
    parser.add_argument("--generation-memory-gib", type=float)
    parser.add_argument("--certification-memory-gib", type=float)
    parser.add_argument("--execution-memory-gib", type=float)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry", nargs="*", choices=("generation", "certification", "execution"), default=[])
    parser.add_argument("--max-systematic-errors", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        configuration = _build_configuration(args)
        default_output = PREFLIGHT_OUTPUT_DIRECTORY if args.preflight else OUTPUT_DIRECTORY
        output = (args.output_directory or default_output).resolve()
        if args.preflight and output == OUTPUT_DIRECTORY.resolve():
            raise ValueError("preflight output must be isolated from the production directory")
        manifest = collect_campaign(
            output, configuration, retries=set(args.retry),
            max_systematic_errors=args.max_systematic_errors, verbose=not args.quiet,
        )
        print(f"A8 campaign materialized at {output} (campaign {manifest['campaign_id']})", flush=True)
        return 0
    except KeyboardInterrupt:
        print("A8 collection interrupted; durable completed stages were retained",
              file=sys.stderr, flush=True)
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"A8 collection failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
