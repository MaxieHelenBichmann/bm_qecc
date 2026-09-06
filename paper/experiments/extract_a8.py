"""Extract the A8 campaign summary without hiding censored or missing cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper.experiments.common import RESULTS_DIR, as_bool, as_float, as_int, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIRECTORY = ROOT / "paper" / "data" / "collected" / "hybrids_a8_v4"
OUTPUT = RESULTS_DIR / "a8" / "by_cell.csv"

ALGORITHMS = ("pm_stb_hybrid", "pm_css_hybrid", "lc_stb_hybrid")
PROBLEMS = {"pm_stb_hybrid": "pm_stb", "pm_css_hybrid": "pm_css", "lc_stb_hybrid": "lc_stb"}
POPULATIONS = ("positive_control", "certified_negative")

CODE_LABELS = (
    ("bell", "Bell pair"), ("3q_rep", "3-bit repetition"),
    ("5q_prf", "5-qubit perfect"), ("steane", "Steane"), ("shor", "Shor"),
    ("carbon", "Carbon"), ("hamming_15", "Hamming-15"),
    ("15q_optimal", "15-qubit optimal"), ("tetrahedral", "Tetrahedral"),
    ("golay", "Golay"), ("rot_surf_d5", "Rotated surface d=5"),
    ("hamming_31", "Hamming-31"), ("coco_488", "CoCo-488"),
    ("coco_666", "CoCo-666"), ("bb_72", "BB-72"), ("bb_90", "BB-90"),
    ("bb_108", "BB-108"), ("bb_144", "BB-144"),
)
CODE_ORDER = {name: index for index, (name, _) in enumerate(CODE_LABELS)}
DISPLAY_NAME = dict(CODE_LABELS)
COMPONENT_ORDER = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE", "trivial")
COMPONENT_INDEX = {component: index for index, component in enumerate(COMPONENT_ORDER)}

REQUIRED = (
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

FIELDS = (
    "campaign_id", "problem", "algorithm", "code", "code_label", "n", "k", "r",
    "population", "applicable", "status", "runtime_statistic",
    "restricted_mean_seconds", "mean_success_seconds", "stddev_success_seconds",
    "maximum_seconds", "primary_decider", "primary_decider_count", "num_decisions",
    "primary_decider_fraction", "deciders", "stuck_at", "num_requested",
    "num_generated", "num_generation_failures", "num_certified_equivalent",
    "num_certified_inequivalent", "num_unresolved_labels",
    "num_certification_failures", "num_execution_attempted", "num_successful",
    "num_correct", "num_incorrect", "num_timeouts", "num_memory_limited",
    "num_errors", "num_blocked", "coverage_fraction", "certification_coverage_fraction",
    "execution_success_fraction", "execution_timeout_seconds",
    "execution_memory_limit_bytes", "issue_summary",
)


def parse_distribution(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not value.strip():
        return counts
    for item in value.split(";"):
        try:
            tag, raw_count = item.rsplit(":", 1)
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid hybrid component distribution {value!r}") from exc
        tag = tag.strip()
        if not tag or count < 0:
            raise ValueError(f"invalid hybrid component distribution {value!r}")
        counts[tag] = counts.get(tag, 0) + count
    return counts


def main_decider(value: str) -> tuple[str, int, int]:
    counts = parse_distribution(value)
    if not counts:
        return "", 0, 0
    largest = max(counts.values())
    tied = sorted((tag for tag, count in counts.items() if count == largest),
                  key=lambda tag: (COMPONENT_INDEX.get(tag, len(COMPONENT_INDEX)), tag))
    return "/".join(tied), largest, sum(counts.values())


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _issues(row: dict[str, str]) -> str:
    labels = (
        ("gen", as_int(row["num_generation_failures"])),
        ("cert", as_int(row["num_certification_failures"])),
        ("unresolved", as_int(row["num_unresolved_labels"])),
        ("timeout", as_int(row["num_timeouts"])),
        ("memory", as_int(row["num_memory_limited"])),
        ("error", as_int(row["num_errors"])),
        ("incorrect", as_int(row["num_incorrect"])),
        ("blocked", as_int(row["num_blocked"])),
    )
    return ";".join(f"{label}:{count}" for label, count in labels if count)


def _manifest_selection(input_directory: Path) -> tuple[list[str], list[str]] | None:
    path = input_directory / "campaign.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    configuration = manifest.get("configuration", {})
    return list(configuration.get("codes", [])), list(configuration.get("algorithms", []))


def extract(input_directory: Path = INPUT_DIRECTORY,
            output_file: Path = OUTPUT) -> list[dict[str, Any]]:
    source = input_directory / "summary.csv"
    raw = read_csv(source, REQUIRED)
    if not raw:
        raise ValueError(f"{source} contains no A8 summary rows")
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in raw:
        if row["algorithm"] not in ALGORITHMS:
            raise ValueError(f"unknown A8 algorithm {row['algorithm']!r}")
        if row["problem"] != PROBLEMS[row["algorithm"]]:
            raise ValueError(f"problem/algorithm mismatch in {source}")
        if row["population"] not in POPULATIONS:
            raise ValueError(f"unknown A8 population {row['population']!r}")
        latest[(row["algorithm"], row["code"], row["population"])] = row

    selection = _manifest_selection(input_directory)
    if selection is not None:
        codes, algorithms = selection
        expected = {(algorithm, code, population) for algorithm in algorithms
                    for code in codes for population in POPULATIONS}
        missing = expected - set(latest)
        if missing:
            raise ValueError(f"{source} is missing {len(missing)} materialized campaign cells")

    output: list[dict[str, Any]] = []
    integer_fields = (
        "num_requested", "num_generated", "num_generation_failures",
        "num_certified_equivalent", "num_certified_inequivalent",
        "num_unresolved_labels", "num_certification_failures",
        "num_execution_attempted", "num_successful", "num_correct", "num_incorrect",
        "num_timeouts", "num_memory_limited", "num_errors", "num_blocked",
    )
    for (algorithm, code, population), row in latest.items():
        decider, decider_count, num_decisions = main_decider(row["deciders"])
        values = {field: as_int(row[field]) for field in integer_fields}
        n, k = as_int(row["n"]), as_int(row["k"])
        applicable = as_bool(row["applicable"])
        requested = values["num_requested"]
        output.append({
            "campaign_id": row["campaign_id"], "problem": row["problem"],
            "algorithm": algorithm, "code": code,
            "code_label": DISPLAY_NAME.get(code, code.replace("_", " ")),
            "n": n, "k": k, "r": n - k, "population": population,
            "applicable": applicable, "status": row["status"],
            "runtime_statistic": "mean min(runtime, execution budget), including failed attempts with runtimes",
            "restricted_mean_seconds": as_float(row["restricted_mean_seconds"]),
            "mean_success_seconds": as_float(row["mean_success_seconds"]),
            "stddev_success_seconds": as_float(row["stddev_success_seconds"]),
            "maximum_seconds": as_float(row["maximum_seconds"]),
            "primary_decider": decider, "primary_decider_count": decider_count,
            "num_decisions": num_decisions,
            "primary_decider_fraction": _fraction(decider_count, num_decisions),
            "deciders": row["deciders"], "stuck_at": row["stuck_at"], **values,
            "coverage_fraction": as_float(row["coverage_fraction"]),
            "certification_coverage_fraction": (
                _fraction(values["num_certified_inequivalent"], requested)
                if population == "certified_negative" else None
            ),
            "execution_success_fraction": _fraction(values["num_successful"], requested),
            "execution_timeout_seconds": as_float(row["execution_timeout_seconds"]),
            "execution_memory_limit_bytes": as_int(row["execution_memory_limit_bytes"]),
            "issue_summary": _issues(row),
        })

    output.sort(key=lambda row: (
        CODE_ORDER.get(row["code"], len(CODE_ORDER)), row["code"],
        ALGORITHMS.index(row["algorithm"]), POPULATIONS.index(row["population"]),
    ))
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=INPUT_DIRECTORY)
    parser.add_argument("--output-file", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    extract(arguments.input_directory, arguments.output_file)
