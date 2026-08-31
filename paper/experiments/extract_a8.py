"""Extract the structured-code hybrid table used by A8."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from paper.experiments.common import (
    COLLECTED_DATA_DIR,
    RESULTS_DIR,
    as_bool,
    as_float,
    as_int,
    read_csv,
    write_csv,
)

INPUT_DIRECTORY = COLLECTED_DATA_DIR / "hybrids"
OUTPUT = RESULTS_DIR / "a8" / "by_cell.csv"

ALGORITHMS = (
    "pm_stb_hybrid",
    "pm_css_hybrid",
    "lc_stb_hybrid",
)
EXCLUDED_CODES = {"gottesman"}
PROBLEMS = {
    "pm_stb_hybrid": "pm_stb",
    "pm_css_hybrid": "pm_css",
    "lc_stb_hybrid": "lc_stb",
}

# Stable registry order and publication-friendly labels. Unknown future named
# codes are retained after these entries instead of silently disappearing.
CODE_LABELS = (
    ("bell", "Bell pair"),
    ("3q_rep", "3-bit repetition"),
    ("5q_prf", "5-qubit perfect"),
    ("steane", "Steane"),
    ("gottesman", "8-qubit Gottesman"),
    ("shor", "Shor"),
    ("carbon", "Carbon"),
    ("tetrahedral", "Tetrahedral"),
    ("15q_optimal", "15-qubit optimal"),
    ("hamming_15", "Hamming-15"),
    ("golay", "Golay"),
    ("rot_surf_d5", "Rotated surface d=5"),
    ("bring", "Bring"),
    ("coco_488", "CoCo-488"),
    ("hamming_31", "Hamming-31"),
    ("coco_666", "CoCo-666"),
    ("bb_72", "BB-72"),
    ("bb_90", "BB-90"),
    ("bb_108", "BB-108"),
    ("bb_144", "BB-144"),
)
CODE_ORDER = {name: index for index, (name, _) in enumerate(CODE_LABELS)}
DISPLAY_NAME = dict(CODE_LABELS)

COMPONENT_ORDER = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE")
COMPONENT_INDEX = {component: index for index, component in enumerate(COMPONENT_ORDER)}

REQUIRED = (
    "algorithm",
    "name",
    "n",
    "k",
    "positive",
    "timeout_seconds",
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
    "deciders",
    "stuck_at",
)

FIELDS = (
    "problem",
    "algorithm",
    "code",
    "code_label",
    "n",
    "k",
    "r",
    "positive",
    "mean_seconds",
    "stddev_seconds",
    "maximum_seconds",
    "primary_decider",
    "primary_decider_count",
    "num_decisions",
    "primary_decider_fraction",
    "deciders",
    "stuck_at",
    "timeout_seconds",
    "num_cases",
    "num_completed",
    "num_expected",
    "num_unexpected",
    "num_timeouts",
    "num_memory_limited",
    "num_errors",
    "num_generation_errors",
)


def parse_distribution(value: str) -> dict[str, int]:
    """Parse the collector's semicolon-separated ``TAG:count`` encoding."""
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
    """Return display tag(s), winning count, and total recorded decisions."""
    counts = parse_distribution(value)
    if not counts:
        return "", 0, 0
    largest = max(counts.values())
    tied = sorted(
        (tag for tag, count in counts.items() if count == largest),
        key=lambda tag: (COMPONENT_INDEX.get(tag, len(COMPONENT_INDEX)), tag),
    )
    return "/".join(tied), largest, sum(counts.values())


def extract(
    input_directory: Path = INPUT_DIRECTORY,
    output_file: Path = OUTPUT,
) -> list[dict[str, Any]]:
    """Select the main decider for every available hybrid/code/polarity row."""
    # Append-only collector files can contain a repeated key after a manual
    # rerun. Dict assignment deliberately keeps the latest occurrence.
    latest: dict[tuple[str, str, bool], dict[str, str]] = {}
    for algorithm in ALGORITHMS:
        path = input_directory / f"{algorithm}.csv"
        for row in read_csv(path, REQUIRED):
            if row["algorithm"] != algorithm:
                raise ValueError(
                    f"{path} contains algorithm {row['algorithm']!r}; expected {algorithm!r}"
                )
            if row["name"] in EXCLUDED_CODES:
                continue
            key = (algorithm, row["name"], as_bool(row["positive"]))
            latest[key] = row

    output: list[dict[str, Any]] = []
    for (algorithm, code, positive), row in latest.items():
        decider, decider_count, num_decisions = main_decider(row["deciders"])
        n, k = as_int(row["n"]), as_int(row["k"])
        output.append(
            {
                "problem": PROBLEMS[algorithm],
                "algorithm": algorithm,
                "code": code,
                "code_label": DISPLAY_NAME.get(code, code.replace("_", " ")),
                "n": n,
                "k": k,
                "r": n - k,
                "positive": positive,
                "mean_seconds": as_float(row["mean_seconds"]),
                "stddev_seconds": as_float(row["stddev_seconds"]),
                "maximum_seconds": as_float(row["maximum_seconds"]),
                "primary_decider": decider,
                "primary_decider_count": decider_count,
                "num_decisions": num_decisions,
                "primary_decider_fraction": (
                    decider_count / num_decisions if num_decisions else None
                ),
                "deciders": row["deciders"],
                "stuck_at": row["stuck_at"],
                "timeout_seconds": as_float(row["timeout_seconds"]),
                **{
                    field: as_int(row[field])
                    for field in (
                        "num_cases",
                        "num_completed",
                        "num_expected",
                        "num_unexpected",
                        "num_timeouts",
                        "num_memory_limited",
                        "num_errors",
                        "num_generation_errors",
                    )
                },
            }
        )

    output.sort(
        key=lambda row: (
            CODE_ORDER.get(row["code"], len(CODE_ORDER)),
            row["code"],
            ALGORITHMS.index(row["algorithm"]),
            not row["positive"],
        )
    )
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=INPUT_DIRECTORY)
    parser.add_argument("--output-file", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    extract(arguments.input_directory, arguments.output_file)
