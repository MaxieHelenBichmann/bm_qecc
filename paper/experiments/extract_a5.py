"""Choose the fastest completed algorithm with no memory or general errors, with a timeout-only fallback."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from paper.experiments.common import (
    ALGORITHM_DATA_DIR,
    RESULTS_DIR,
    aggregate_statistics,
    load_algorithm,
    write_csv,
)

A5_ALGORITHMS = (
    "pm_stb_aut",
    "pm_stb_bruteforce",
    "pm_stb_classical",
    "pm_stb_graph_iso",
    "pm_stb_sat",
    "pm_css_bruteforce",
    "pm_css_classical",
    "pm_css_matroid",
    "pm_css_sat",
    "lc_stb_lse",
    "lc_stb_bruteforce",
    "lc_stb_graph_iso",
    "lc_stb_kls",
    "lc_stb_sat",
)

OUTPUT_DIRECTORY = RESULTS_DIR / "a5"
METHOD_FIELDS = (
    "problem", "algorithm", "n", "k", "r", "num_requested", "num_successful",
    "mean_seconds", "stddev_seconds", "maximum_seconds", "num_unexpected",
    "num_timeouts", "num_memory_limited", "num_errors", "num_generation_errors",
    "complete",
)
WINNER_FIELDS = (
    "problem", "n", "k", "r", "winner", "mean_seconds", "runner_up",
    "runner_up_mean_seconds", "speed_ratio", "num_eligible_algorithms",
    "selection", "winner_num_timeouts",
)


def timeout_candidate(cell: dict[str, Any]) -> bool:
    """Whether a paired cell failed only by reaching the runtime limit."""
    return (
        cell["has_positive"]
        and cell["has_negative"]
        and cell["mean_seconds"] is not None
        and cell["num_timeouts"] > 0
        and cell["num_successful"] + cell["num_timeouts"] == cell["num_requested"]
        and not any(
            cell[field]
            for field in (
                "num_unexpected",
                "num_memory_limited",
                "num_errors",
                "num_generation_errors",
            )
        )
    )


def select_winners(methods: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the fastest backend per parameter cell, as A5 defines "best".

    A3 pairs invariants against exactly these winners, so the choice lives here
    rather than being reimplemented against a slightly different rule.
    """
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in methods:
        groups[(cell["problem"], cell["n"], cell["k"])].append(cell)
    winners = []
    for (problem, n, k), methods_in_cell in sorted(groups.items()):
        completed = [
            cell
            for cell in methods_in_cell
            if cell["complete"] and cell["mean_seconds"] is not None
        ]
        if completed:
            ordered = sorted(completed, key=lambda cell: cell["mean_seconds"])
            selection = "completed"
        else:
            timed_out = [cell for cell in methods_in_cell if timeout_candidate(cell)]
            if not timed_out:
                continue
            ordered = sorted(timed_out, key=lambda cell: cell["mean_seconds"])
            selection = "timeout_fallback"
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        winners.append({
            "problem": problem, "n": n, "k": k, "r": n - k,
            "winner": winner["algorithm"], "mean_seconds": winner["mean_seconds"],
            "runner_up": runner_up["algorithm"] if runner_up else "",
            "runner_up_mean_seconds": runner_up["mean_seconds"] if runner_up else "",
            "speed_ratio": runner_up["mean_seconds"] / winner["mean_seconds"] if runner_up and winner["mean_seconds"] else "",
            "num_eligible_algorithms": len(ordered),
            "selection": selection,
            "winner_num_timeouts": winner["num_timeouts"],
        })
    return winners


def extract(
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    output_directory: Path = OUTPUT_DIRECTORY,
    algorithm_names: Sequence[str] = A5_ALGORITHMS,
) -> list[dict[str, Any]]:
    rows = [
        row
        for algorithm in algorithm_names
        for row in load_algorithm(algorithm, algorithm_directory)
    ]
    methods = aggregate_statistics(rows)
    winners = select_winners(methods)
    write_csv(output_directory / "by_method.csv", methods, METHOD_FIELDS)
    write_csv(output_directory / "by_cell.csv", winners, WINNER_FIELDS)
    return winners


if __name__ == "__main__":
    extract()
