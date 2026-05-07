"""Command-line benchmark runner."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.algorithms.lc_css_bruteforce import is_lceq_css_bruteforce
from src.algorithms.lc_css_graph_state import is_lceq_css_graph_state
from src.algorithms.lc_css_kls import is_lceq_css_kls
from src.algorithms.lc_css_orbit import is_lceq_css_orbit
from src.algorithms.lc_eq_graph_state import are_lceq_graph_state
from src.algorithms.p_css_bruteforce import are_peq_css_bruteforce
from src.algorithms.p_css_classical import are_peq_css_classical
from src.algorithms.p_css_graph_iso import are_peq_css_graph_iso
from src.algorithms.p_css_matroid import are_peq_css_matroid

from src.core.stabilizer_code import StabilizerCode
from src.core.css_code import CSSCode


@dataclass(frozen=True)
class Case:
    """One benchmark input."""

    name: str
    problem: str
    inputs: tuple[StabilizerCode, ...]
    expected_p: bool | None = None
    expected_lc: bool | None = None


@dataclass(frozen=True)
class Result:
    """One benchmark measurement."""

    algorithm: str
    case: str
    problem: str
    n: int
    k: int
    seconds: float
    result: bool | None
    expected: bool | None
    success: bool
    error: str = ""


Algorithm = Callable[..., bool]

ALGORITHMS: dict[str, dict[str, tuple[Algorithm, str]]] = {
    "equivalence": {
        "p_bruteforce": (are_peq_css_bruteforce, "p"),
        "p_classical": (are_peq_css_classical, "p"),
        "p_graph_iso": (are_peq_css_graph_iso, "p"),
        "p_matroid": (are_peq_css_matroid, "p"),
        "lc_eq_graph_state": (are_lceq_graph_state, "lc")
    },
    "search": {
        "lc_bruteforce": (is_lceq_css_bruteforce, "lc"),
        "lc_graph_state": (is_lceq_css_graph_state, "lc"),
        "lc_kls": (is_lceq_css_kls, "lc"),
        "lc_orbit": (is_lceq_css_orbit, "lc"),
    },
}


def all_algorithm_names() -> list[str]:
    """Return all registered algorithm names."""
    return sorted({name for algorithms in ALGORITHMS.values() for name in algorithms})


def default_cases() -> list[Case]:
    """Return tiny smoke-test cases."""
    return [
        Case(
            name="bell_pair_same",
            problem="equivalence",
            inputs=(StabilizerCode(["ZZ"]), StabilizerCode(["ZZ"])),
            expected_p=True,
            expected_lc=True,
        ),
        Case(
            name="three_qubit_repetition_reordered_generators",
            problem="equivalence",
            inputs=(StabilizerCode(["ZZI", "IZZ"]), StabilizerCode(["IZZ", "ZZI"])),
            expected_p=True,
            expected_lc=True,
        ),
        Case(
            name="single_z_not_weight_two",
            problem="equivalence",
            inputs=(StabilizerCode(["ZII"]), StabilizerCode(["ZZI"])),
            expected_p=False,
            expected_lc=False,
        ),
    ]


def run_case(algorithm_name: str, algorithm: Algorithm, problem_type: str, case: Case, repeats: int) -> Result:
    """Run one algorithm on one case and return the average runtime."""
    total_seconds = 0.0
    last_result: bool | None = None

    try:
        for _ in range(repeats):
            start = perf_counter()
            last_result = algorithm(*case.inputs)
            total_seconds += perf_counter() - start
        success = (None if case.expected_p is None and case.expected_lc is None else (last_result == case.expected_p)) if problem_type == "p" else (None if case.expected_lc is None else (last_result == case.expected_lc))
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            problem=case.problem,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=total_seconds / repeats,
            result=last_result,
            expected=case.expected_p if problem_type == "p" else case.expected_lc,
            success=success,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            algorithm=algorithm_name,
            case=case.name,
            problem=case.problem,
            n=case.inputs[0].n,
            k=case.inputs[0].k,
            seconds=0.0,
            result=None,
            expected=case.expected_p if problem_type == "p" else case.expected_lc,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_benchmarks(cases: Sequence[Case], algorithm_names: Sequence[str], repeats: int) -> list[Result]:
    """Run selected algorithms on cases with the matching problem type."""
    results: list[Result] = []
    selected_names = set(algorithm_names)

    for case in cases:
        algorithms = ALGORITHMS[case.problem]
        for algorithm_name in sorted(selected_names & algorithms.keys()):
            algorithm, problem_type = algorithms[algorithm_name]
            results.append(run_case(algorithm_name, algorithm, problem_type, case, repeats))
    return results


def write_csv(results: Sequence[Result], output: Path) -> None:
    """Write benchmark results to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "algorithm": result.algorithm,
            "case": result.case,
            "problem": result.problem,
            "n": result.n,
            "k": result.k,
            "seconds": f"{result.seconds:.9f}",
            "result": "" if result.result is None else result.result,
            "expected": "" if result.expected is None else result.expected,
            "success": result.success,
            "error": result.error,
        }
        for result in results
    ]
    if not rows:
        return

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions; the average runtime is recorded.")
    parser.add_argument(
        "--algorithm",
        choices=all_algorithm_names(),
        action="append",
        help="Algorithm to run. Can be passed multiple times. Defaults to all implemented algorithms.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/latest.csv"), help="CSV output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    algorithm_names = args.algorithm or all_algorithm_names()
    results = run_benchmarks(default_cases(), algorithm_names, args.repeats)
    write_csv(results, args.output)

    for result in results:
        status = "ok" if result.success else "failed"
        print(
            f"{status:6} {result.problem:11} {result.algorithm:24} {result.case:42} "
            f"n={result.n:<2} k={result.k:<2} t={result.seconds:.6f}s result={result.result}"
        )
        if result.error:
            print(f"       {result.error}")
    print(f"Wrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
