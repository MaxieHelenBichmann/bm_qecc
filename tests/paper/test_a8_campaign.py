from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import z3

from paper.benchmarks import collect_a8
from paper.experiments.extract_a8 import extract
from paper.hybrids.pm_css import preserved_d as css_distance_preserved
from paper.hybrids.pm_stb import preserved_d as stabilizer_distance_preserved
from paper.visualizations.visualize_a8 import render
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


def _configuration(*extra: str) -> dict[str, object]:
    args = collect_a8.parse_args([
        "--preflight", "--codes", "bell", "--algorithms", "pm-stb",
        "--seeds", "89", "--generation-timeout", "2",
        "--certification-timeout", "2", "--execution-timeout", "2",
        "--generation-memory-gib", "2", "--certification-memory-gib", "2",
        "--execution-memory-gib", "2", *extra,
    ])
    return collect_a8._build_configuration(args)


def test_population_and_applicability_matrix_is_exact() -> None:
    args = collect_a8.parse_args(["--preflight", "--seeds", "89"])
    configuration = collect_a8._build_configuration(args)
    assert tuple(configuration["codes"]) == collect_a8.REQUESTED_CODES
    assert "gottesman" not in configuration["codes"]
    assert "bring" not in configuration["codes"]
    tasks = collect_a8._execution_tasks(configuration)
    assert len(tasks) == 108
    assert sum(task["applicable"] for task in tasks) == 104
    not_applicable = {(task["code"], task["population"])
                      for task in tasks if not task["applicable"]}
    assert not_applicable == {
        (code, population) for code in collect_a8.NON_CSS_CODES
        for population in collect_a8.POPULATIONS
    }


def test_pm_hybrids_share_controls_but_have_relation_specific_negatives() -> None:
    args = collect_a8.parse_args([
        "--preflight", "--codes", "steane", "--algorithms", "pm-stb", "pm-css",
        "--seeds", "89",
    ])
    tasks = collect_a8._execution_tasks(collect_a8._build_configuration(args))
    controls = {task["input_id"] for task in tasks
                if task["population"] == "positive_control"}
    negatives = {task["input_id"] for task in tasks
                 if task["population"] == "certified_negative"}
    assert len(controls) == 1
    assert len(negatives) == 2


def test_campaign_task_order_is_code_major() -> None:
    args = collect_a8.parse_args([
        "--preflight", "--codes", "bell", "steane",
        "--algorithms", "pm-stb", "pm-css", "lc-stb", "--seeds", "89",
    ])
    configuration = collect_a8._build_configuration(args)

    execution = collect_a8._execution_tasks(configuration)
    assert [task["code"] for task in execution[:6]] == ["bell"] * 6
    assert [task["algorithm"] for task in execution[:3]] == list(
        configuration["algorithms"]
    )
    assert [task["code"] for task in execution[6:12]] == ["steane"] * 6

    certification = collect_a8._certification_tasks(configuration)
    assert [task["code"] for task in certification[:3]] == ["bell"] * 3
    assert [task["code"] for task in certification[3:6]] == ["steane"] * 3


def test_every_named_stabilizer_proposal_is_reproducible_valid_and_local() -> None:
    for code_name in collect_a8.REQUESTED_CODES:
        spec = {"input_id": f"negative_pm_stb__{code_name}__89",
                "mode": "negative_pm_stb", "relation": "pm_stb",
                "code": code_name, "seed": 89, "attempt": 0}
        first = collect_a8.generate_input_payload(spec, 2, 4)
        second = collect_a8.generate_input_payload(spec, 2, 4)
        assert first == second
        left, right = collect_a8._pair_from_payload(first)
        assert (left.n, left.k) == (right.n, right.k)
        assert len(first["operations"]) == 2
        assert left.distance is None and right.distance is None
        assert isinstance(right, StabilizerCode)
        assert all(operation["gate"] in {
            "H", "S", "Sdg", "X", "Y", "Z", "CX", "CZ", "SWAP"
        } for operation in first["operations"])


def test_css_proposal_preserves_css_structure_and_check_ranks() -> None:
    spec = {"input_id": "negative_pm_css__steane__89",
            "mode": "negative_pm_css", "relation": "pm_css",
            "code": "steane", "seed": 89, "attempt": 0}
    payload = collect_a8.generate_input_payload(spec, 2, 4)
    left, right = collect_a8._pair_from_payload(payload)
    assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
    assert not np.any((right.Hx @ right.Hz.T) % 2)
    assert (collect_a8._rank(left.Hx), collect_a8._rank(left.Hz)) == (
        collect_a8._rank(right.Hx), collect_a8._rank(right.Hz)
    )
    assert {operation["gate"] for operation in payload["operations"]} == {"CX"}


def test_large_css_proposal_remains_a_fixed_depth_named_code_perturbation() -> None:
    spec = {"input_id": "negative_pm_css__hamming_31__89",
            "mode": "negative_pm_css", "relation": "pm_css",
            "code": "hamming_31", "seed": 89, "attempt": 0}
    payload = collect_a8.generate_input_payload(spec, 2, 4)
    left, right = collect_a8._pair_from_payload(payload)

    assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
    assert len(payload["operations"]) == 2
    assert all(operation["gate"] == "CX" for operation in payload["operations"])
    assert payload["source"]["registry_name"] == "hamming_31"
    assert payload["inequivalence_witness"]["method"] in {
        "additive_collision_rank", "support_rank", "stabilizer_weight_enumerator"
    }
    assert isinstance(payload["inequivalence_witness"]["separates"], bool)


def test_matching_large_css_invariant_stays_unresolved_without_replacement(
    tmp_path: Path,
) -> None:
    args = collect_a8.parse_args([
        "--preflight", "--codes", "hamming_31", "--algorithms", "pm-css",
        "--seeds", "89",
    ])
    configuration = collect_a8._build_configuration(args)
    spec = collect_a8._input_specs(configuration)[1]
    assert spec["mode"] == "negative_pm_css"
    payload = collect_a8.generate_input_payload(spec, 2, 4)
    payload["inequivalence_witness"]["separates"] = False
    payload["construction_witness"] = None
    payload.pop("payload_sha256")
    payload["payload_sha256"] = collect_a8._digest(payload)
    proposal = collect_a8._proposal_path(tmp_path, str(spec["input_id"]), 0)
    collect_a8._atomic_json(proposal, payload)
    generation = tmp_path / "stages" / "generation" / f"{spec['input_id']}.json"
    collect_a8._atomic_json(generation, {
        "status": "success", "input_path": str(proposal.relative_to(tmp_path)),
    })

    collect_a8._collect_certification(
        tmp_path, configuration, set(),
        collect_a8.EventLog(tmp_path / "events.jsonl", "test-campaign"),
        collect_a8.FailureGuard(0), verbose=False,
    )

    record_path = next((tmp_path / "stages" / "certification").glob("*.json"))
    record = json.loads(record_path.read_text())
    assert record["status"] == "unresolved"
    assert record["label"] == "unresolved"
    assert record["num_proposals"] == 1
    assert not (tmp_path / "inputs" / f"{spec['input_id']}.json").exists()
    assert len(list((tmp_path / "proposals").rglob("*.json"))) == 1


def test_generation_persists_large_payload_and_returns_only_metadata(
    tmp_path: Path,
) -> None:
    spec = {"input_id": "positive_pm__bb_144__89", "mode": "positive_pm",
            "code": "bb_144", "seed": 89}
    input_path = tmp_path / "input.json"

    metadata = collect_a8.generate_input_file(spec, 2, 4, str(input_path))
    payload = collect_a8._read_input(input_path, str(spec["input_id"]))

    assert metadata == {
        "input_id": spec["input_id"],
        "payload_sha256": payload["payload_sha256"],
    }
    assert "left" not in metadata and "right" not in metadata
    assert input_path.stat().st_size > len(json.dumps(metadata)) * 100


def test_positive_controls_are_constructive_and_negative_proposals_are_certifiable(
    tmp_path: Path,
) -> None:
    pm_spec = {"input_id": "positive_pm__bell__89", "mode": "positive_pm",
               "code": "bell", "seed": 89}
    lc_spec = {"input_id": "positive_lc__bell__89", "mode": "positive_lc",
               "code": "bell", "seed": 89}
    perturb_spec = {"input_id": "negative_pm_stb__bell__89",
                    "mode": "negative_pm_stb", "relation": "pm_stb",
                    "code": "bell", "seed": 89, "attempt": 0}
    pm = collect_a8.generate_input_payload(pm_spec, 2, 4)
    lc = collect_a8.generate_input_payload(lc_spec, 2, 4)
    perturb = collect_a8.generate_input_payload(perturb_spec, 2, 4)
    assert pm["construction_witness"]
    assert lc["construction_witness"]
    assert collect_a8.certify_payload("pm_stb", perturb)["label"] in {
        "equivalent", "inequivalent"
    }
    perturb_path = tmp_path / "perturbation.json"
    collect_a8._atomic_json(perturb_path, perturb)
    assert collect_a8.certify_input_file(
        "pm_stb", str(perturb_path), str(perturb_spec["input_id"])
    ) == collect_a8.certify_payload("pm_stb", perturb)


def test_pm_css_certifier_policy_matches_invariant_rejection_benchmark() -> None:
    assert collect_a8._css_certifier_kind(47, 38) == "sat"
    assert collect_a8._css_certifier_kind(28, 18) == "matroid"
    assert collect_a8._css_certifier_kind(29, 19) == "invariant_witness"


def test_solver_unknown_is_retained_as_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnknownSolver:
        def check(self) -> z3.CheckSatResult:
            return z3.unknown

        def reason_unknown(self) -> str:
            return "test unknown"

    spec = {"input_id": "positive_pm__bell__89", "mode": "positive_pm",
            "code": "bell", "seed": 89}
    payload = collect_a8.generate_input_payload(spec, 2, 4)
    monkeypatch.setattr(collect_a8, "_build_peq_stab_sat_solver", lambda *args: UnknownSolver())
    result = collect_a8.certify_payload("pm_stb", payload)
    assert result == {"label": "unresolved", "solver_result": "unknown",
                      "reason_unknown": "test unknown", "method": "exact_pm_stb_sat"}


def test_equivalent_negative_proposal_advances_to_next_persisted_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration("--negative-max-attempts", "2")
    events = collect_a8.EventLog(tmp_path / "events.jsonl", "test-campaign")
    guard = collect_a8.FailureGuard(0)
    collect_a8._collect_generation(
        tmp_path, configuration, set(), events, guard, verbose=False
    )

    def certify_attempt(relation: str, input_path: str,
                        expected_input_id: str) -> dict[str, str]:
        payload = collect_a8._read_input(Path(input_path), expected_input_id)
        equivalent = int(payload["attempt"]) == 0
        return {
            "label": "equivalent" if equivalent else "inequivalent",
            "solver_result": "sat" if equivalent else "unsat",
            "reason_unknown": "", "method": f"exact_{relation}_sat",
        }

    monkeypatch.setattr(collect_a8, "certify_input_file", certify_attempt)
    collect_a8._collect_certification(
        tmp_path, configuration, set(), events, guard, verbose=False
    )

    proposals = sorted((tmp_path / "proposals").rglob("*.json"))
    assert len(proposals) == 2
    record = next((tmp_path / "stages" / "certification").glob("*.json"))
    certification = json.loads(record.read_text())
    assert certification["label"] == "inequivalent"
    assert certification["selected_attempt"] == 1
    assert certification["num_proposals"] == 2
    assert certification["num_certified_equivalent_proposals"] == 1


def test_unavailable_distance_never_becomes_a_false_witness() -> None:
    left = StabilizerCode.get_trivial_code(2)
    right = StabilizerCode.get_trivial_code(2)
    left.distance, right.distance = 7, None
    assert stabilizer_distance_preserved(left, right)
    css_left, css_right = CSSCode(n=2), CSSCode(n=2)
    css_left.x_distance, css_left.z_distance = 7, 7
    css_right.x_distance, css_right.z_distance = None, None
    assert css_distance_preserved(css_left, css_right)


def test_campaign_resumes_without_work_and_refuses_mismatch(tmp_path: Path) -> None:
    configuration = _configuration()
    collect_a8.collect_campaign(tmp_path, configuration, retries=set(), verbose=False)
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    collect_a8.collect_campaign(tmp_path, configuration, retries=set(), verbose=False)
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8") == events
    incompatible = copy.deepcopy(configuration)
    incompatible["certified_negative"]["depth"] = 3
    with pytest.raises(ValueError, match="incompatible resume"):
        collect_a8.collect_campaign(tmp_path, incompatible, retries=set(), verbose=False)


def test_campaign_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with collect_a8.CampaignLock(tmp_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with collect_a8.CampaignLock(tmp_path):
                pass


def test_atomic_stage_write_preserves_previous_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "stage.json"
    collect_a8._atomic_json(path, {"status": "old"})

    def fail_replace(*args: object) -> None:
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr(collect_a8.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        collect_a8._atomic_json(path, {"status": "new"})
    assert json.loads(path.read_text()) == {"status": "old"}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["stage.json"]


def test_incorrect_certified_answer_is_persisted_and_stops_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = collect_a8.HYBRIDS["pm_stb_hybrid"]

    def wrong(*args: object) -> tuple[bool, str]:
        return False, "CI"

    monkeypatch.setitem(
        collect_a8.HYBRIDS,
        "pm_stb_hybrid",
        collect_a8.Hybrid(original.name, original.problem, wrong, original.css_only),
    )
    with pytest.raises(RuntimeError, match="incorrect hybrid answer"):
        collect_a8.collect_campaign(
            tmp_path, _configuration(), retries=set(),
            max_systematic_errors=0, verbose=False,
        )
    path = tmp_path / "stages" / "execution" / "pm_stb_hybrid__bell__positive_control__89.json"
    saved = json.loads(path.read_text())
    assert saved["status"] == "incorrect"
    assert saved["expected"] is True and saved["decision"] is False


def test_generation_failure_is_visible_and_explicit_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration()
    original = collect_a8.generate_input_payload

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("deterministic generation defect")

    monkeypatch.setattr(collect_a8, "generate_input_payload", fail)
    collect_a8.collect_campaign(tmp_path, configuration, retries=set(),
                                max_systematic_errors=0, verbose=False)
    with (tmp_path / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(int(row["num_generation_failures"]) == 1 for row in rows)
    assert all(int(row["num_successful"]) == 0 for row in rows)

    monkeypatch.setattr(collect_a8, "generate_input_payload", original)
    collect_a8.collect_campaign(tmp_path, configuration, retries={"generation"},
                                max_systematic_errors=0, verbose=False)
    with (tmp_path / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(int(row["num_generation_failures"]) == 0 for row in rows)
    assert all(int(row["num_successful"]) == 1 for row in rows)
    generation_records = list((tmp_path / "stages" / "generation").glob("*.json"))
    assert all(len(json.loads(path.read_text())["history"]) == 1
               for path in generation_records)


@pytest.mark.filterwarnings("ignore:This figure includes Axes")
def test_collection_extract_visualize_small_isolated_dataset(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    results = tmp_path / "by_cell.csv"
    image = tmp_path / "a8.png"
    args = collect_a8.parse_args([
        "--preflight", "--codes", "bell", "--algorithms", "pm-stb", "pm-css", "lc-stb",
        "--seeds", "89", "--generation-timeout", "2",
        "--certification-timeout", "2", "--execution-timeout", "2",
        "--generation-memory-gib", "2", "--certification-memory-gib", "2",
        "--execution-memory-gib", "2",
    ])
    collect_a8.collect_campaign(campaign, collect_a8._build_configuration(args),
                                retries=set(), verbose=False)
    certification_names = {
        path.name for path in (campaign / "stages" / "certification").glob("*.json")
    }
    assert certification_names
    assert all("positive" not in name for name in certification_names)
    extracted = extract(campaign, results)
    assert len(extracted) == 6
    assert {row["population"] for row in extracted} == set(collect_a8.POPULATIONS)
    negative_pm_stb = next(
        row for row in extracted
        if row["algorithm"] == "pm_stb_hybrid"
        and row["population"] == "certified_negative"
    )
    assert negative_pm_stb["num_certified_equivalent"] > 0
    assert negative_pm_stb["certification_coverage_fraction"] == 1.0
    assert render(results, image) == image
    assert image.stat().st_size > 0
