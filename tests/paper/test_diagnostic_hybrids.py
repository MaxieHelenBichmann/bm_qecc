"""Regression tests for the paper's diagnostic hybrid copies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from paper.hybrids import lc_stb, pm_css, pm_stb
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


def test_pm_css_matroid_backend_is_reflexive() -> None:
    code = CSSCode(Hx=np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int8))

    assert pm_css.are_peq_css(code, code) == (True, "MI")


def test_pm_css_matroid_colors_survive_an_empty_circuit_class() -> None:
    code = CSSCode(Hx=np.eye(6, dtype=np.int8))

    assert pm_css.are_peq_css(code, code) == (True, "MI")


def test_pm_css_matroid_backend_reports_circuit_mismatch() -> None:
    code1 = CSSCode(
        Hx=np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int8),
        Hz=np.array(
            [[0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=np.int8
        ),
    )
    code2 = CSSCode(
        Hx=np.array([[0, 1, 1, 0, 0, 1]], dtype=np.int8),
        Hz=np.array(
            [[0, 0, 0, 1, 0, 0], [0, 1, 1, 1, 0, 0]], dtype=np.int8
        ),
    )

    assert pm_css.are_peq_css(code1, code2) == (False, "MI")


def test_pm_css_sat_backend_uses_satisfiability_as_equivalence() -> None:
    hx = np.array([[1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    hz = np.zeros((0, 6), dtype=np.uint8)
    partition = {0: list(range(6))}

    assert pm_css._sat(hx, hz, partition, hx, hz, partition) == (True, "SAT")

    different_weight = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.uint8)
    assert pm_css._sat(
        hx, hz, partition, different_weight, hz, partition
    ) == (False, "SAT")


def test_pm_css_trivial_codes_are_equivalent() -> None:
    code = CSSCode(n=6)

    assert (code.distance, code.x_distance, code.z_distance) == (1, 1, 1)
    assert code.x_logicals_as_pauli_strings() == [
        "XIIIII",
        "IXIIII",
        "IIXIII",
        "IIIXII",
        "IIIIXI",
        "IIIIIX",
    ]
    assert code.z_logicals_as_pauli_strings() == [
        "ZIIIII",
        "IZIIII",
        "IIZIII",
        "IIIZII",
        "IIIIZI",
        "IIIIIZ",
    ]
    assert [str(pauli) for pauli in code.x_logicals] == code.x_logicals_as_pauli_strings()
    assert [str(pauli) for pauli in code.z_logicals] == code.z_logicals_as_pauli_strings()
    assert pm_css.are_peq_css(code, code) == (True, "CI")


def test_pm_stb_sat_backend_is_reflexive() -> None:
    code = StabilizerCode(["XXXXXX"])

    assert pm_stb.are_peq_stab(code, code) == (True, "SAT")


def test_pm_stb_sat_backend_rejects_different_support_weight() -> None:
    c1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    c2 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    partition = {(0,): list(range(6))}

    assert pm_stb._sat(c1, partition, c2, partition) == (False, "SAT")


def test_pm_stb_graph_backend_returns_diagnostic_decision() -> None:
    tableau = np.array([[1, 0, 0, 1]], dtype=np.uint8)
    partition = {(0,): [0, 1]}

    assert pm_stb._graph_iso(
        tableau, partition, tableau, partition
    ) == (True, "GI")


def test_lc_stb_sat_uses_stabilizer_rank_for_tableau_rows() -> None:
    code = StabilizerCode(["XIIII", "IXIII", "IIXII"])
    reduced = lc_stb._row_basis(code.symplectic)

    assert code.n == 5
    assert code.k == 2
    assert reduced.shape[0] == 3
    assert lc_stb._sat(reduced, reduced) == (True, "SAT")
    assert lc_stb.are_lceq(code, code) == (True, "SAT")


def test_lc_stb_sat_rejects_different_support_weight() -> None:
    c1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    c2 = np.array([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    assert lc_stb._sat(c1, c2) == (False, "SAT")


def test_lc_stb_graph_backend_returns_diagnostic_decision() -> None:
    c1 = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    c2 = np.array([[1, 1, 0, 0]], dtype=np.uint8)

    assert lc_stb._graph_iso(c1, c1) == (True, "GI")
    assert lc_stb._graph_iso(c1, c2) == (False, "GI")


def test_lc_stb_empty_codes_are_equivalent() -> None:
    code = SimpleNamespace(n=0, k=0)

    assert lc_stb.are_lceq(code, code) == (True, "")  # type: ignore[arg-type]
