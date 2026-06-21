"""Focused checks for the KLS/HK normal-form conversion."""

from __future__ import annotations

import numpy as np
import pytest

import ldpc.mod2.mod2_numpy as mod2

from benchmarks.utils import random_stabilizer_code, lc_equivalent_code
from src.algorithms.lc_eq.lc_eq_kls import (
    GSLC,
    _code_to_encoder_circuit,
    _code_to_graph,
    _hk_normal_form,
    _kls_normal_form,
    are_lceq_kls,
)
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode

# ----------------------------------------------------------------------------------------------------
# _code_to_graph
# ----------------------------------------------------------------------------------------------------


def _assert_semi_bipartite_encoder_graph(graph: GSLC) -> None:
    assert len(graph.vertices) == graph.k + graph.n
    assert graph.get_input_output_adjacency().shape == (graph.k, graph.n)

    for u, v in graph.edges:
        assert 0 <= u < v < graph.k + graph.n
        assert not (u < graph.k and v < graph.k)


def _assert_graph_represents_encoder(code: StabilizerCode, graph: GSLC) -> None:
    encoder_tableau = _encoder_state_tableau(code)
    graph_tableau = _graph_state_tableau(graph)
    _assert_same_row_space(encoder_tableau, graph_tableau)


def _encoder_state_tableau(code: StabilizerCode) -> np.ndarray:
    circuit = _code_to_encoder_circuit(code)
    num_qubits = code.n + code.k
    tableau = StabilizerTableau(
        np.hstack(
            [
                np.zeros((num_qubits, num_qubits), dtype=np.int8),
                np.eye(num_qubits, dtype=np.int8),
            ]
        )
    )

    for gate in circuit.gates:
        if gate.name == "HAD":
            tableau.apply_h(gate.target)
        elif gate.name == "ZPhase":
            if gate.phase == 1 / 2:
                tableau.apply_s(gate.target)
            elif gate.phase == 1:
                tableau.apply_z(gate.target)
            elif gate.phase == 3 / 2:
                tableau.apply_sdg(gate.target)
            else:
                raise ValueError(f"Unexpected Z phase {gate.phase}.")
        elif gate.name == "CNOT":
            tableau.apply_cx(gate.control, gate.target)
        else:
            raise ValueError(f"Unexpected gate {gate.name}.")

    return tableau.tableau.matrix


def _graph_state_tableau(graph: GSLC) -> np.ndarray:
    num_qubits = graph.n + graph.k
    tableau = StabilizerTableau(
        np.hstack(
            [
                np.eye(num_qubits, dtype=np.int8),
                np.zeros((num_qubits, num_qubits), dtype=np.int8),
            ]
        )
    )

    for u, v in graph.edges:
        tableau.apply_cz(u, v)

    for qubit, (decoration, has_z) in enumerate(graph.vertices):
        for gate in decoration:
            if gate == "H":
                tableau.apply_h(qubit)
            elif gate == "S":
                tableau.apply_s(qubit)
            elif gate == "Z":
                tableau.apply_z(qubit)
            else:
                raise ValueError(f"Unexpected local Clifford gate {gate}.")
        if has_z:
            tableau.apply_z(qubit)

    return tableau.tableau.matrix


def _assert_same_row_space(left: np.ndarray, right: np.ndarray) -> None:
    left = np.asarray(left, dtype=np.uint8) & 1
    right = np.asarray(right, dtype=np.uint8) & 1
    left_rank = mod2.rank(left)

    assert left.shape == right.shape
    assert left_rank == left.shape[0]
    assert left_rank == mod2.rank(right)
    assert left_rank == mod2.rank(np.vstack([left, right]))


@pytest.mark.parametrize(
    ("code", "expected_n", "expected_k", "expected_vertices", "expected_edges"),
    [
        pytest.param(
            StabilizerCode(["Z"]),
            1,
            0,
            [(["H"], False)],
            set(),
            id="one-qubit-z-state",
        ),
        pytest.param(
            StabilizerCode(["X"]),
            1,
            0,
            [([], False)],
            set(),
            id="one-qubit-x-state",
        ),
        pytest.param(
            StabilizerCode(["Y"]),
            1,
            0,
            [(["S"], False)],
            set(),
            id="one-qubit-y-state",
        ),
    ],
)
def test_code_to_graph_small_codes(
    code: StabilizerCode,
    expected_n: int,
    expected_k: int,
    expected_vertices: list[tuple[list[str], bool]],
    expected_edges: set[tuple[int, int]],
) -> None:
    graph = _code_to_graph(code)

    assert graph.n == expected_n
    assert graph.k == expected_k
    _assert_graph(graph, expected_vertices, expected_edges)
    _assert_semi_bipartite_encoder_graph(graph)
    _assert_graph_represents_encoder(code, graph)


def test_code_to_graph_rejects_identity_stabilizer_row() -> None:
    with pytest.raises(ValueError, match="identity stabilizer row"):
        _code_to_graph(StabilizerCode(["I"]))

@pytest.mark.parametrize(
    "code",
    [
        pytest.param(StabilizerCode.get_trivial_code(1), id="one-qubit-trivial-code"),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            id="two-qubit-repetition-code",
        ),
        pytest.param(
            StabilizerCode(["ZZI", "IZZ"], z_logicals=["ZII"], x_logicals=["XXX"]),
            id="three-qubit-repetition-code",
        ),
        pytest.param(
            StabilizerCode(["XX", "ZZ"]),
            id="bell-state",
        ),
        pytest.param(
            StabilizerCode(["IZI"]),
            id="IZI-code",
        ),
        pytest.param(
            StabilizerCode(["IXI"]),
            id="IXI-code",
        ),
        pytest.param(
            StabilizerCode(["XZZ"]),
            id="XZZ-code",
        ),
        pytest.param(
            StabilizerCode(["IXI", "YII", "YXX"]),
            id="failing-example-1",
        ),
        pytest.param(
            StabilizerCode(["ZIZI"]),
            id="failing-example-2",
        ),
        pytest.param(
            StabilizerCode(["ZXZZI", "IXIIZ", "IIIZI"]),
            id="failing-example-3",
        ),
                pytest.param(
            StabilizerCode(["XZXYYXIZX", "IIXIIIIII", "XXXZXIZIZ", "XZXZYYXXY", "IYXIZXIZI"]),
            id="failing-example-4",
        ),
    ],
)
def test_code_to_graph_encoder_respecting_form_smoke(code: StabilizerCode) -> None:
    assert isinstance(_code_to_graph(code), GSLC)
    graph = _code_to_graph(code)
    _assert_graph_represents_encoder(code, graph)

# ----------------------------------------------------------------------------------------------------
# _hk_normal_form
# ----------------------------------------------------------------------------------------------------


def _hk(vertices: list[tuple[list[str], bool]], edges: set[tuple[int, int]]) -> GSLC:
    graph = GSLC()
    graph.n = len(vertices)
    graph.k = 0
    graph.vertices = [(list(deco), z) for deco, z in vertices]
    graph.edges = set(edges)
    _hk_normal_form(graph)
    return graph


def _assert_graph(
    graph: GSLC,
    vertices: list[tuple[list[str], bool]],
    edges: set[tuple[int, int]],
) -> None:
    assert graph.vertices == vertices
    assert graph.edges == edges


def _assert_adjacency_matches_edges(graph: GSLC) -> None:
    expected = [set() for _ in range(graph.n + graph.k)]
    for u, v in graph.edges:
        expected[u].add(v)
        expected[v].add(u)

    graph.neighbors(0)
    assert graph.adj == expected


def test_gslc_rebuilds_adjacency_for_manually_assigned_edges() -> None:
    graph = _graph(
        2,
        2,
        [([], False), ([], False), ([], False), ([], False)],
        {(0, 2), (1, 3), (2, 3)},
    )

    assert set(graph.neighbors(2)) == {0, 3}
    _assert_adjacency_matches_edges(graph)


def test_gslc_apply_cz_edge_keeps_adjacency_cache_in_sync() -> None:
    graph = _graph(
        3,
        0,
        [([], False), ([], False), ([], False)],
        {(0, 1)},
    )

    assert set(graph.neighbors(0)) == {1}

    graph.apply_cz_edge(0, 1)
    assert set(graph.neighbors(0)) == set()
    assert set(graph.neighbors(1)) == set()
    _assert_adjacency_matches_edges(graph)

    graph.apply_cz_edge(2, 0)
    assert set(graph.neighbors(0)) == {2}
    assert set(graph.neighbors(2)) == {0}
    _assert_adjacency_matches_edges(graph)


def test_gslc_copy_has_independent_adjacency_cache() -> None:
    graph = _graph(
        3,
        0,
        [([], False), ([], False), ([], False)],
        {(0, 1)},
    )
    copied = graph.copy()

    copied.apply_cz_edge(1, 2)

    assert graph.edges == {(0, 1)}
    assert copied.edges == {(0, 1), (1, 2)}
    _assert_adjacency_matches_edges(graph)
    _assert_adjacency_matches_edges(copied)


def test_gslc_set_edges_replaces_adjacency_cache() -> None:
    graph = _graph(
        3,
        0,
        [([], False), ([], False), ([], False)],
        {(0, 1)},
    )

    assert set(graph.neighbors(0)) == {1}

    graph.set_edges({(1, 2)})

    assert graph.edges == {(1, 2)}
    assert set(graph.neighbors(0)) == set()
    assert set(graph.neighbors(1)) == {2}
    _assert_adjacency_matches_edges(graph)


def _assert_hk_requirements(graph: GSLC) -> None:
    for i, (deco, _) in enumerate(graph.vertices):
        assert deco in ([], ["S"], ["H"])
        if deco == ["H"]:
            assert all(i < neighbor for neighbor in graph.neighbors(i))


@pytest.mark.parametrize(
    ("vertices", "expected_vertices"),
    [
        pytest.param([(["H", "H"], False)], [([], False)], id="hh-cancels"),
        pytest.param([(["S", "S"], False)], [([], True)], id="ss-becomes-z-phase"),
        pytest.param([(["Z", "Z"], True)], [([], True)], id="zz-cancels-before-z-bit"),
        pytest.param([(["Z", "S"], False)], [(["S"], True)], id="push-z-through-s"),
        pytest.param([(["H", "Z", "S"], True)], [(["S"], True)], id="internal-z-expands-to-s-square"),
    ],
)
def test_hk_normal_form_reduces_local_clifford_words(
    vertices: list[tuple[list[str], bool]],
    expected_vertices: list[tuple[list[str], bool]],
) -> None:
    graph = _hk(vertices, set())

    _assert_graph(graph, expected_vertices, set())
    _assert_hk_requirements(graph)


@pytest.mark.parametrize(
    ("vertices", "edges", "expected_vertices", "expected_edges"),
    [
        pytest.param(
            [(["H"], False), (["S"], False), ([], False)],
            {(0, 1), (1, 2)},
            [(["H"], False), (["S"], False), ([], False)],
            {(0, 1), (1, 2)},
            id="already-hk-normal-form",
        ),
        pytest.param(
            [(["H", "S"], False), ([], False), ([], False)],
            {(0, 1), (0, 2)},
            [(["S"], True), (["S"], True), (["S"], True)],
            {(0, 1), (0, 2), (1, 2)},
            id="reduce-trailing-hs",
        ),
        pytest.param(
            [(["S", "H"], False), ([], False), ([], False)],
            {(0, 1), (0, 2)},
            [(["H"], False), (["S"], False), (["S"], False)],
            {(0, 1), (0, 2), (1, 2)},
            id="reduce-solo-trailing-sh",
        ),
        pytest.param(
            [(["S", "H"], False), (["H"], False), ([], False)],
            {(0, 1), (0, 2)},
            [(["S"], False), ([], False), ([], False)],
            {(0, 1), (1, 2)},
            id="reduce-shared-trailing-sh",
        ),
        pytest.param(
            [(["S"], False), (["S"], False), (["H"], False), (["H"], False)],
            {(0, 2), (1, 3)},
            [(["H"], False), (["H"], False), ([], True), ([], True)],
            {(0, 2), (1, 3)},
            id="h-slide-and-sh-cleanup",
        ),
        pytest.param(
            [(["H"], False), (["H"], False)],
            {(0, 1)},
            [([], False), ([], False)],
            {(0, 1)},
            id="h-slide-onto-existing-h",
        ),
        pytest.param(
            [(["S"], False), (["H"], False)],
            {(0, 1)},
            [(["H"], False), ([], True)],
            {(0, 1)},
            id="h-slide-creates-and-cleans-sh-with-empty-neighbor",
        ),
        pytest.param(
            [(["S"], False), (["S"], False), (["H"], False)],
            {(0, 2), (1, 2)},
            [(["H"], False), ([], True), ([], True)],
            {(0, 1), (0, 2), (1, 2)},
            id="h-slide-creates-and-cleans-sh-with-s-neighbor",
        ),
        pytest.param(
            [([], False), ([], False), (["H"], False), ([], False)],
            {(0, 2), (1, 2), (2, 3)},
            [(["H"], False), ([], False), ([], False), ([], False)],
            {(0, 1), (0, 2), (0, 3)},
            id="h-slide-through-star",
        ),
    ],
)
def test_hk_normal_form_cases(
    vertices: list[tuple[list[str], bool]],
    edges: set[tuple[int, int]],
    expected_vertices: list[tuple[list[str], bool]],
    expected_edges: set[tuple[int, int]],
) -> None:
    graph = _hk(vertices, edges)
    _assert_graph(graph, expected_vertices, expected_edges)
    _assert_hk_requirements(graph)


def test_hk_normal_form_rejects_unsupported_residual_word() -> None:
    with pytest.raises(ValueError, match="Expected only I, S, or H decorations"):
        _hk([(["H", "S", "H"], False), ([], False)], {(0, 1)})

# ----------------------------------------------------------------------------------------------------
# _kls_normal_form
# ----------------------------------------------------------------------------------------------------


def _graph(
    n: int,
    k: int,
    vertices: list[tuple[list[str], bool]],
    edges: set[tuple[int, int]],
) -> GSLC:
    graph = GSLC()
    graph.n = n
    graph.k = k
    graph.vertices = [(list(deco), z) for deco, z in vertices]
    graph.edges = set(edges)
    return graph


def _assert_rref(matrix: np.ndarray) -> None:
    pivots : list[int] = []
    zero_row_seen = False

    for row_idx, row in enumerate(matrix.astype(np.uint8)):
        ones = np.flatnonzero(row)
        if len(ones) == 0:
            zero_row_seen = True
            continue

        assert not zero_row_seen

        pivot = int(ones[0])
        assert not pivots or pivot > pivots[-1]

        expected_col = np.zeros(matrix.shape[0], dtype=np.uint8)
        expected_col[row_idx] = 1
        np.testing.assert_array_equal(matrix[:, pivot].astype(np.uint8), expected_col)

        pivots.append(pivot)


def _assert_kls_requirements(graph: GSLC) -> None:
    _assert_semi_bipartite_encoder_graph(graph)

    for input_vertex in range(graph.k):
        assert graph.vertices[input_vertex] == ([], False)

    input_output_adjacency = graph.get_input_output_adjacency()
    _assert_rref(input_output_adjacency)


def _pivot_output_vertices(graph: GSLC) -> list[int]:
    input_output_adjacency = graph.get_input_output_adjacency()
    pivot_vertices: list[int] = []

    for row in input_output_adjacency.astype(np.uint8):
        nonzero_cols = np.flatnonzero(row)
        if len(nonzero_cols) == 0:
            continue
        pivot_vertices.append(graph.k + int(nonzero_cols[0]))

    return pivot_vertices


def _assert_kls_pivot_requirements(graph: GSLC) -> None:
    pivot_vertices = _pivot_output_vertices(graph)

    for pivot_vertex in pivot_vertices:
        assert graph.vertices[pivot_vertex] == ([], False)

    for i, pivot_a in enumerate(pivot_vertices):
        for pivot_b in pivot_vertices[i + 1:]:
            assert (pivot_a, pivot_b) not in graph.edges


@pytest.mark.parametrize(
    ("graph_hk", "expected_io_adjacency"),
    [
        pytest.param(
            _graph(
                3,
                2,
                [(["H"], True), (["S"], True), (["S"], False), (["H"], False), ([], True)],
                {(0, 1), (0, 2), (1, 4)},
            ),
            np.array([[1, 0, 0], [0, 0, 1]], dtype=bool),
            id="strips-input-decorations-and-input-edge",
        ),
        pytest.param(
            _graph(
                3,
                2,
                [(["H"], False), (["S"], True), ([], False), (["S"], False), (["H"], True)],
                {(0, 1), (2, 3)},
            ),
            np.array([[0, 0, 0], [0, 0, 0]], dtype=bool),
            id="zero-input-output-adjacency",
        ),
        pytest.param(
            _graph(
                4,
                3,
                [([], False), (["S"], False), (["H"], True), ([], False), ([], False), (["S"], False), ([], True)],
                {(0, 3), (1, 5)},
            ),
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=bool),
            id="rank-deficient-already-rref",
        ),
        pytest.param(
            _graph(
                3,
                3,
                [([], False), (["S"], True), (["H"], False), ([], False), ([], False), ([], False)],
                {(0, 3), (1, 4), (2, 5), (3, 4)},
            ),
            None,
            id="pivot-edge-cleanup-preserves-kls-structure",
        ),
        pytest.param(
            _graph(
                3,
                0,
                [(["H"], False), (["S"], False), ([], True)],
                {(0, 1)},
            ),
            np.zeros((0, 3), dtype=bool),
            id="no-inputs",
        ),
    ],
)
def test_kls_normal_form_structural_cases(
    graph_hk: GSLC,
    expected_io_adjacency: np.ndarray | None,
) -> None:
    _assert_hk_requirements(graph_hk)

    _kls_normal_form(graph_hk)

    _assert_kls_requirements(graph_hk)
    if expected_io_adjacency is not None:
        np.testing.assert_array_equal(graph_hk.get_input_output_adjacency(), expected_io_adjacency)


def test_kls_normal_form_clears_pivot_output_decorations() -> None:
    graph_hk = _graph(
        3,
        2,
        [(["H"], True), (["S"], True), (["S"], False), (["H"], False), ([], True)],
        {(0, 1), (0, 2), (1, 4)},
    )

    _kls_normal_form(graph_hk)

    _assert_kls_pivot_requirements(graph_hk)

# ----------------------------------------------------------------------------------------------------
# is_lceq_css_kls
# ----------------------------------------------------------------------------------------------------

def test_are_lceq_kls_choi_failing() -> None:
    code1 = StabilizerCode(["ZIYX", "ZIII"])
    code2 = StabilizerCode(["IIYZ", "XIYZ"])

    assert are_lceq_kls(code1, code2) is True

@pytest.mark.parametrize(
    ("code1", "code2", "expected"),
    [
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["X"]), True, id="one-qubit-z-vs-x"),
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["Y"]), True, id="one-qubit-z-vs-y"),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XI", "IX"]),
            True,
            id="two-product-bases",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XX", "ZZ"]),
            False,
            id="product-vs-bell-state",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["XX"], z_logicals=["XI"], x_logicals=["ZZ"]),
            True,
            id="repetition-code-under-hadamards",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            False,
            id="weight-two-vs-weight-one-stabilizer",
        ),
    ],
)
def test_are_lceq_kls_small_codes(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: bool,
) -> None:
    assert are_lceq_kls(code1, code2) is expected

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in [3, 28, 35]])
def test_are_lceq_kls_failing_choi(seed: int) -> None:
    """
    3:  < IZI > | < IXI > 
    28: < IXI > | < IYI >
    35: < XZZ > | < ZXX >
    """
    code1 = random_stabilizer_code(3, 2, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)
    assert are_lceq_kls(code1, code2) is True

def test_are_lceq_kls_random_smoke() -> None:
    for n in range(3, 6):
        for k in range(n + 1):
            code1 = random_stabilizer_code(n, k, seed=1000 + 17 * n + k)
            code2 = lc_equivalent_code(code1, seed=2000 + 17 * n + k)

            assert isinstance(are_lceq_kls(code1, code2), bool)

@pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(10)])
def test_are_lceq_kls_random_positive(seed: int) -> None:
    n = 2 + (3 * seed + 1) % 5
    k = 1 + (2 * seed + 1) % (n - 1)
    code1 = random_stabilizer_code(n, k, seed=1000 + seed)
    code2 = lc_equivalent_code(code1, seed=2000 + seed)
    assert are_lceq_kls(code1, code2) is True
