"""KLS normal form for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
import pyzx as zx
from ..core.stabilizer_code import StabilizerCode

def _code_to_encoder_circuit(code) -> zx.Circuit:
    def _delete_first_row_and_qubit(tab: np.ndarray) -> np.ndarray:
        n = tab.shape[1] // 2
        return np.delete(np.delete(np.delete(tab, 0, axis=0), 0, axis=0) , n, axis=0)

    tableau = np.asarray(code.symplectic.copy(), dtype=np.uint8) & 1
    n = code.n
    k = code.k
    original_qubits = list(range(n))

    # Elimination gates are recorded in forward elimination order.
    elimination_gates: list[tuple[str, tuple[int, ...]]] = []
    while tableau.shape[0] > 0:
        cur_n = tableau.shape[1] // 2
        x0 = tableau[0, :cur_n]
        z0 = tableau[0, cur_n:]
        support = np.flatnonzero(x0 | z0)
        if len(support) == 0:
            raise ValueError("Encountered an identity stabilizer row.")

        # 1.) turn every non-identity Pauli in row 0 into Z
        # I = (0|0) --I---> I = (0|0)
        # X = (0|1) --H---> Z = (1|0)
        # Y = (1|1) --HS--> Z = (1|0)
        # Z = (0|1) --I---> Z = (1|0)
        for q in range(cur_n):
            x_q = bool(tableau[0, q])
            z_q = bool(tableau[0, cur_n + q])
            if x_q and z_q:
                # Y -> X under S
                tableau[:, q + n] ^= tableau[:, q]
                tableau[:, [q, q + n]] = tableau[:, [q + n, q]]
                elimination_gates.append(("S", (original_qubits[q],)))
                elimination_gates.append(("H", (original_qubits[q],)))
            elif x_q and not z_q:
                # X -> Z under H
                tableau[:, [q, q + n]] = tableau[:, [q + n, q]]
                elimination_gates.append(("H", (original_qubits[q],)))

        # 2.) make the first qubit Z CNOT(0 -> pivot)
        z_support = np.flatnonzero(tableau[0, cur_n:])

        if len(z_support) == 0:
            raise RuntimeError("Failed to convert stabilizer row to Z support.")
        pivot = int(z_support[0])

        tableau[:, pivot] ^= tableau[:, 0]
        tableau[:, cur_n + 0] ^= tableau[:, cur_n + pivot]

        # 3.) clear all other Zs in row 0 using CNOT(q -> 0)
        # control: (x_c|z_c) --CNOT--> (  x_c  |z_c^z_t)
        # target : (x_t|z_t) --CNOT--> (x_t^x_c|  z_t  )
        for q in range(1, cur_n):
            if tableau[:, cur_n + q] == 1:
                tableau[:, 0] ^= tableau[:, q]
                tableau[:, cur_n + q] ^= tableau[:, cur_n + 0]

                elimination_gates.append(
                    ("CNOT", (original_qubits[q], original_qubits[0]))
                )

        if (
            np.count_nonzero(tableau[0, :cur_n]) != 0
            or np.count_nonzero(tableau[0, cur_n:]) != 1
            or tableau[0, cur_n + pivot] != 1
        ):
            raise RuntimeError("Failed to isolate a stabilizer as a single Z.")

        # 4.) clear pivot column
        for r in range(1, tableau.shape[0]):
            if tableau[r, cur_n + pivot]:
                tableau[r] ^= tableau[0]
            if tableau[r, pivot]:
                raise RuntimeError(
                    "A remaining row has X on the pivot qubit."
                )

        # 5.) remove stabilizer and qubit
        tableau = _delete_first_row_and_qubit(tableau)
        del original_qubits[pivot]

    # encoder = inverse elimination Cliffords
    circuit = zx.Circuit(n)
    circuit.initialize_qubits([True] * (n-k) + [False] * k)
    for name, qubits in reversed(elimination_gates):
        if name == "H":
            circuit.add_gate("HAD", qubits[0])
        elif name == "S":
            # S† = Z phase 3π/2
            circuit.add_gate("ZPhase", qubits[0], phase=3 / 2)
        elif name == "CNOT":
            circuit.add_gate("CNOT", qubits[0], qubits[1])
        
    return circuit 

def _code_to_graph_state(code):
    # 1.) code -> encoder circuit
    circuit = _code_to_encoder_circuit(code)

    # 2.) Encoder circuit -> zx diagram
    zx_diagram = circuit.to_graph()

    # 3.) zx diagram -> graph state
    zx_diagram.normalize() 

    pass

def _encoder_circuit_to_graph_state(circuit):
    circuit = zx.Circuit.to_basic_gates(circuit)
    zx_diagram = circuit.to_graph()
    zx_diagram.normalize() 
    pass

def _hk_normal_form(graph_state):
    pass

def _kls_normal_form(hk_normal_form):
    pass


def is_lceq_css_kls(code: StabilizerCode) -> bool:
    return False
