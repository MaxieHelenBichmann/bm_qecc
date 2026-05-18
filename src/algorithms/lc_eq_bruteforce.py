"""Graph-state machinery for local-Clifford equivalence checking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
from itertools import product

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

from ..core.stabilizer_code import StabilizerCode

LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")

def are_lceq_bruteforce(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check Local-Clifford equivalence by brute-force search over all possible actions Local Cliffords can have on the qubits.

    Each action is checked by applying it to the qubits and checking whether the stabilizer tableaus describe the same row space.

    The action of the Local Clifford group on a qubit in symplectic form is defined as:
    I: (x, z) -> (x, z)
    H: (x, z) -> (z, x)
    S: (x, z) -> (x, x + z)
    HS: (x, z) -> (x + z, x)
    SH: (x, z) -> (z, x + z)
    HSH: (x, z) -> (x + z, z)

    Each row space check should be done in O(n^3) time, and there are O(n^6) Local Clifford actions on the tableau, so the overall runtime is O(n^6 * n^3) which is obviously not efficient at all.
    """
    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)
    
    n = c1.n
    rank_c1 = _rank(c1.symplectic)

    def apply_lc(tableau: npt.NDArray[np.int8], lc: str, qubit: int) -> npt.NDArray[np.int8]:
        if lc == "I":
            pass
        elif lc  == "H":
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "S":
            tableau[:, qubit + n] ^= tableau[:, qubit]
        elif lc == "HS":
            tableau[:, qubit + n] ^=  tableau[:, qubit]
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "SH":
            tableau[:, qubit] ^= tableau[:, qubit + n]
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "HSH":
            tableau[:, qubit] ^= tableau[:, qubit + n]
        return tableau

        
    for action in product(LOCAL_CLIFFORDS, repeat=n):
        lc_tableau = c2.symplectic.copy()

        for qubit, lc in enumerate(action):
            lc_tableau = apply_lc(lc_tableau, lc, qubit)

        if (rank_c1 == _rank(lc_tableau) == _rank(np.vstack([c1.symplectic, lc_tableau]))):
            return True
    
    return False
