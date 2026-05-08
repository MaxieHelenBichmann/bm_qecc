"""Brute-force algorithm for checking whether a stabilizer code is LC-equivalent to a CSS code."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from ldpc.mod2.mod2_numpy import rank

from ..core.stabilizer_code import StabilizerCode

if TYPE_CHECKING:
    import numpy.typing as npt


LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")

def is_lceq_css_bruteforce(code: StabilizerCode) -> bool:
    """Check permutation equivalence by brute-force search over all possible actions Local Cliffords can have on the qubits.

    Each action is checked by applying it to the qubits and checking whether the sum of the ranks of the X and Z part in the tableau [X | Z] is the rank of the full tableau, which is a sufficient and necessary condition for the resulting tableau being in CSS form.

    The action of the Local Clifford group on a qubit in symplectic form is defined as:
    I: (x, z) -> (x, z)
    H: (x, z) -> (z, x)
    S: (x, z) -> (x, x + z)
    HS: (x, z) -> (x + z, x)
    SH: (x, z) -> (z, x + z)
    HSH: (x, z) -> (x + z, z)

    Each row space check should be done in O(n^3) time, and there are O(n^6) Local Clifford actions on the tableau, so the overall runtime is O(n^6 * n^3) which is obviously not efficient at all.
    """
    n = code.n
    r = rank(code.symplectic)

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
        lc_tableau = code.symplectic.copy()

        for qubit, lc in enumerate(action):
            lc_tableau = apply_lc(lc_tableau, lc, qubit)

        if (rank(lc_tableau[:, :n]) + rank(lc_tableau[:, n:]) == r):
            return True
    
    return False
