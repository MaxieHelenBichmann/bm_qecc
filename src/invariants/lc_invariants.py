"""Small invariants that are preserved under local Clifford equivalence, which can be used to quickly rule out LC-equivalence in some cases.
"""

from __future__ import annotations

import numpy as np
import ldpc.mod2.mod2_numpy as mod2
from itertools import combinations

from ..core.stabilizer_code import StabilizerCode

def preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.n == c2.n

def preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.k == c2.k

def preserved_local_weight_distribution(c1: StabilizerCode, c2: StabilizerCode, max_subset_size: int | None = None) -> bool:
    """Check whether the r = 2 local weight distribution is preserved, which is a necessary condition for LC-equivalence.

    for each w^1, w^2, w^12 ⊆ {1, ..., n}: 
        dim(V_n,2) = dim{ (v_1, v_2) in S x S | supp(v_1) ⊆ w^1, supp(v_2) ⊆ w^2, supp(v_1 + v_2) ⊆ w^12 } 
            =!= 
        dim(V'_n,2) = dim{ (v_1, v_2) in S' x S' | supp(v_1) ⊆ w^1, supp(v_2) ⊆ w^2, supp(v_1 + v_2) ⊆ w^12 }
    
    Reference for this invariant: 
    - Maarten Van den Nest, Jeroen Dehaene, Bart De Moor: Finite set of invariants to characterize local Clifford equivalence of stabilizer states
    """
    n = c1.n
    rk = c1.n - c1.k # assumption: c1 and c2 have the same n and k AND no redundant stabilizer generators

    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)
    
    def _loc_weight_dim(code: StabilizerCode, w1: tuple[int, ...], w2: tuple[int, ...], w12: tuple[int, ...]) -> int:
        """
        v in S |  supp(v) ⊆ w
        -> y in F_2 |  supp(yG) ⊆ w (means "outside of w (aka all qubits not in w), there can only be identity", aka all columns outside of w must be zero)
        -> y (G|_(w^c)) = 0 for the restricted matrix outside
        -> {v in S | supp(v) ⊆ w} = kernel of G|_(w^c)
        -> multiple conditions: supp(v_1) ⊆ w^1  &  supp(v_2) ⊆ w^2  &  supp(v_1 + v_2) ⊆ w^12
        -> y_1 (G|_(w1^c)) = 0   &   y_2 (G|_(w2^c)) = 0   &   (y_1 + y_2) (G|_(w12^c)) = 0
        -> combination in one constraint system: (y1, y2) @  M = 0

        M =
        [ G|_(w1^c)    0      G|_(w12^c) ]
        [    0     G|_(w2^c)  G|_(w12^c) ]
        """
       
        def _restricted_to_outside(G: np.ndarray, w: tuple[int, ...]) -> np.ndarray:
            G = np.asarray(G, dtype=np.uint8) & 1

            outside = [i for i in range(n) if i not in w]
            cols = outside + [i + n for i in outside]

            return G[:, cols]
        
        G1 = _restricted_to_outside(code.symplectic, w1)
        G2 = _restricted_to_outside(code.symplectic, w2)
        G12 = _restricted_to_outside(code.symplectic, w12)

        top = np.hstack([G1, np.zeros((rk, G2.shape[1]), dtype=np.uint8), G12])
        bottom = np.hstack([np.zeros((rk, G1.shape[1]), dtype=np.uint8), G2, G12])
        M = np.vstack([top, bottom])

        return 2*rk - _rank(M)
    
    def subsets_up_to_size(max_size):
        for a in range(max_size + 1):
            yield from combinations(range(n), a)
    
    if not max_subset_size:
        max_subset_size = c1.n # theoretically O(2^(3n)), but smaller subsets are also valid, only weaker
    else:
        max_subset_size = min(max_subset_size, c1.n)
        
    for w1 in subsets_up_to_size(max_subset_size):
        for w2 in subsets_up_to_size(max_subset_size):
            for w12 in subsets_up_to_size(max_subset_size):
                if _loc_weight_dim(c1, w1, w2, w12) != _loc_weight_dim(c2, w1, w2, w12):
                    return False
    
    return True

def preserved_low_degree_local_invariant(c1: StabilizerCode, c2: StabilizerCode, max_subset_size: int | None = None) -> bool:
    """Check whether the r = 2 local invariant is preserved, which is a necessary condition for LC-equivalence.

    for each A ⊆ {1, ..., n}: 
        d(A) = dim({s in S | supp(s) ⊆ A})
            =!= 
        d'(A) = dim({s in S' | supp(s) ⊆ A})
    
    Reference for this invariant: 
    - Maarten Van den Nest, Bart De Moor: Local Invariants of Stabilizer Codes
    """
    n = c1.n
    rk = c1.n - c1.k # assumption: c1 and c2 have the same n and k AND no redundant stabilizer generators

    def _rank(matrix: np.ndarray) -> int:
        if matrix.shape[0] == 0:
            return 0
        return mod2.rank(matrix)
    
    def _supp_subcode_dim(code: StabilizerCode, subset: tuple[int, ...]) -> int:
        """
        d(A) = dim({s in S | supp(s) ⊆ A}) = dim({y in F_2 | supp(yG) ⊆ A}) 
             = rank(G) - rank(G|_(A^c)})

        -> supp(yG) ⊆ A means "outside of A (aka all qubit not in A), there can only be identity", aka all columns outside of A must be zero 
        -> y (G|_(A^c)) = 0 for the restricted matrix outside if A 
        -> {y in F_2 | supp(yG) ⊆ A} = kernel of G|_(A^c) -> dim ker = n - rank
        """
        G = np.asarray(code.symplectic, dtype=np.uint8) & 1

        A = set(subset)
        outside = [i for i in range(n) if i not in A]

        cols = outside + [i + n for i in outside]

        if not cols:
            return rk
        
        restricted = G[:, cols]
        return rk - _rank(restricted)
    
    if not max_subset_size:
        max_subset_size = n # theoretically O(2^n), but smaller subsets are also valid, only weaker
    else:
        max_subset_size = min(max_subset_size, n)

    for a in range(max_subset_size + 1):
        for subset in combinations(range(c1.n), a):
            if _supp_subcode_dim(c1, subset) != _supp_subcode_dim(c2, subset):
                return False

    return True
