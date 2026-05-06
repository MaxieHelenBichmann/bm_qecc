# Benchmarking Equivalence Checking QECC

This Repository is used for Benchmarking different approaches for Equivalence Checking QECC. It is used as part of the implementation for my Bachelor's Thesis "Automated Equivalence Checking of Stabilizer Codes", which contributes to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc). Thus, the required classes are taken from this Repository.


## Problem

The following problems are to be benchmarked.

### Permutation Equivalence

- A) Are two given Stabilizer Codes C and C' equivalent (define the same codespace) up to permutation of the output qubits?

$$C \thicksim_P C'  \iff \exists P \in S_n : \exists R \in GL(2) : tableu(C') = R \ tableu(C) \begin{pmatrix} P & 0 \\ 0 & P\end{pmatrix} $$

### Local-Clifford Equivalence

- B) Are two given Stabilizer Codes C and C' Local-Clifford equivalent (define the same codespace) up to Local Clifford gates on the output qubits?

$$C \thicksim_{LC} C'  \iff \exists U = U_1 \bigotimes ... \bigotimes U_n \ with \ U_i \in \{I, H ,S\}^l : rowspace(tableu(C')) = U \ rowspace(tableu(C)) \ U^\dagger = \{U \ g \ U^\dagger \ | \ g \in rowspace(tableu(C)) \} $$

- C) Is a given Stabilizer Code C Local-Clifford equivalent (define the same codespace up to Local Clifford gates on the output qubits) to a CSS Code?

$$\exists C_{CSS} \ with \ tableu(C_{CSS}) = \begin{pmatrix} H_x & 0 \\ 0 & H_z\end{pmatrix} : C \thicksim_{LC} C_{CSS} $$

## Approaches

### Permutation Equivalence
#### CSS Codes
- Brute Force
- Matroid Isomorphism
- Classical Algorithms
- Graph Isomorphism

#### Stabilizer Codes
- Brute Force
- Graph State Machinery
- Classical Algorithms
- Automorphisms
- Graph Isomorphism
- SAT

### Local-Clifford Equivalence
#### $C \thicksim_{LC} C'$
- Graph State Machinery

#### $C \thicksim_{LC} C_{CSS}$
- Brute Force
- Graph State Machinery
- KLS Normal Form
- LC Orbit


## Scope

I use the term "Benchmark" to measure the runtime of the python algorithms with an expected workload - input codes $\llbracket n,k,d \rrbracket$ with $n$ ranging from $2$ to $\thicksim 50$. NOT meant are more detailed benchmark or profiling analyses.
