# Benchmarking Equivalence Checking QECC

This Repository is used for Benchmarking different approaches for Equivalence Checking QECC. It is used as part of the implementation for my Bachelor's Thesis "Automated Equivalence Checking of Stabilizer Codes" \[WIP\], which contributes to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc). Thus, the required infrastructure classes (e.g. `CSSCode`, `Pauli`, ...) are taken from the MQT-Repository.


## Problem

The following problems are to be benchmarked.

### Permutation Equivalence

- A) Are two given Stabilizer Codes C and C' equivalent (define the same codespace) up to permutation of the output qubits?

$$C \thicksim_P \ C'  \quad \iff \quad \exists P \in S_n : \quad \exists R \in GL(2) : \quad \text{tableu}(C') = R \cdot \text{tableu}(C) \cdot \begin{pmatrix} P & 0 \\\ 0 & P\end{pmatrix} $$

### Local-Clifford Equivalence

- B) Are two given Stabilizer Codes C and C' Local-Clifford equivalent (define the same codespace) up to Local Clifford gates on the output qubits?

$$C \thicksim_{LC} \ C'  \quad \iff \quad \exists U = U_1 \otimes ... \otimes U_n \ \text{with} \ U_i \in \\{I, H ,S\\}^l : \quad rowspace(\text{tableu}(C')) = U \cdot rowspace(\text{tableu}(C)) \cdot U^\dagger = \\{U \cdot g \cdot U^\dagger \ | \ g \in rowspace(\text{tableu}(C)) \\} $$

- C) Is a given Stabilizer Code C Local-Clifford equivalent (define the same codespace up to Local Clifford gates on the output qubits) to a CSS Code?

$$\exists C_{CSS} \ \text{with} \ \text{tableu}(C_{CSS}) = \begin{pmatrix} H_x & 0 \\\ 0 & H_z\end{pmatrix} : \quad C \thicksim_{LC} \ C_{CSS} $$

## Approaches

### Permutation Equivalence
#### CSS Codes
- [x] Brute Force
- [x] Matroid Isomorphism
- [x] Classical Algorithms
- [x] Graph Isomorphism

#### Stabilizer Codes
- [x] Brute Force
- [x] Graph State Machinery
- [ ] Classical Algorithms
- [ ] Automorphisms
- [x] Graph Isomorphism
- [x] SAT

### Local-Clifford Equivalence
#### $C \thicksim_{LC} \ C'$
- [x] Graph State Machinery

#### $C \thicksim_{LC} \ C_{CSS}$
- [x] Brute Force
- [ ] KLS Normal Form
- [x] Graph State Machinery with LC Orbit


## Scope

I use the term "Benchmark" to measure the runtime of the python algorithms with an expected workload - input codes [[ $n,k,d$ ]] with $n$ ranging from $2$ to $\thicksim 50$.\
This Repo is **NOT** meant for more detailed benchmark or profiling analyses for now [^1]. The goal is to get a feeling for the different complexity classes of the algorithms, to make a more informed decision for the implementation in the MQT.\
Furthermore, the input is guaranteed to always be valid, thus the implemented core algorithms do not have to check small invariants.

[^1]: Maybe if I have the time, I will implement the algorithms in C++ and run actual benchmarks, but I doubt that will happen during the thesis :(

## Repository Structure

The repository is structured as a minimal local benchmark application.

```text
src/
  core/            # base QECC classes from MQT
    pauli.py
    symplectic.py
    stabilizer_code.py
    css_code.py
  algorithms/      # equivalence-checking implementations

benchmarks/
  run.py           # cases, timing, and CSV output
  utils.py         # randomization utilities

data/              # non-randomized case inputs

results/           # generated CSV output, ignored by git
```

## How to run the benchmarks?

Install the local runtime dependencies first if needed:

```bash
python3 -m pip install -r requirements.txt
```

Run benchmarks from the repository root:
```bash
python3 -m benchmarks.run
```

Certain parameters can be set as well, e.g. in the following commands
```bash
python -m benchmarks.run --repeats 1
python -m benchmarks.run --output results/bm_output.csv
python -m benchmarks.run --seed 69
python -m benchmarks.run --algorithm pm_css_bruteforce --algorithm lc_equ_graph_state
```

### Dependencies

For the Automorphism algorithm, I use [GAP / Guava](https://docs.gap-system.org/pkg/guava/doc/manual.pdf), thus the GAP executable is a dependency.
Before running a benchmark with this algorithm, the following command has to be executed:
```bash
export GAP_EXECUTABLE=/path/to/gap
```

