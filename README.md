# Benchmarking Equivalence Checking QECC

This Repository is used for Benchmarking different approaches for Equivalence Checking QECC. It is used as part of the implementation for my Bachelor's Thesis "Automated Equivalence Checking of Stabilizer Codes" \[WIP\], which contributes to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc). Thus, the required infrastructure classes (e.g. `CSSCode`, `Pauli`, ...) are taken from the MQT-Repository.


## Problem

The following problems are to be benchmarked.

### Permutation Equivalence

- A) Are two given Stabilizer Codes C and C' equivalent (define the same codespace) up to permutation of the output qubits?

$$C \thicksim_P \ C'  \quad \iff \quad \exists P \in S_n : \quad \exists R \in GL(2) : \quad \text{tableau}(C') = R \cdot \text{tableau}(C) \cdot \begin{pmatrix} P & 0 \\\ 0 & P\end{pmatrix} $$

### Local-Clifford Equivalence

- B) Are two given Stabilizer Codes C and C' Local-Clifford equivalent (define the same codespace) up to Local Clifford gates on the output qubits?

$$C \thicksim_{LC} \ C'  \quad \iff \quad \exists U = U_1 \otimes ... \otimes U_n \ \text{with} \ U_i \in \\{I, H ,S\\}^l : \quad rowspace(\text{tableau}(C')) = U \cdot rowspace(\text{tableau}(C)) \cdot U^\dagger = \\{U \cdot g \cdot U^\dagger \ | \ g \in rowspace(\text{tableau}(C)) \\} $$

- C) Is a given Stabilizer Code C Local-Clifford equivalent (define the same codespace up to Local Clifford gates on the output qubits) to a CSS Code?

$$\exists C_{CSS} \ \text{with} \ \text{tableau}(C_{CSS}) = \begin{pmatrix} H_x & 0 \\\ 0 & H_z\end{pmatrix} : \quad C \thicksim_{LC} \ C_{CSS} $$

## Approaches

### Permutation Equivalence
#### CSS Codes
- [x] Brute Force
- [x] Matroid Isomorphism
- [x] Classical Algorithms
- [x] Graph Isomorphism
- [x] SAT

#### Stabilizer Codes
- [x] Brute Force
- [x] Classical Algorithms
- [x] Automorphisms
- [x] Graph Isomorphism
- [x] SAT

### Local-Clifford Equivalence
#### $C \thicksim_{LC} \ C'$
- [x] Brute Force
- [x] Graph State Machinery **(ONLY VALID FOR _k < 2_ OR _RESTRICTIONS ON LOGICAL OPERATORS_)**
- [x] KLS Normal Form + LC Orbit
- [x] Graph Isomorphism
- [x] SAT

#### $C \thicksim_{LC} \ C_{CSS}$
- [x] Brute Force
- [x] Simple Choi + LC Orbit **(ONLY VALID FOR _k < 2_ OR _RESTRICTIONS ON LOGICAL OPERATORS_)**
- [x] Simple Choi + Clifford Orbit
- [x] KLS Normal Form + LC Orbit
- [x] SAT


## Scope

I use the term "Benchmark" to measure the runtime of the python algorithms with an expected workload - input codes [[ $n,k,d$ ]] with $n$ ranging from $2$ to $\thicksim 50$.\
This Repo is **NOT** meant for more detailed benchmark or profiling analyses for now [^1]. The goal is to get a feeling for the different complexity classes of the algorithms, to make a more informed decision for the implementation in the MQT.\
Furthermore, the input is guaranteed to always be valid, thus the implemented core algorithms do not have to check small invariants.\
Trivial and more complex invariants for the different equivalence notions are collected separately. 

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
    lc_css/
    lc_eq/
    p_css/
    p_stab/
  invariants/      # invariants under equivalence-relations
  hybrids/         # hybrid solutions combining best aspects of the algorithms
    lc_css.py
    lc_eq.py
    p_css.py
    p_stab.py

tests/             # partially randomized and edge case tests
  lc_css/
  lc_eq/
  p_css/
  p_stab/

benchmarks/
  run.py           # cases, timing, and CSV output
  utils.py         # randomization utilities

data/              # non-randomized case inputs
  convert.py
  generate.py

results/           # generated CSV output, ignored by git
  visualize_named.py
  visualize_random.py
  visualize_invariants.py
```

## How to run the benchmarks?

Install the local runtime dependencies first if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run benchmarks from the repository root:
```bash
python3 -m benchmarks.run
```

The benchmark runner has three mutually exclusive modes:
- `--raw` which is the default when no mode flag is given, runs benchmarks on default known or random cases *(good for flamegraphs/profiling)*
- `--stats` which runs a suite of multiple codes of certain fixed dimensions, but different random instances, and then calculates statistics out of the different seeded instances *(good for broader algorithm evaluation/edge cases)*
- `--inv` which runs a fixed suite as input for all invariant-checker algorithms *(good for invariant comparison)*

$\rightarrow$ `--random` and `--algorithm` can only be used with raw and stats mode.

Apart from the hardcoded limits for n for each algorithm (they are in place because running a bruteforce algorithm on a code with n=25 is simply unrealistic and only wastes resources), the limits of each run can be customized with `--nmin` and `--nmax`.

I included two optional guards:
- `--timeout` which restricts the algorithm execution in the time dimension
- `--memory-limit` which restricts the algorithm execution in the space dimension

Other parameters (`--output`, `--seed`, `--verbose`) are more straightforward, example commands are:

```bash
python3 -m benchmarks.run --seed 69 --output results/bm.csv --verbose --timeout 200

python3 -m benchmarks.run --algorithm pm_css_bruteforce --algorithm lc_equ_graph_state
python3 -m benchmarks.run --algorithm 'pm_css*'

python3 -m benchmarks.run --raw --random --nmin 5 --nmax 8
python3 -m benchmarks.run --stats --algorithm pm_css_sat --random --verbose
python3 -m benchmarks.run --inv --verbose --output results/invariants.csv --timeout 20 --memory-limit 512M
```

Run more detailed analysis, optionally:
```bash
py-spy record \
  --format flamegraph \
  --output results/pm_css_matroid_flame.svg \
  -- python -m benchmarks.run --algorithm pm_css_matroid --output results/pm_css_matroid_profile.csv --random
```

### Evaluation

The visualization scripts in `results/` can create some small plots for getting a quick overview of performance and bottlenecks, not as a finalized analysis. Use `visualize_named.py` for known/structured benchmark cases, `visualize_random.py` for randomized benchmark statistics (`--stats` mode), and `visualize_invariants.py` for invariant timings (`--inv` mode).

Create plots, optionally:
```bash
python3 results/visualize_named.py results/statistics.csv --x r \
 --algorithm pm_css_matroid \
 --output results/matroid_plot_n17.png

python3 results/visualize_random.py results/statistics.csv --x n --k 1 \
  --algorithm pm_css_sat \
  --output results/sat_plot_k1.png

python3 results/visualize_random.py results/statistics.csv --x n --k 2 \
  --algorithm pm_stb_bruteforce --algorithm pm_stb_classical \
  --output results/compare.png

python3 results/visualize_invariants.py results/invariants_lc.csv --x n \
  --output results/invariants_lc_plot.png
```


### Tests

[![Tests](https://github.com/MaxieHelenBichmann/bm_qecc/actions/workflows/tests.yml/badge.svg)](https://github.com/MaxieHelenBichmann/bm_qecc/actions/workflows/tests.yml)

There are also **some** (not yet extensive) tests in the directory `tests/`. For the near future, they won't be particularly substantial, still missing tests for some algorithm sections.
```bash
python3 -m pytest
```

### Dependencies

For the Automorphism-based algorithm, I use [GAP / Guava](https://docs.gap-system.org/pkg/guava/doc/manual.pdf), thus the GAP executable and Guava's dependencies are needed.
Before running a benchmark with this algorithm, Guava's dependencies should be in `bm_qecc/.gap` AND the following command has to be executed:
```bash
export GAP_EXECUTABLE=/path/to/gap
```
But as I would estimate that this Automorphism-based algorithm will NOT be efficient at all, it will probably not be used in the MQT and thus I will not deal with this dependency apart from getting some measurements in this Repo.
