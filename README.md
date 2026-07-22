# ***E******Q***uivalence Checking for ***Q***uantum ***E***rror ***C***orrection **C**odes

This repository benchmarks different approaches to equivalence checking for quantum error-correcting codes (QECCs) under different equivalence notions. It is part of the implementation for my Bachelor's thesis, "Automated Equivalence Checking of Stabilizer Codes" [WIP], which contributes to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc). The required infrastructure and code representation classes are taken from that project.

This README does not discuss the examined equivalence notions or their theoretical foundations in depth; those are covered in the thesis. References for the implemented algorithms are cited in the corresponding source files.

This repository is currently not intended to be installed as a package. Its final hybrid algorithms are contributed to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc).

## Problems

The following problems are benchmarked. They are expressed using the repository's input representation for a QECC $C$: a binary symplectic stabilizer matrix $\text{S}(C) \in \mathbb{F}_2^{r \times 2n}$, or the parity-check matrices $\text{H}_x(C) \in \mathbb{F}_2^{r_x \times n}$ and $\text{H}_z(C) \in \mathbb{F}_2^{r_z \times n}$ when the code is CSS.

### Permutation Equivalence

- **PM-STB**: Are two given stabilizer codes $C$ and $C'$ equivalent up to a permutation of the output qubits?

$$ \exists P \in \mathfrak{S}_n : \enspace \exists R \in \mathrm{GL}(r, \mathbb{F}_2) : \quad \text{S}(\text{C}') = R \cdot \text{S}(\text{C}) \cdot \left[\begin{smallmatrix} P & 0 \\ 0 & P\end{smallmatrix}\right] $$

- **PM-CSS**: Are two given CSS codes $C$ and $C'$ equivalent up to a permutation of the output qubits?

$$ \exists P \in \mathfrak{S}_n : \enspace \exists R_x \in \mathrm{GL}(r_x, \mathbb{F}_2), R_z \in \mathrm{GL}(r_z, \mathbb{F}_2): \enspace
 \text{H}_x(\text{C}') = R_x \cdot \text{H}_x(\text{C}) \cdot P \enspace , \enspace \text{H}_z(\text{C}') = R_z \cdot \text{H}_z(\text{C}) \cdot P $$

### Local-Clifford Equivalence

- **LC-STB**: Are two given stabilizer codes $C$ and $C'$ local-Clifford equivalent, meaning that they define the same codespace up to local Clifford gates on the output qubits?

$$\exists Q = \left[\begin{smallmatrix} A & B \\ C & D \end{smallmatrix}\right] \in \text{Sp}(2n, \mathbb{F}_2): Q_i = \left[\begin{smallmatrix} a_{ii} & b_{ii} \\ c_{ii} & d_{ii} \end{smallmatrix}\right] \in \text{Sp}(2, \mathbb{F}_2): \enspace \exists R \in \mathrm{GL}(r, \mathbb{F}_2): \enspace \text{S}(\text{C}')  = R \cdot \text{S}(\text{C}) \cdot Q$$

- **LC-CSS**: Is a given stabilizer code $C$ local-Clifford equivalent to a CSS code?

$$ \exists \text{C}' \enspace \text{with} \enspace
  \text{S}(\text{C}') =
  \left[\begin{smallmatrix}
     \text{H}_x(\text{C}') & 0 \\
    0 &  \text{H}_z(\text{C}')
  \end{smallmatrix}\right]:$$
$$
\exists Q = \left[\begin{smallmatrix} A & B \\ C & D \end{smallmatrix}\right] \in \text{Sp}(2n, \mathbb{F}_2): Q_i = \left[\begin{smallmatrix} a_{ii} & b_{ii} \\ c_{ii} & d_{ii} \end{smallmatrix}\right] \in \text{Sp}(2, \mathbb{F}_2): \enspace \exists R \in \mathrm{GL}(r, \mathbb{F}_2): \enspace \text{S}(\text{C}') = R \cdot \text{S}(\text{C})\cdot Q
$$

## Approaches

### PM-STB

- [x] Brute-force Search
- [x] Classical Approaches
- [x] Automorphism Groups
- [x] Graph Isomorphism
- [x] SAT

### PM-CSS

- [x] Brute-force Search
- [x] Classical Approaches
- [x] Graph Isomorphism
- [x] Matroid Isomorphism
- [x] SAT

### LC-STB

- [x] Brute-force Search
- [x] Graph Isomorphism
- [x] Graph-state LSE **(only valid for $k < 2$ or with restrictions on logical operators)**
- [x] KLS normal form and LC orbit
- [x] SAT

### LC-CSS

- [x] Brute-force Search
- [x] KLS normal form and LC orbit
- [x] Clifford orbit
- [x] LC orbit **(only valid for $k < 2$ or with restrictions on logical operators)**
- [x] SAT

## Scope

Here, a benchmark measures the runtime of the Python algorithms on the expected workload: input codes $[[n,k,d]]$ with $n$ ranging from 2 to approximately 50, plus some larger structured cases.

This repository is not currently intended for detailed benchmarking or profiling analyses.[^1] The goal is to understand the algorithms' different complexity classes and make a more informed decision about the hybrid implementations.

Inputs are guaranteed to be valid, so the core algorithms do not need to verify basic validity conditions. Simple and more involved invariants for the different equivalence notions are collected separately.

[^1]: A future C++ implementation could support more rigorous benchmarks, but that is currently outside the scope of the thesis.

## Repository Structure

The repository is structured as a minimal local benchmark application.

```text
src/
  core/              # base QECC classes from MQT-QECC
    pauli.py
    symplectic.py
    stabilizer_code.py
    css_code.py
  algorithms/        # equivalence-checking implementations
    lc_css/
    lc_stb/
    p_css/
    p_stab/
  invariants/        # invariants under the equivalence relations
  hybrids/           # hybrid solutions combining best aspects of the algorithms
    lc_css.py
    lc_stb.py
    p_css.py
    p_stab.py

tests/               # partially randomized tests and edge-case tests
  lc_css/
  lc_eq/
  lc_stb/
  p_css/
  p_stab/

benchmarks/
  run.py             # cases, timing, and CSV output
  utils.py           # randomization utilities
  run_invariants.sh
  run_multiple.sh

data/                # non-randomized case inputs
  convert.py
  generate.py

results/             # plotting tools; generated result artifacts are ignored by git
  visualize_named.py
  visualize_random.py
  visualize_invariants.py
```

## Running the benchmarks

Install the local runtime dependencies first:

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

- `--raw`, the default when no mode is specified, benchmarks the default known or random cases. It is useful for flamegraphs and profiling.
- `--stats` benchmarks multiple seeded instances with fixed dimensions and calculates aggregate statistics. It is useful for broader algorithm evaluation and detecting edge cases.
- `--inv` runs a fixed suite for all invariant-checking algorithms. It is useful for comparing invariants.

The `--random`, `--algorithm`, `--nmin`, and `--nmax` arguments can only be used in raw or statistics mode, not with `--inv`.

For each algorithm, the range of benchmarked instance sizes can be customized with `--nmin` and `--nmax`.

Two optional resource guards are available:

- `--timeout` limits the execution time of each repeat in seconds.
- `--memory-limit` limits the memory available to each benchmark child process. Accepted forms include `4096M`, `32G`, and `16GiB`.

Other parameters include `--output`, `--seed`, and `--verbose`. For example:

```bash
python3 -m benchmarks.run --seed 69 --output results/bm.csv --verbose --timeout 200

python3 -m benchmarks.run --algorithm pm_css_bruteforce --algorithm lc_stb_lse
python3 -m benchmarks.run --algorithm 'pm_css*'

python3 -m benchmarks.run --raw --random --nmin 5 --nmax 8
python3 -m benchmarks.run --stats --algorithm pm_css_sat --random --verbose
python3 -m benchmarks.run --inv --verbose --output results/invariants.csv --timeout 20 --memory-limit 512M
```

Optionally, record a flamegraph for more detailed analysis:

```bash
py-spy record \
  --format flamegraph \
  --output results/pm_css_matroid_flame.svg \
  -- python -m benchmarks.run --algorithm pm_css_matroid --output results/pm_css_matroid_profile.csv --random
```

The Bash scripts `run_invariants.sh` and `run_multiple.sh` run customizable benchmark suites in parallel on the benchmark server.

### Evaluation

The visualization scripts in `results/` create plots that provide a quick overview of performance and bottlenecks; they are not intended as a finalized analysis. Use `visualize_named.py` for known or structured benchmark cases, `visualize_random.py` for randomized benchmark statistics (`--stats` mode), and `visualize_invariants.py` for invariant timings (`--inv` mode).

For example:

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

python3 results/visualize_random.py results/sat.csv results/matroid.csv --x n --k 1 \
  --output results/compare_csvs.png

python3 results/visualize_invariants.py results/lc_invariants.csv --x n \
  --output results/lc_invariants_plot.png
```

### Tests

[![Tests](https://github.com/MaxieHelenBichmann/bm_qecc/actions/workflows/tests.yml/badge.svg)](https://github.com/MaxieHelenBichmann/bm_qecc/actions/workflows/tests.yml)

The test suite is located in `tests/`. They include unit, regression and randomized tests. Some algorithm sections do not yet have comprehensive coverage.

```bash
python3 -m pytest
```

### Dependencies

Apart from the dependencies in `requirements.txt`, the automorphism-group algorithm uses [GAP and Guava](https://docs.gap-system.org/pkg/guava/doc/manual.pdf), so a GAP executable and Guava's dependencies are required. Place the Guava dependencies in `bm_qecc/.gap` and set the path to the GAP executable before running this algorithm:

```bash
export GAP_EXECUTABLE=/path/to/gap
```

This approach is expected to be less efficient than the alternatives and is included primarily for comparative measurements.
