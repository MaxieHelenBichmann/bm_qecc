# Replication Package for "Methods for Automated Equivalence Checking of Quantum Error Correction Codes" 

Everything needed to reproduce the figures of the associated paper lives in the directory `paper/`. 
It contains server-side measurement scripts, the deterministic aggregation step, and plotting entry points.

Nothing here is a library. Every stage is a runnable `python3 -m` entry point, run from the **repository root**.
The artifacts of this pipeline are not the exact same figures used in the paper, as those are LaTeX-native, but they represent the data in the same way. The data in the paper was collected on a server with the following specifications:


| Hardware | | Software | |
|---|---|---|---|
| Architecture | x86\_64 | Operating system | Debian GNU/Linux 13.5 |
| Processor | AMD Ryzen 7 3700X (8 cores) | Kernel | Linux 6.12.90+deb13.1-amd64 |
| Memory | 32 GB DDR4-3200 (2 × 16 GB, CL16) | Python | 3.13.5 |
| | | NumPy | 2.4.6 |
| | | LDPC | 2.4.1 |
| | | PyNauty | 2.8.8.1 |
| | | Z3 Solver | 4.16.0 |

---

## Figures

Figures are referred to everywhere in this package as **A1 … A7**, and figure-specific files carry **only** the `aN` identifier:

| ID | Related Question | Figure |
|---|---|---|
| **A1** | Are invariants even useful? | `results/a1/a1.png` |
| **A2** | Are signatures even useful? | `results/a2/a2.png` |
| **A3** | When are invariants useful? | `results/a3/a3.png` |
| **A4** | How do methods eliminating the representation degree of freedom perform? | `results/a4/a4.png` |
| **A5** | Which exact method performs best? | `results/a5/a5.png` |
| **A6** | Is SAT's poor CSS behavior caused by the encoding or by the CSS inputs? | `results/a6/a6.png` |
| **A7** | How do the hybrids perform? | `results/a7/a7.png` |

As the paper focuses on pairwise equivalence checking, three problems are in scope: `pm_stb` (permutation equivalence of stabilizer codes), `pm_css` (permutation equivalence of CSS codes), and `lc_stb` (local-Clifford equivalence of stabilizer codes). `lc_css` is out of scope.

---

## Layered Data Collection

```text
  PHASE 1  collect                 PHASE 2  extract              PHASE 3  visualize
  benchmarks/collect_*.py          experiments/extract_a<N>.py   visualizations/visualize_a<N>.py
  - resource-supervised,           - deterministic, local,       - plotting only, 
    hours-to-days, server            seconds, pure CSV→CSV         no selection logic
         │                                   │                             │
         ▼                                   ▼                             ▼
  data/collected/*.csv  ───────────►  results/a<N>/by_cell.csv  ─────────►  results/a<N>/a<N>.png
     (raw, per instance)                (figure-ready, per cell)
```

This pipeline hss three phases for each experiment, with strict boundaries, making the package replicable:

- Phase 1 is the only phase that generates codes, runs algorithms, or consumes meaningful compute time. It produces the necessary data for the following steps, but never aggregates or writes into `results/`.
- Phase 2 is the only phase that selects and aggregates (winner choice, backend choice, eligibility rules, normalization). It never generates inputs and never runs a benchmark algorithm.
- Phase 3 is the only phase that draws. It reads `results/` exclusively and never touches `data/collected/`.

Phase 1 normally runs on a benchmark server; phases 2 and 3 run locally. The transfer between machines is exactly the contents of `data/collected/`.

### Phase 1 — Data Collection

Long-running. Run under `tmux` or an equivalent; each collector appends to its CSV incrementally and **resumes** by skipping keys already present, so an interrupted run continues without duplicating completed work. Delete an output file only to deliberately restart that collection from scratch.
Collected data from `collect_algorithm.py` is not figure-specific, but used by  multiple aggregators in the next steps.

```bash
python3 -m paper.benchmarks.collect_*
```

Writes CSV data into `data/collected/`.

### Phase 2 — Information Extraction

Fast, deterministic, safe to re-run at any time.

```bash
python3 -m paper.experiments.extract_a<N>
```

Reads CSVs from `data/collected/` and writes CSV data into `results/a<N>/`.

### Phase 3 — Information Visualization

Fast, deterministic, safe to re-run at any time.

```bash
python3 -m paper.visualizations.visualize_a<N>
```

Reads CSVs from `results/a<N>/` and writes exactly one PNG, `results/a<N>/a<N>.png`.

---

## Measurement Definitions

### A1 — Rejection Rates
*How many and which input instances are rejected by the utilized invariants?*
Invariants measured: linear dependency and punctured-hull/Sendrier signatures for `pm_stb` and `pm_css`; the degree-2 low-degree local invariant for `lc_stb`. 
10 certified-negative randomized instances per parameter setting; no runtime measurement. 
The result value is how many of the 10 were rejected, per invariant and combined per equivalence notion.

The generation of randomized instances has to be carefully considered here, as otherwise selection bias has a significant effect on the results.
Generating two tableaus (of the same dimensions) completely independently and certifying their inequivalence leads to high rejection rates,  as usually fully independent tableaus are structurally very different. This however might not represent practical instances considered in equivalence checking, as two actually compared codes might usually be somewhat related.
Additionally, the certification method of their inequivalence might introduce a selection bias. When certifying their inequivalence by a mismatched invariant due to runtime constraints, this must not be the invariant whose rejection rate is being measured, or the estimate is circular and the pair will be rejected by construction.

Therefore:

- `pm_stb` / `lc_stb`: apply a short random Clifford circuit (`STABILIZER_CLIFFORD_GATE_STEPS`) to one source code, then keep the candidate only if the corresponding exact SAT backend proves inequivalence. This yields structurally related negatives selected by an exact backend rather than by a measured invariant.
- `pm_css`: two independent draws sharing one X-check rank. Because `r_x + r_z = n - k`, this fixes a shared Z-check rank too and prevents trivial rank-mismatch negatives. Certification uses SAT for `r ≤ 9` and matroid isomorphism for `r > 9, n ≤ 28`, as the verification on those parameter sizes is practically possible. Parameter sizes outside this region falls back to `css_codes_cascaded`, which emits a negative carrying its own permutation-invariant certificate. That certificate can correlate with a measured invariant.

### A2 — Signature Space
*How well does column partition induced by a signature refine the permutation search space?*
Signatures measured: Sendrier using implementation of `pm_css` and `pm_stb`
No pairs, no equivalence labels, no runtime. For each seed, generate one unconditioned random code and measure how its Sendrier signature partitions the physical qubits. 

For a column partition $s_1, s_2, ...$ collector stores the quantity
$$
  q = \sum\limits_{i} \frac{|s_i|^2}{n^2} \in [\frac{1}{n}, 1]
$$
which the extractor normalizes to 
$$
  q_{\text{norm}} = \frac{q - \frac{1}{n}}{1 - \frac{1}{n}} \in [0, 1]
$$
representing `0` with every class being a singleton (complete refinement), and `1` as one undivided class (no refinement).

### A3 — Relative Preprocessing Cost
*Does the computation of an invariant actually take longer than a full decision procedure backend?*
Invariants measured: linear dependency and punctured-hull/Sendrier signatures for `pm_stb` and `pm_css`; the degree-2 low-degree local invariant for `lc_stb`. 
5 positive and 5 negative randomized instances per parameter setting.

For negative instances similar concerns about the generation method occur as for A1, to not bias the mean runtime due to an unrepresentative number of early rejections.

The extractor compared the runtime of the invariants to the runtimes of the best-performing backend for this parameter setting (see A5), thus showcasing the worst-case for invariant usability
$$
 \frac{T_{\text{invariant}}}{T_{\text{backend}}}
$$

The visualizer creates one temperature map per invariant.

### A4 - Representation Cost
*Where do methods trade fast search for excessive representation cost?*
Methods measured: graph-isomorphism based methods for `pm_stb` and `lc_stb` on general stabilizer codes.
10 positive and 10 negative randomized instances per parameter setting.

The visualizer creates standard mean runtime heatmaps, explicitly marking runs where at least one instance resulted in a memory error.


### A5 - Best-Performing Methods
*Which method performs best for each parameter setting?*
Methods measured: all proposed methods for `pm_stb`, `pm_css` and `lc_stb` on their according input codes.
10 positive and 10 negative randomized instances per parameter setting.

For each parameter setting, the method with the lowest mean runtime is extracted, under the condition that is has no memory errors. A memory error is considered a more sever type of error than a timeout.

### A6 - SAT on CSS-Code Permutation Equivalence 
*How does the SAT method perform using the tableau or parity-check-matrix encoding on CSS codes, compared to general stabilizer codes?*
Methods measured: `pm_css_sat` and `pm_stb_sat` on CSS codes, and `pm_stb_sat` on general stabilizer codes.
10 positive and 10 negative randomized instances per parameter setting.

The visualizer creates standard mean runtime heatmaps.


### A7 — Hybrid Component Attribution
*What runtimes do the hybrids achieve and which component actually decides the input?*
Methods measured: paper hybrids (NOT thesis hybrids, as those are designed with maintainability in mind due to their integration in MQT-QECC, and follow a different design strategy)
10 positive and 10 negative instances based on real structures codes such as the Steane code or bivariate bicycle codes.

The result value is the measured runtime as well as diagnostic information about which component decided the input.
Component tags are `CI` (cheap invariants), `EI` (expensive invariants), `S` (signatures), and the decision procedures `BF` (brute force), `MI` (matroid isomorphism), `GI` (graph isomorphism), `SAT`, and `LSE`.

---

## Design Rules

These are the constraints the code is written to:

- **Every collector is self-contained.** Its constants, input generation, certification choices, invariant calls, resume keys, and CSV persistence are all visible in that one file. Some small helpers are intentionally duplicated between collectors so that a server-side run does not depend on any paper-specific helper package. Collectors reuse only the repository's generic generators, supervised runner, and statistics machinery.
- **Paired inputs.** Every method in one comparison receives the exact same serialized code pair for a given `instance_id`. Generate an input once and reuse it; never regenerate an allegedly matching input per algorithm.
