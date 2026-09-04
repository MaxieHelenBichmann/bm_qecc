# Replication Package for "Methods for Automated Equivalence Checking of Quantum Error Correction Codes" 

Everything needed to reproduce the figures of the associated paper lives in the directory `paper/`. 
It contains server-side measurement scripts, the deterministic aggregation step, and plotting entry points.

Nothing here is a library. Every stage is a runnable `python3 -m` entry point, run from the **repository root**.
The artifacts of this pipeline are not the exact same figures used in the paper, as those are LaTeX-native, but they represent the data in the same way. The measurements in the paper were collected on a server with the following hardware and software configuration:


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

Figures and Tables are referred to everywhere in this package as **A1 … A8**, and figure-specific files carry **only** the `aN` identifier:

| ID | Related Question | Figure |
|---|---|---|
| **A1** | Are invariants even useful? | `results/a1/a1.png` |
| **A2** | Are signatures even useful? | `results/a2/a2.png` |
| **A3** | When are invariants useful? | `results/a3/a3.png` |
| **A4** | How do methods eliminating the representation degree of freedom perform? | `results/a4/a4.png` |
| **A5** | Which exact method performs best? | `results/a5/a5.png` |
| **A6** | Is SAT's poor CSS behavior caused by the encoding or by the CSS inputs? | `results/a6/a6.png` |
| **A7** | Why does SAT perform so poorly on CSS Permutations? | `results/a7/a7.png` |
| **A8** | How do the hybrids perform? | `results/a8/a8.png` |

As the paper focuses on pairwise equivalence checking, only three problems from the underlying thesis are in scope: `pm_stb` (permutation equivalence of stabilizer codes), `pm_css` (permutation equivalence of CSS codes), and `lc_stb` (local-Clifford equivalence of stabilizer codes).

---

## Layered Data Collection

```text
  PHASE 1  collect                 PHASE 2  extract              PHASE 3  visualize
  benchmarks/collect_*.py          experiments/extract_a<N>.py   visualizations/visualize_a<N>.py
  - primarily collection,          - deterministic, local,       - primarily plotting
    hours-to-days, server            seconds, CSV→CSV
         │                                   │                             │
         ▼                                   ▼                             ▼
  data/collected/       ───────────►  results/a<N>/by_cell.csv  ─────────►  results/a<N>/a<N>.png
   (raw or batch summaries)             (figure-ready, per cell)
```

This pipeline has three phases for each experiment, with strict boundaries, making the package replicable:

- Phase 1 primarily collects measurements and is the only phase that generates codes, runs algorithms, or consumes meaningful compute time. Depending on the collector, it stores either per-instance rows or summaries of a seeded batch, but never writes into `results/`.
- Phase 2 performs deterministic extraction, including the main selection and aggregation steps (winner choice, backend choice, eligibility rules, and normalization). It never generates inputs or runs a benchmark algorithm.
- Phase 3 primarily draws the figures from `results/`. Visualizers may apply presentation-specific grouping, range restrictions, or annotations, but never read from `data/collected/`.

Phase 1 normally runs on a benchmark server; phases 2 and 3 run locally. The transfer between machines is exactly the contents of `data/collected/`.

### Phase 1 — Data Collection

Long-running. Run under `tmux` or an equivalent. Almost every collector appends incrementally and resumes by skipping keys already present. Collectors based on shared batch statistics instead append one summary row per completed batch; re-running such a batch may repeat its computation, while extraction keeps its latest row. Delete the relevant output file or files only to deliberately restart a collection from scratch.
Collected data from `collect_algorithm.py` is not figure-specific, but used by multiple aggregators in the next steps.

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

Reads CSVs from `results/a<N>/` and writes one `results/a<N>/a<N>.png` or multiple PNGs, depending on the experiment.

---

## Measurement Definitions

### A1 — Rejection Rates

|  |  |
|---|---|
| **Question** | How many—and which—input instances are rejected by the utilized invariants? |
| **Invariants and signatures measured** | Linear dependency and punctured-hull/Sendrier signatures on general stabilizer and CSS codes; the degree-2 local invariant on general stabilizer codes |
| **Method** | 10 certified-negative randomized instances per parameter setting and record rejection counts; no runtimes are measured directly |

The result value is how many of the 10 were rejected, per invariant and combined per equivalence notion.
The aggregated table shows the percentage of rejected instances per invariant and code family.

> The generation of randomized instances has to be carefully considered here, as otherwise selection bias has a significant effect on the results.
> Generating two tableaus (of the same dimensions) completely independently and certifying their inequivalence leads to high rejection rates, as usually fully independent tableaus are structurally very different. This however might not represent practical instances considered in equivalence checking, as two actually compared codes might usually be somewhat related.
> Additionally, the certification method of their inequivalence might introduce a selection bias. When certifying their inequivalence by a mismatched invariant due to runtime constraints, this must not be the invariant whose rejection rate is being measured, or the estimate is circular and the pair will be rejected by construction.
>
> Therefore:
> - general stabilizer codes: apply a short random Clifford circuit (`STABILIZER_CLIFFORD_GATE_STEPS`) to one source code, then keep the candidate only if the corresponding exact SAT backend proves inequivalence. This yields structurally related negatives selected by an exact backend rather than by a measured invariant.
> - CSS codes: apply a short physical-CNOT circuit (`CSS_CNOT_GATE_STEPS`) to one source code, then retain the candidate only when SAT or matroid isomorphism proves inequivalence. The perturbation preserves the CSS form and both check ranks without consulting a measured invariant. Parameter sizes outside this exact-certifier region keep using `css_codes_cascaded`, which emits a negative carrying its own permutation-invariant certificate; that certificate can correlate with a measured invariant.

### A2 — Signature Space

|  |  |
|---|---|
| **Question** | How well does the column partition induced by a signature refine the permutation search space? |
| **Signatures measured** | Sendrier signatures, as implemented for `pm_css` and `pm_stb`, on CSS and general stabilizer codes |
| **Method** | 10 randomized codes per parameter setting; no code pairs, equivalence labels, or direct runtime measurements |

The result value of a parameter setting is the mean of the 10 seeds, of how the Sendrier signatures partition the physical qubits of the code $J_1, J_2, ...$, as a normalized fraction
$$
  \bar{q} = 1 - \frac{q - \frac{1}{n}}{1 - \frac{1}{n}}
       = \frac{1-q}{1-\frac{1}{n}} \in [0, 1] \text{ with } q = \sum\limits_{i} \frac{|J_i|^2}{n^2} \in [\frac{1}{n}, 1]
$$
Here, `0` means one undivided class (no refinement), and `1` means
every class is a singleton (complete refinement). This is a pairwise refinement
score, not the literal fraction of the $n!$ permutation search space removed.
Parameter settings where each seed is censored (due to timeout) are left uncolored.

### A3 — Relative Preprocessing Cost

|  |  |
|---|---|
| **Question** | Does computing an invariant take longer than running a complete decision-procedure backend? |
| **Invariants and signatures measured** | Linear dependency and punctured-hull/Sendrier signatures on general stabilizer and CSS codes; the degree-2 local invariant on general stabilizer codes |
| **Method** | 5 positive and 5 negative randomized instances per parameter setting; invariant runtime compared with the best-performing backend from A5 |

The result value is the mean of the comparisons of the runtime of the invariants to the runtimes of the best-performing backend for this parameter setting (see A5), thus showcasing the worst-case for invariant usability
$$
 \frac{T_{\text{invariant}}}{T_{\text{backend}}}
$$

### A4 — Representation Cost

|  |  |
|---|---|
| **Question** | Where do methods trade fast search for excessive representation cost? |
| **Algorithms measured** | Graph-isomorphism-based algorithms for `pm_stb` and `lc_stb` on general stabilizer codes; the matroid-isomorphism-based algorithm for `pm_css` on CSS codes |
| **Method** | 10 positive and 10 negative randomized instances per parameter setting; mean runtime and occurrence of memory errors |

The result value for a parameter setting is the mean runtime of the instances, explicitly marking runs where at least one instance resulted in a memory error.


### A5 — Best-Performing Methods

|  |  |
|---|---|
| **Question** | Which exact algorithm performs best for each parameter setting? |
| **Algorithms measured** | All algorithms discussed in the paper for `pm_stb`, `pm_css`, and `lc_stb` on their corresponding code families |
| **Method** | 10 positive and 10 negative randomized instances per parameter setting; eligible algorithm with the lowest mean runtime selected per setting |

For each parameter setting, the method with the lowest mean runtime is extracted, under the condition that is has no memory errors. A memory error is considered a more sever type of error than a timeout.

### A6 — SAT on CSS-Code Permutation Equivalence

|  |  |
|---|---|
| **Question** | How does SAT perform with the tableau and check-matrix encodings on CSS codes, compared with its performance on general stabilizer codes? |
| **Algorithms measured** | `pm_css_sat` and `pm_stb_sat` on CSS codes; `pm_stb_sat` on general stabilizer codes |
| **Method** | 10 positive and 10 negative randomized instances per parameter setting; comparison of mean runtimes |

The result value for a parameter setting is the mean runtime of the instances.

### A7 — SAT Encodings on CSS-Code Permutation Equivalence

|  |  |
|---|---|
| **Question** | Why does SAT perform so poorly on CSS permutation equivalence, even with the check-matrix encoding? |
| **Algorithms measured** | `pm_stb_sat` on unrestricted and block-structured general stabilizer codes; `pm_css_sat` on balanced CSS codes |
| **Method** | 10 positive randomized instances per parameter setting; no direct runtime measurements |

Two experiments are measured, first the number of solver decisions required to solve an code size and the number of decisions required to reject deliberately wrong qubit mappings. Both on instances with different amount of (in)dependent (un)coupled row-transformations. 
The second compares clean/separated and fully row-mixed presentations of the same CSS groups using the `pm_stb_sat` encoding.

### A8 — Hybrid Component Attribution

|  |  |
|---|---|
| **Question** | What runtimes do the hybrids achieve, and which component actually decides each input? |
| **Algorithms measured** | Paper hybrids, rather than the thesis hybrids whose MQT-QECC integration follows a maintainability-oriented design strategy |
| **Method** | 10 positive and 10 negative instances derived from structured codes, such as the Steane and bivariate bicycle codes; runtime and deciding component recorded |

The result value is comprised of measured runtime as well as diagnostic information about which component decided the input.
Component tags are `CI` (cheap invariants), `EI` (expensive invariants), `S` (signatures), and the decision procedures `BF` (brute force), `MI` (matroid isomorphism), `GI` (graph isomorphism), `SAT`, and `LSE`.

---

## Design Rules

Collector-specific constants, input populations, certification choices, and output paths are kept close to each collection entry point. Most collectors also expose their resume keys and CSV persistence directly; shared generators, supervision/statistics utilities, and the diagnostic paper hybrids are reused where appropriate.

---

## AI Usage Transparency

For components shared with the main project, AI usage is disclosed in the top-level README. Within this replication package, AI tools assisted with implementing collectors, extractors, and visualizers, but not with their conceptual design. The research ideas, experimental methodology, data flow, visualization design, supervision strategy, and documentation were developed by the human author. All AI-assisted code was reviewed and validated by the author.
