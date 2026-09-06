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
| | | Stim | 1.16.0 |
| | | Matplotlib | 3.11.1 |
| | | PyNauty | 2.8.8.1 |
| | | Z3 Solver | 4.16.0.0 |
| | | PyZX | 0.10.6 |
| | | pytest | 9.1.1 |
| | | py-spy | 0.4.2 |
| | | GAP + Guava | exact measurement versions were not recorded; see below |

`requirements.txt` pins every direct Python dependency. For the exact resolved
transitive environment, install `requirements.lock.txt` instead.

---

## Figures

Figures and Tables are referred to everywhere in this package as **A1 … A8**, and figure-specific files carry **only** the `aN` identifier:

| ID | Related Question | Collector(s) | Collected file(s) | Figure(s) |
|---|---|---|---|---|
| **A1** | Are invariants even useful? | `collect_a1` | `invariant_rejections.csv` | `paper/results/a1/a1.png`<br>`paper/results/a1/a1_overall.png` (overall rejection-rate table) |
| **A2** | Are signatures even useful? | `collect_a2` | `signature_space.csv` | `paper/results/a2/a2.png` |
| **A3** | When are invariants useful? | `collect_a3` + `collect_algorithm` | `invariant_timings.csv`, `algorithms/*.csv` | `paper/results/a3/a3.png` |
| **A4** | How do methods eliminating the representation degree of freedom perform? | `collect_algorithm` | `algorithms/{pm_stb_graph_iso,lc_stb_graph_iso,pm_css_matroid}.csv` | `paper/results/a4/a4.png` |
| **A5** | Which exact method performs best? | `collect_algorithm` | `algorithms/*.csv` (all configured methods) | `paper/results/a5/a5.png` |
| **A6** | Is SAT's poor CSS behavior caused by the encoding or by the CSS inputs? | `collect_a6` + `collect_algorithm` | `pm_stb_sat_on_css.csv`, `algorithms/pm_{stb,css}_sat.csv` | `paper/results/a6/a6.png`<br>`paper/results/a6/a6_css.png` (direct CSS-encoding comparison) |
| **A7** | Why does SAT perform so poorly on CSS Permutations? | `collect_a7` | `a7_sat_css_structure.csv` | `paper/results/a7/a7.png` |
| **A8** | How do the hybrids perform? | `collect_a8` | `hybrids_a8_v4/{campaign.json,inputs/,proposals/,stages/,instances.csv,summary.csv}` | `paper/results/a8/a8.png` |

All collected-file paths in this table are relative to
`paper/data/collected/`. The top-level `data/` and `results/` directories
belong to the thesis benchmark pipeline and are unrelated to this package.

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
  paper/data/collected/ ───────────►  paper/results/a<N>/by_cell.csv ─────► paper/results/a<N>/a<N>.png
   (raw or batch summaries)             (figure-ready, per cell)
```

This pipeline has three phases for each experiment, with strict boundaries, making the package replicable:

- Phase 1 primarily collects measurements and is the only phase that generates codes, runs algorithms, or consumes meaningful compute time. Depending on the collector, it stores either per-instance rows or summaries of a seeded batch, but never writes into `paper/results/`.
- Phase 2 performs deterministic extraction, including the main selection and aggregation steps (winner choice, backend choice, eligibility rules, and normalization). It never generates inputs or runs a benchmark algorithm.
- Phase 3 primarily draws the figures from `paper/results/`. Visualizers may apply presentation-specific grouping, range restrictions, or annotations, but never read from `paper/data/collected/`.

Phase 1 normally runs on a benchmark server; phases 2 and 3 run locally. The transfer between machines is exactly the contents of `paper/data/collected/`.

### Phase 1 — Data Collection

Long-running. Run under `tmux` or an equivalent. Almost every collector appends incrementally and resumes by skipping keys already present. Collectors based on shared batch statistics instead append one summary row per completed batch; re-running such a batch may repeat its computation, while extraction keeps its latest row. Delete the relevant output file or files only to deliberately restart a collection from scratch.
Collected data from `collect_algorithm.py` is not figure-specific, but used by multiple aggregators in the next steps.

```bash
python3 -m paper.benchmarks.collect_*
```

Writes CSV data into `paper/data/collected/`.

#### Optional dependency for `pm_stb_aut`

The automorphism-group method requires a GAP executable with the Guava package.
The exact GAP and Guava versions used for the original measurements cannot be
recovered from the collected outputs; this missing provenance is stated here
rather than replaced with an inferred version.
Install GAP and Guava, place any Guava dependencies under the repository's
`.gap/` directory as described in the top-level README, and set
`GAP_EXECUTABLE` if GAP is not on `PATH`. Without it,
`collect_algorithm` skips `pm_stb_aut`; A5 extraction warns and produces the
figure from the remaining methods, with the exclusion stated on the figure.

### Phase 2 — Information Extraction

Fast, deterministic, safe to re-run at any time.

```bash
python3 -m paper.experiments.extract_a<N>
```

Reads CSVs from `paper/data/collected/` and writes CSV data into `paper/results/a<N>/`.

### Phase 3 — Information Visualization

Fast, deterministic, safe to re-run at any time.

```bash
python3 -m paper.visualizations.visualize_a<N>
```

Reads CSVs from `paper/results/a<N>/` and writes one
`paper/results/a<N>/a<N>.png` or multiple PNGs, depending on the experiment.

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

Negative stabilizer instances use A1's short Clifford perturbation followed by
exact certification. Negative CSS instances instead use
`css_codes_independent_candidate`: two independently sampled CSS codes with a
pinned X-check rank `rx`, followed by SAT or matroid certification. When
`n > 28` and `r > 9`, the suite falls back to `css_codes_cascaded`; this changes
the negative-family composition across cells and, as its generator warns, is
not a single stable construction over the entire grid.


### A5 — Best-Performing Methods

|  |  |
|---|---|
| **Question** | Which exact algorithm performs best for each parameter setting? |
| **Algorithms measured** | All algorithms discussed in the paper for `pm_stb`, `pm_css`, and `lc_stb` on their corresponding code families |
| **Method** | 10 positive and 10 negative randomized instances per parameter setting; eligible algorithm with the lowest mean runtime selected per setting |

For each parameter setting, the lowest-mean method is selected only when both
positive and negative batches are present, all requested calls succeed, and
there are no timeouts, memory failures, wrong results, execution errors, or
generation errors. If no method completes but one or more methods fail only by
timeout, the lowest censored mean is used as a timeout-fallback winner. It is
rendered with the selected method's normal color and remains identified by the
`selection` column in `by_cell.csv`. A censored fallback ordering can be
uncertain, but cases where multiple timeout-only methods actually compete are
too few in the reported data to warrant a separate marker in the figure.
Methods excluded because data is missing or calls errored are named on the
figure.

Negative instances use the same constructions and large-parameter CSS fallback
described under A4: short Clifford perturbations for stabilizer codes,
`css_codes_independent_candidate` with pinned `rx` for CSS codes, and
`css_codes_cascaded` beyond the exact-certifier region.

### A6 — SAT on CSS-Code Permutation Equivalence

|  |  |
|---|---|
| **Question** | How does SAT perform with the tableau and check-matrix encodings on CSS codes, compared with its performance on general stabilizer codes? |
| **Algorithms measured** | `pm_css_sat` and `pm_stb_sat` on CSS codes; `pm_stb_sat` on general stabilizer codes |
| **Method** | 10 positive and 10 negative randomized instances per parameter setting; comparison of mean runtimes |

The result value for a parameter setting is the mean runtime of the instances.

Negative general-stabilizer instances use A1's short Clifford perturbation.
Negative CSS instances use `css_codes_independent_candidate` with pinned `rx`
and exact SAT/matroid certification. Beyond the exact-certifier region they use
`css_codes_cascaded`, so the negative-family composition changes across cells
as described under A4.

### A7 — SAT Encodings on CSS-Code Permutation Equivalence

|  |  |
|---|---|
| **Question** | Why does SAT perform so poorly on CSS permutation equivalence, even with the check-matrix encoding? |
| **Algorithms measured** | `pm_stb_sat` on unrestricted and block-structured general stabilizer codes; `pm_css_sat` on balanced CSS codes |
| **Method** | 10 positive randomized instances per parameter setting; no direct runtime measurements |

Two experiments are measured, first the number of solver decisions required to solve a code size and the number of decisions required to reject deliberately wrong qubit mappings. Both on instances with different amount of (in)dependent (un)coupled row-transformations.
The second compares clean/separated and fully row-mixed presentations of the same CSS groups using the `pm_stb_sat` encoding.

### A8 — Hybrid Component Attribution

|  |  |
|---|---|
| **Question** | What runtimes do the hybrids achieve, and which component actually decides each input? |
| **Algorithms measured** | Paper hybrids, rather than the thesis hybrids whose MQT-QECC integration follows a maintainability-oriented design strategy |
| **Method** | 10 constructive positive controls and 10 A1-style certified negatives per named code; runtime and decision-stage outcomes are reported separately |

The fixed population is `bell`, `3q_rep`, `5q_prf`, `steane`, `shor`,
`carbon`, `hamming_15`, `15q_optimal`, `tetrahedral`, `golay`,
`rot_surf_d5`, `hamming_31`, `coco_488`, `coco_666`, `bb_72`, `bb_90`,
`bb_108`, and `bb_144`. Gottesman and Bring are not part of A8. PM-STB and
LC-STB run on all 18 codes. PM-CSS runs on the 16 CSS codes; the two non-CSS
codes (`5q_prf` and `15q_optimal`) remain explicit `not_applicable` rows.

Positive controls are known equivalent by their recorded construction and do
not enter a certification backend. For
PM-STB/PM-CSS they use one uniformly sampled physical-qubit permutation plus a
fixed number of independently recorded invertible row additions (separate X/Z
row operations for CSS), and both hybrids consume the identical CSS pair. LC
controls use an independently uniform local Clifford from
`I,H,S,HS,SH,HSH` on every coordinate plus recorded row additions.

The negative stratum follows the A1 invariant-rejection benchmark. PM-STB and
LC-STB proposals apply two independently sampled Clifford gates from
`H,S,Sdg,X,Y,Z,CX,CZ,SWAP` to the named code. PM-CSS proposals apply two CNOTs,
each using a uniform ordered pair of distinct coordinates, and preserve CSS
orthogonality and both check ranks. Proposals are generated from a deterministic
attempt-seed sequence. SAT certifies PM-STB and LC-STB. PM-CSS uses SAT when
`n-k <= 9`, matroid isomorphism through `n <= 28`, and the same scalable
rank/weight invariant as A1 beyond that region. Large PM-CSS still uses exactly
the fixed-depth two-CNOT perturbation of the named code; it never falls back to
an independently generated partner. Only the first exact-certified-inequivalent
small/medium proposal becomes a timed negative. For a large proposal, an
invariant mismatch is a sound inequivalence witness, while an invariant match
is retained as unresolved and is not replaced.

This is not an unbiased negative sample. Small/medium negatives use rejection
sampling conditioned on leaving the relevant equivalence orbit. Large CSS uses
a fixed proposal population with certificate-dependent missingness: matching
invariants remain unresolved. All attempted proposal files and
certification-attempt records are retained. A certification timeout, memory
limit, solver `unknown`, or error blocks that negative case; it does not become
a negative and does not trigger selection of an easier attempt.
Certification and timed execution use separate supervised workers. They do not
share solver objects, although earlier work can warm OS file/page caches.
Equivalent exact-certified proposals remain visible in the exported
`num_certified_equivalent` proposal count and as `e#` plot annotations.

Named-code text files do not provide verified distance metadata. A8 persists
distance as unavailable and its paper PM hybrids skip the distance comparison
when either value is unavailable. It never copies the named code's placeholder
distance onto a perturbation and then treats it as measured truth.

The plotted runtime is `restricted_mean_seconds`: the mean observed execution
time with timed-out calls capped by their execution budget. It is not a
success-only unconditional runtime estimate. `mean_success_seconds` is also
exported, explicitly named, while unresolved labels, generation/certification
failures, execution timeouts/memory/errors/wrong answers, pending work, and
coverage remain separate columns and annotations. Hatched cells have at least
one unresolved or failed case.

#### A8 preflight, production, and resume

Use the repository virtual environment; bare `python3` may not contain the
dependencies. Preflight defaults to one seed and four-second stage budgets,
but an explicit isolated path is recommended:

```bash
tmux new-session -d -s paper-hybrids-preflight -c "$PWD" "mkdir -p '$PWD/paper/data/preflight' '$PWD/paper/data/logs' '$PWD/paper/data/.matplotlib' && exec env MPLCONFIGDIR='$PWD/paper/data/.matplotlib' .venv/bin/python -u -m paper.benchmarks.collect_a8 --preflight --output-directory '$PWD/paper/data/preflight/a8-final' >>'$PWD/paper/data/logs/hybrids_a8_preflight.log' 2>&1"
```

Production defaults are 10 deterministic seeds, depth two, a 300 s/4 GiB
generation budget, 900 s/13 GiB certification budget, and 5400 s/13 GiB hybrid
budget. Use a fresh explicit output directory. Re-running the identical command
resumes compatible completed stages. Any configuration, dependency, Python,
or relevant-source hash mismatch is refused instead of mixed.

The no-argument production command writes the campaign to the explicit default
`paper/data/collected/hybrids_a8_v4` and appends a persistent log:

```bash
tmux new-session -d -s paper-hybrids -c "$PWD" "mkdir -p '$PWD/paper/data/collected' '$PWD/paper/data/logs' '$PWD/paper/data/.matplotlib' && exec env MPLCONFIGDIR='$PWD/paper/data/.matplotlib' .venv/bin/python -u -m paper.benchmarks.collect_a8 >>'$PWD/paper/data/logs/hybrids_a8_v4.log' 2>&1"
```

Failures are durable terminal outcomes by default. Retry selected failed
stages explicitly, preserving prior attempts in each stage record and in
`events.jsonl`:

```bash
.venv/bin/python -u -m paper.benchmarks.collect_a8 \
  --output-directory paper/data/collected/hybrids-a8-paper-v4 \
  --retry generation certification execution
```

Only one writer can hold an output directory's lock. JSON input/stage writes
and CSV materialization use atomic replacement. Interruptions retain finished
stage records. Three identical
execution/generation/certification errors stop a run by default, and any wrong
answer against a certified label stops immediately. Short-budget timeouts are
expected preflight outcomes, not automatically defects.

Extract and render any campaign with explicit paths:

```bash
.venv/bin/python -m paper.experiments.extract_a8 \
  --input-directory paper/data/collected/hybrids-a8-paper-v4 \
  --output-file paper/results/a8/by_cell.csv
.venv/bin/python -m paper.visualizations.visualize_a8 \
  --input-file paper/results/a8/by_cell.csv \
  --output-file paper/results/a8/a8.png
```

Component tags are `CI` (cheap invariants), `EI` (expensive invariants), `S` (signatures), and the decision procedures `BF` (brute force), `MI` (matroid isomorphism), `GI` (graph isomorphism), `SAT`, and `LSE`.

---

## Design Rules

Collector-specific constants, input populations, certification choices, and output paths are kept close to each collection entry point. Most collectors also expose their resume keys and CSV persistence directly; shared generators, supervision/statistics utilities, and the diagnostic paper hybrids are reused where appropriate.

---

## AI Usage Transparency

For components shared with the main project, AI usage is disclosed in the top-level README. Within this replication package, AI tools assisted with implementing collectors, extractors, and visualizers, but not with their conceptual design. The research ideas, experimental methodology, data flow, visualization design, supervision strategy, and documentation were developed by the human author. All AI-assisted code was reviewed and validated by the author.
