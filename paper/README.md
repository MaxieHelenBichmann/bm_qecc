# `paper/` — replication package for the paper figures

Everything needed to reproduce the figures of the paper lives in this
directory: the server-side measurement scripts, the raw measurements they
produce, the deterministic aggregation step, and the plotting entry points.

Nothing here is a library. Every stage is a runnable `python3 -m` entry point,
run from the **repository root**, and every stage writes files that the next
stage reads. There is no hidden state between stages.

The authoritative statement of *what* each figure must show is
[`bm_qecc_benchmark_visualization_spec.md`](bm_qecc_benchmark_visualization_spec.md).
This README states *how to produce it*.

---

## 1. Figure index

Figures are referred to everywhere in this package as **A1 … A7**. The spec
uses a different lettering (`A`/`B`/`C`/`D`) and predates A7; the mapping is
fixed here and is the only place the two schemes are related.

**A1 … A6 are the prototype figures and are the current scope of work. A7 is
planned and deferred** — its collector exists and is complete, but its
extractor and visualizer are intentionally not written yet.

| ID | Spec section | Question the figure answers | Figure |
|---|---|---|---|
| **A1** | A1 | When do the hybrid preprocessors actually reject inequivalent inputs? | `results/a1/a1.png` |
| **A2** | A2 | How strongly do signatures refine the permutation search space? | `results/a2/a2.png` |
| **A3** | A3 | When is preprocessing cheap relative to the selected exact backend? | `results/a3/a3.png` |
| **A4** | B1 | Where do graph-based methods trade fast search for excessive representation cost? | `results/a4/a4.png` |
| **A5** | C1 | Which exact algorithm wins in each parameter regime? | `results/a5/a5.png` |
| **A6** | D1 | Is SAT's poor CSS behavior caused by the encoding or by the CSS inputs? | `results/a6/a6.png` |
| **A7** | — | Which hybrid component actually decides a case, and where do the hybrids get stuck? | `results/a7/a7.png` *(deferred)* |

Three problems are in scope throughout: `pm_stb` (permutation equivalence of
stabilizer codes), `pm_css` (permutation equivalence of CSS codes), and
`lc_stb` (local-Clifford equivalence of stabilizer codes). `lc_css` is out of
scope.

### Naming convention

Figure-specific files carry **only** the `aN` identifier:

```text
benchmarks/collect_aN.py        experiments/extract_aN.py
visualizations/visualize_aN.py  results/aN/
```

One collector deliberately breaks the pattern because it is **not**
figure-specific, and renaming it would misrepresent what it measures:

| File | Why it keeps a descriptive name |
|---|---|
| `benchmarks/collect_algorithm.py` | One shared random suite per algorithm. It is the *sole* input to A4 and A5 and a *partial* input to A3 and A6. No `aN` name is correct for it. |

Raw files under `data/collected/` also keep descriptive names, for the same
reason: a raw population is reusable across figures, whereas a `results/aN/`
directory belongs to exactly one figure.

> **One rename is still pending.** `benchmarks/collect_invariant_rejections.py`
> is A1's collector and should become `benchmarks/collect_a1.py`, but a
> long-running collection is currently executing from that module path. Rename
> it once that run finishes.

---

## 2. The three phases

```text
  PHASE 1  collect                 PHASE 2  extract              PHASE 3  visualize
  benchmarks/*.py                  experiments/extract_aN.py     visualizations/visualize_aN.py
  ── resource-supervised,          ── deterministic, local,      ── plotting only, no
     hours-to-days, server            seconds, pure CSV→CSV         selection logic
         │                                   │                             │
         ▼                                   ▼                             ▼
  data/collected/*.csv  ───────────►  results/aN/by_cell.csv  ─────────►  results/aN/aN.png
     (raw, per instance)                (figure-ready, per cell)
```

The phase boundaries are strict and are what make the package replicable:

- Phase 1 is the only phase that generates codes, runs algorithms, or consumes
  meaningful compute. It never aggregates and never writes into `results/`.
- Phase 2 is the only phase that selects and aggregates (winner choice, backend
  choice, eligibility rules, normalization). It never generates inputs and
  never runs a benchmark algorithm.
- Phase 3 is the only phase that draws. It reads `results/` exclusively and
  never touches `data/collected/`.

Phase 1 normally runs on a benchmark server; phases 2 and 3 run locally. The
transfer between machines is exactly the contents of `data/collected/`.

### Phase 1 — collect

Long-running. Run under `tmux` or an equivalent; each collector appends to its
CSV incrementally and **resumes** by skipping keys already present, so an
interrupted run continues without duplicating completed work. Delete an output
file only to deliberately restart that collection from scratch.

```bash
# A1 — raw invariant decisions on certified inequivalent pairs.
#      (module still named collect_invariant_rejections; see the pending rename above)
python3 -m paper.benchmarks.collect_invariant_rejections

# A2 — signature partition sizes on typical random codes.
python3 -m paper.benchmarks.collect_a2

# A3 — marginal invariant runtimes on prepared matrices.
python3 -m paper.benchmarks.collect_a3

# A6 — the one extra population: PM-STB SAT run on CSS inputs.
python3 -m paper.benchmarks.collect_a6

# Shared algorithm suite. Sole input to A4/A5, partial input to A3/A6.
# --algorithm accepts an exact name, a shell wildcard, or a regex, and repeats.
# Omit it to run every configured algorithm.
python3 -m paper.benchmarks.collect_algorithm --algorithm pm_stb_sat
python3 -m paper.benchmarks.collect_algorithm

# A7 — hybrid component traces on named structured codes (deferred figure).
python3 -m paper.benchmarks.collect_a7
```

`collect_algorithm.py` is the only collector with a CLI. The others are
configured by editing the constants at the top of the file — each documents its
own grid, seed schedule, timeout, memory limit, and verbosity in its module
docstring.

### Phase 2 — extract

Fast, deterministic, safe to re-run at any time.

```bash
python3 -m paper.experiments.extract_a1
python3 -m paper.experiments.extract_a2
python3 -m paper.experiments.extract_a3
python3 -m paper.experiments.extract_a4
python3 -m paper.experiments.extract_a5
python3 -m paper.experiments.extract_a6
```

### Phase 3 — visualize

```bash
python3 -m paper.visualizations.visualize_a1
python3 -m paper.visualizations.visualize_a2
python3 -m paper.visualizations.visualize_a4
python3 -m paper.visualizations.visualize_a5
python3 -m paper.visualizations.visualize_a6
```

Each writes exactly one PNG, `results/aN/aN.png`.

---

## 3. Data flow: from where to where

### Phase 1 → `data/collected/`

| Produced by | Raw file | Consumed by |
|---|---|---|
| `collect_invariant_rejections.py` *(→ `collect_a1.py`)* | `invariant_rejections.csv` | A1 |
| `collect_a2.py` | `signature_space.csv` | A2 |
| `collect_a3.py` | `invariant_timings.csv` | A3 |
| `collect_a6.py` | `pm_stb_sat_on_css.csv` | A6 |
| `collect_algorithm.py` | `algorithms/<algorithm>.csv` | A3, A4, A5, A6 |
| `collect_a7.py` | `hybrids/<hybrid>.csv`, `hybrids/<hybrid>_instances.csv` | A7 |

```text
data/collected/
├── algorithms/
│   └── <algorithm>.csv          e.g. pm_stb_sat.csv, lc_stb_graph_iso.csv
├── hybrids/
│   ├── <hybrid>.csv
│   └── <hybrid>_instances.csv
├── invariant_rejections.csv
├── signature_space.csv
├── invariant_timings.csv
└── pm_stb_sat_on_css.csv
```

The algorithm and `pm_stb_sat_on_css` files use the common aggregate-statistics
schema. The rejection, signature, and invariant-timing collectors persist every
individual seeded result; their extractors validate and aggregate later.

### Phase 2 → `results/`

| Extractor | Reads | Writes | Selection performed |
|---|---|---|---|
| `extract_a1` | `invariant_rejections.csv` | `results/a1/by_cell.csv`, `results/a1/overall.csv` | component and combined rejection counts and rates |
| `extract_a2` | `signature_space.csv` | `results/a2/by_cell.csv` | typical random-code aggregate per cell |
| `extract_a3` | `invariant_timings.csv` + `algorithms/*.csv` | `results/a3/by_cell.csv` | fastest valid backend per problem/cell, then invariant/backend ratio |
| `extract_a4` | `algorithms/pm_stb_graph_iso.csv`, `algorithms/lc_stb_graph_iso.csv` | `results/a4/by_cell.csv` | the two graph-isomorphism algorithms only |
| `extract_a5` | all complete `algorithms/*.csv` | `results/a5/by_method.csv`, `results/a5/by_cell.csv` | fastest completed prototype, with a timeout-only fallback |
| `extract_a6` | `algorithms/pm_stb_sat.csv`, `algorithms/pm_css_sat.csv`, `pm_stb_sat_on_css.csv` | `results/a6/by_cell.csv` | the three SAT populations, aggregated across input labels |

### Phase 3 → `results/aN/aN.png`

Each `visualize_aN` reads `results/aN/by_cell.csv` and writes
`results/aN/aN.png` beside it.

---

## 4. Current state of the package

A1 … A6 are the prototype figures currently being finished. A7 is deliberately
left until they are done.

| | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
|---|---|---|---|---|---|---|---|
| collector | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| raw data collected | 🟡 partial | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| extractor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ deferred |
| visualizer | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ deferred |

- **A1 raw data is incomplete.** `pm_stb` was collected up to `n = 28` in a
  separate run; `pm_css` and `lc_stb` are being collected now. `PROBLEMS` in the
  collector is temporarily narrowed to `("pm_css", "lc_stb")` and must be
  restored to `("pm_stb", "pm_css", "lc_stb")` afterwards.
- **A3 has no raw data yet.** `collect_a3.py` has not been run, so
  `invariant_timings.csv` does not exist and `results/a3/` is empty.
- **A3 has no visualizer.** `visualize_a3.py` still has to be written to the
  layout in §6.
- **A7 is deferred by choice.** `collect_a7.py` is complete and runnable, but
  `extract_a7.py`, `visualize_a7.py`, and the figure layout are intentionally
  postponed until A1 … A6 are finished.

Until an extractor has written real data, a visualizer may fall back to a
temporary `synthetic_by_cell.csv` placed beside it. Such output is visibly
stamped **SYNTHETIC PLACEHOLDER DATA** and is a layout fixture, never a
measurement; a real `by_cell.csv` always takes precedence.
`visualizations/placeholders/` holds hand-drawn sketches of the intended
figures and is documentation, not data.

---

## 5. Measurement definitions

The grid is the `thesis_prototypes` measurement grid: the integer `(n, k)`
triangle from `n = 3` to `n = 47`. Both `n` and `r = n - k` appear in raw and
processed data; heatmaps use `n` horizontally and `r` vertically.

Unless stated otherwise: **timeout 5400 s**, **memory limit 13 GiB**,
**10 seeded instances per cell**, results reported as mean and standard
deviation. Completed runtimes are measured *inside* the supervised worker
around the call itself, excluding process startup, result transfer, and
resource monitoring. A timed-out call has no in-worker measurement and
therefore records the capped parent-observed wall time. `success`, `timeout`,
`memory_limited`, and `error` are recorded separately and never conflated.

**A1 — rejection rates.** Invariants measured: linear dependency and
punctured-hull/Sendrier signatures for `pm_stb` and `pm_css`; the degree-2
low-degree local invariant for `lc_stb`. 10 certified-negative instances per
cell; no runtime measurement. The result value is how many of the 10 were
rejected, per invariant and combined per equivalence notion.

Negatives must not be selected using the invariant whose rejection rate is
being measured, or the estimate is circular. Therefore:

- `pm_stb` / `lc_stb`: apply a short random Clifford circuit
  (`STABILIZER_CLIFFORD_GATE_STEPS`, currently 2) to one source code, then keep
  the candidate only if the corresponding exact SAT backend proves
  inequivalence. This yields structurally related negatives selected by an
  exact backend rather than by a measured invariant.
- `pm_css`: two independent draws sharing one X-check rank. Because
  `r_x + r_z = n - k`, this fixes a shared Z-check rank too and prevents
  trivial rank-mismatch negatives. Certification uses SAT for `r ≤ 9` and
  matroid isomorphism for `r > 9, n ≤ 28`.
- `pm_css` with `r > 9, n > 28`: **no independent exact certifier is
  configured.** This region falls back to `css_codes_cascaded`, which emits a
  negative carrying its own permutation-invariant certificate. That certificate
  can correlate with a measured invariant, so results in this region describe
  the fallback population, not invariant-neutral negatives, and **must be
  labelled as such in the paper.**

**A2 — signature space.** No pairs, no equivalence labels, no runtime. For each
seed, generate one unconditioned random code and measure how its Sendrier
signature partitions the physical qubits. `pm_css` runs first, sampling the
X-check rank uniformly from `0 … n-k`; `pm_stb` runs second on the repository's
layered random-Clifford ensemble. A2 therefore describes typical codes in these
ensembles and is not biased toward partitions that happen to match a partner.

The collector stores the raw `q_pairs = Σ_i |s_i|² / n²`. The extractor removes
the unavoidable self-pair contribution and writes
`(q_pairs − 1/n) / (1 − 1/n)` as `mean_distinct_pair_fraction` /
`stddev_distinct_pair_fraction`. This is the fraction of *distinct ordered*
qubit pairs remaining in the same signature class, with a dimension-independent
range: `0` = every class a singleton (complete refinement), `1` = one undivided
class (no refinement).

**A3 — relative preprocessing cost.** **5** instances per cell, not 10. The
value is `T_invariant / T_backend`. Timings are *marginal*: case generation,
inequivalence certification, and row-basis normalization all complete before
the timed call, so their cost is absent from the raw `runtime_seconds`;
`mean_seconds` is computed only by the extractor. The numerator and the shared
backend denominator use the same deterministic case family — the invariant
collector's 5 seeds are a subset of the algorithm collector's 10. The
denominator is the fastest *fully successful* backend for that problem and
cell. A certification failure counts as a generation error and makes the
extractor drop the whole cell rather than substitute a censored mean.

**A4 / A5 / A6.** Ordinary paired positive and negative random suites at the
standard limits, taken from `collect_algorithm.py`. A4 uses
`pm_stb_graph_iso` and `lc_stb_graph_iso`. A5 uses every prototype for
`pm_stb`, `pm_css`, and `lc_stb` (automorphism variants excluded). A6 uses
`pm_stb_sat` on stabilizer inputs, `pm_css_sat` on CSS inputs, and
`pm_stb_sat` on CSS inputs.

**A7 — hybrid component attribution.** Unlike every other figure, A7 does not
run on the random `(n, k)` grid: it measures all three diagnostic hybrids from
[`hybrids/`](hybrids) on every compatible **named structured code**, at the
standard limits and 10 instances per cell. The hybrids return a
`(decision, component)` pair and print a tag on entering each stage, which
yields the two things the plain algorithm collector cannot express: which
component actually decided a case, and, for a case that never finishes, which
component it was stuck in. Component tags are `CI` (cheap invariants), `EI`
(expensive invariants), `S` (signatures), and the decision procedures `BF`
(brute force), `MI` (matroid isomorphism), `GI` (graph isomorphism), `SAT`, and
`LSE`. A `TracedHybrid` wrapper redirects stdout line-buffered into a log file
*inside* the supervised process, so a trace survives the kill that ends a
timed-out call. `mean_seconds` averages only completed instances;
`mean_seconds_capped` follows the `statistics.py` convention of adding
timed-out instances at their capped runtime.

**Eligibility.** A3 disqualifies an algorithm in a cell if either polarity is
missing or any run was unexpected, timed out, memory-limited, errored, or
failed during generation. A5 first ranks methods under those same conditions;
if none completed, it permits paired methods whose *only* failure was a
timeout, while memory-limited, erroneous, unexpected, and generation-failed
methods stay excluded. Consequently the backend used by A3 and the winner
chosen by A5 can both vary with `(problem, n, k)`.

---

## 6. Figure layouts

All figures show the complete integer `(n, r)` triangle from `n = 3` to `47` as
an edge-to-edge square grid, so panels are directly comparable. A near-white
square means no usable measured value. A blue cross marks a memory-limit hit; a
purple star marks an error or a wrong result. Timeouts remain in the CSV but
have no figure glyph — an empty cell and a timeout cell must not be confused.
Only A1 subdivides cells, because its subdivisions represent distinct
invariants.

- **A1** — two panels: permutation equivalence left, LC equivalence right. Each
  invariant gets its own color family, deepening with the number of instances
  it rejects. A cell carries the colors of every invariant rejecting at least
  one instance there.
- **A2** — two panels: stabilizer left, CSS right. Shared linear `0…1` color
  scale for the normalized distinct-pair fraction, so a given color means the
  same thing at every `n`.
- **A3** — three panels: linear dependency, signatures, local invariant. The
  value is centered on 1: white at 1, blue below, red above. In the linear
  dependency and signature panels each cell is split — left half stabilizer,
  right half CSS.
- **A4** — two panels: permutation equivalence left, LC equivalence right. Mean
  runtime as the heat value, on the shared logarithmic runtime scale.
- **A5** — one winner map: each cell colored by the algorithm with the best
  runtime there.
- **A6** — three panels: `pm_stb_sat` on stabilizer codes, `pm_css_sat` on CSS
  codes, `pm_stb_sat` on CSS codes. Mean runtime as the heat value, shared
  logarithmic scale.
- **A7** — layout not yet decided. It is not an `(n, r)` heatmap: its rows are
  named structured codes, not grid cells, and its values are component
  distributions rather than a single scalar.

Runtime figures share one logarithmic color scale across panels.

---

## 7. Design rules

These are the constraints the code is written to; keep them when extending it.

- **Do not modify or overwrite existing raw measurements.** Add new files.
- **Every collector is self-contained.** Its constants, input generation,
  certification choices, invariant calls, resume keys, and CSV persistence are
  all visible in that one file. Some small helpers are intentionally duplicated
  between collectors so that a server-side run does not depend on any
  paper-specific helper package. Collectors reuse only the repository's generic
  generators, supervised runner, and statistics machinery.
- **Paired inputs.** Every method in one comparison receives the exact same
  serialized code pair for a given `instance_id`. Generate an input once and
  reuse it; never regenerate an allegedly matching input per algorithm.
- **Keep per-instance rows.** Rejection fractions, paired ratios, winner
  selection, and uncertainty all need them. Cell-level means alone are not
  sufficient.
- **Never plot an unverified result.** Algorithm outputs are checked against the
  known instance label; a benchmark error must not appear as a valid
  measurement.
- **Never interpret a fraction without its denominator.** Valid sample counts
  belong in the processed CSVs.
