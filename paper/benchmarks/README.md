# Paper data collection

This directory contains only the server-side entry points that produce raw,
reusable measurements. They do not select winners, aggregate figure cells, or
write into `paper/results`; that work lives in
[`paper/experiments`](../experiments/README.md).

Run from the repository root:

```bash
# Complete positive/negative random-suite data, one file per algorithm.
python3 -m paper.benchmarks.collect_algorithm --algorithm pm_stb_sat

# Diagnostic hybrid runtimes and deciding/stuck components.
python3 -m paper.benchmarks.collect_hybrids

# The one additional input population needed by A6.
python3 -m paper.benchmarks.collect_css_sat_encoding

# Explicit preprocessing timings needed by A3.
python3 -m paper.benchmarks.collect_invariant_timings

# Raw decisions and raw signature-space values needed by A1/A2.
python3 -m paper.benchmarks.collect_invariant_rejections
python3 -m paper.benchmarks.collect_signature_space
```

`collect_hybrids.py` runs the diagnostic hybrids from
[`paper/hybrids`](../hybrids) on the named structured codes and records, per
cell, the mean runtime, which component decided each instance, and which stage
a timed-out instance was stuck in. Its editable `HYBRID_N_RANGES` table and
nearby constants control the block-length window, seed count, master seed,
timeout, memory limit, and verbosity.

`collect_algorithm.py` exposes only repeatable `--algorithm` selection. If it is
omitted, all configured paper algorithms run. Its editable
`ALGORITHM_N_RANGES` table and the nearby constants control each algorithm's
inclusive range, seed count, master seed, timeout, memory limit, and verbosity.
The other five scripts intentionally have no CLI; their editable run settings
are documented in their top-level help-style docstrings.

## Raw files

All transferable results are under `paper/data/collected/`:

```text
paper/data/collected/
├── algorithms/
│   └── <algorithm>.csv
├── hybrids/
│   ├── <hybrid>.csv
│   └── <hybrid>_instances.csv
├── pm_stb_sat_on_css.csv
├── invariant_timings.csv
├── invariant_rejections.csv
└── signature_space.csv
```

The algorithm, PM-STB-on-CSS, and invariant-timing files use the common
append-only statistical schema from `benchmarks/experiments/statistics.py`.
Those collectors persist after every completed parameter/label statistic. The
hybrid, rejection, and signature collectors persist each individual instance
result; the hybrid collector writes its cell summary once every requested seed
has a durable instance row.
Restarting the instance-level collectors skips keys already present in their
CSVs, so an interrupted server run continues without duplicating completed
work. Remove an output file only when intentionally starting that collection
from scratch.

Everything that is not a runnable collector lives under `utils/`: `config.py`
contains shared constants and CSV writing, `generation.py` contains direct
deterministic negative-pair generation and SAT certification, and
`invariants.py` contains the five invariant calls. There is no paper-specific
runner, cached-instance object model, or serialized intermediate format; the
collectors reuse `benchmarks/experiments/run.py`.

## Which collection feeds which experiment?

| Experiment | Required raw data |
|---|---|
| A1 rejection rates | `invariant_rejections.csv` |
| A2 signature space | `signature_space.csv` |
| A3 invariant cost | `invariant_timings.csv` plus relevant `algorithms/*.csv` |
| A4 representation cost | `pm_stb_graph_iso.csv`, `lc_stb_graph_iso.csv` |
| A5 winners | all complete `algorithms/*.csv` files |
| A6 SAT/CSS | `pm_stb_sat.csv`, `pm_css_sat.csv`, `pm_stb_sat_on_css.csv` |

The grid and populations follow [`plan.md`](plan.md). Runtime collections use a
5400-second limit and 13 GiB memory cap by default. Generic algorithms use ten
cases per polarity/cell; the explicit A3 invariant timings use five.
