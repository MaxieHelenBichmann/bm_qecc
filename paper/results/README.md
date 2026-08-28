# Paper experiment results

This directory contains figure-ready CSVs produced by `paper/experiments/` and
the PNGs produced by `paper/visualizations/`. Raw server measurements do not
belong here; they live under `paper/data/collected/`.

| Directory | Derived data |
|---|---|
| `invariant_rejection/` | A1 per-cell component/combined counts and overall rates |
| `signature_space/` | A2 positive+negative signature-space aggregate |
| `invariant_cost/` | A3 invariant timing relative to the fastest valid backend per cell |
| `representation_cost/` | A4 two graph-isomorphism algorithm aggregates |
| `winners/` | A5 all eligible method summaries and the winner per cell |
| `sat_css/` | A6 three SAT populations, aggregated across input labels |

Run the six extractors documented in `paper/experiments/README.md` after copying
the collected CSVs from the benchmark server. The four currently implemented
plotters are documented in `paper/visualizations/README.md`.

`synthetic_by_cell.csv` files and PNGs stamped `SYNTHETIC PLACEHOLDER DATA` are
temporary layout fixtures, not measurements.
