# Paper experiments

These scripts turn shared raw collections into the exact CSVs needed by the six
paper experiments. They are local, deterministic CSV transformations: they do
not generate inputs or run benchmark algorithms.

```bash
python3 -m paper.experiments.a1_rejection_rates.extract
python3 -m paper.experiments.a2_signature_space.extract
python3 -m paper.experiments.a3_invariant_cost.extract
python3 -m paper.experiments.a4_representation_cost.extract
python3 -m paper.experiments.a5_winners.extract
python3 -m paper.experiments.a6_sat_css.extract
```

| Directory | Selection performed | Output |
|---|---|---|
| `a1_rejection_rates` | component/combined rejection counts and rates | `paper/results/invariant_rejection/{by_cell,overall}.csv` |
| `a2_signature_space` | positive+negative aggregate per cell | `paper/results/signature_space/by_cell.csv` |
| `a3_invariant_cost` | fastest valid backend per problem/cell and invariant/backend ratio | `paper/results/invariant_cost/by_cell.csv` |
| `a4_representation_cost` | only PM-STB and LC-STB graph-isomorphism algorithms | `paper/results/representation_cost/by_cell.csv` |
| `a5_winners` | fastest fully successful prototype per problem/cell | `paper/results/winners/{by_method,by_cell}.csv` |
| `a6_sat_css` | ordinary PM-STB SAT, ordinary PM-CSS SAT, and PM-STB SAT on CSS | `paper/results/sat_css/by_cell.csv` |

A2, A4, and A6 deliberately combine positive and negative populations into one
cell. A3 and A5 disqualify an algorithm in a cell if either polarity is absent
or any run was unexpected, timed out, memory-limited, errored, or failed during
generation. Therefore the backend used for A3 can vary with `(problem, n, k)`.

The plotting entry points in `paper/visualizations/` read these result CSVs;
they never read raw server collections directly.
