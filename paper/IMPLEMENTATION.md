# Paper benchmark architecture

The paper pipeline has three deliberately separate stages:

```text
paper/benchmarks/     resource-supervised server collection
        ↓ raw CSVs
paper/data/collected/
        ↓ fixed experiment selection/aggregation
paper/experiments/    six figure-specific extractors
        ↓ figure-ready CSVs
paper/results/
        ↓ plotting only
paper/visualizations/
```

The public collectors, their global run settings, and their raw outputs are
documented in `paper/benchmarks/README.md`. All supporting Python code is under
`paper/benchmarks/utils/`: shared configuration, three direct generation
functions, and the invariant calls. Supervision comes from the repository's
normal benchmark runner. The six experiment dependencies and derived schemas
are documented in `paper/experiments/README.md`. The canonical measurement
definitions remain in `paper/benchmarks/plan.md`.

The generic algorithm collector delegates to the same random-suite and
statistics infrastructure as `benchmarks/thesis/thesis_prototypes.py`. This is
what allows A3, A4, A5, and A6 to reuse one measurement of an algorithm rather
than running overlapping suites independently.
