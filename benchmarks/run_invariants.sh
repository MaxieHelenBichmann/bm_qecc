#!/usr/bin/env bash

set -euo pipefail

echo "Starting invariant benchmarks"
exec python3 -u -m benchmarks.run \
    --inv \
    --timeout 60 \
    --memory-limit "13GiB" \
    --verbose \
    --output "results/invariants.csv" \
    >"results/invariants.log" \
    2>"results/invariants.err"
