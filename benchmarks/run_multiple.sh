#!/usr/bin/env bash

for algo in sat matroid graph_iso classical bruteforce; do
    echo "Starting pm_css_${algo}"
    python3 -m benchmarks.run \
        --stats \
        --algorithm "pm_css_${algo}" \
        --random \
        --output "results/pm_css_${algo}_rnd.csv" \
        >"results/pm_css_${algo}_rnd.log" 2>&1 &
done

wait
