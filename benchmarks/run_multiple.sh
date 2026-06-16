#!/usr/bin/env bash

set -u

problem_type="pm_stb"
random=true

max_avail_mem=26
max_jobs=2
memory_limit_gib=$((max_avail_mem / max_jobs))
status=0
pids=()

nmin_for() {
    case "$1" in
        pm_css_sat) echo "" ;;
        pm_css_matroid) echo "" ;;
        pm_css_graph_iso) echo "" ;;
        pm_css_classical) echo "" ;;
        pm_css_bruteforce) echo "" ;;
        pm_stb_sat) echo "" ;;
        pm_stb_graph_iso) echo "" ;;
        pm_stb_classical) echo "" ;;
        pm_stb_bruteforce) echo "" ;;
        pm_stb_aut) echo "" ;;
        lc_equ_sat) echo "" ;;
        lc_equ_kls) echo "" ;;
        lc_equ_graph_state) echo "" ;;
        lc_equ_graph_iso) echo "" ;;
        lc_equ_bruteforce) echo "" ;;
        lc_css_sat) echo "" ;;
        lc_css_kls) echo "" ;;
        lc_css_cliff_orbit) echo "" ;;
        lc_css_lc_orbit) echo "" ;;
        lc_css_bruteforce) echo "" ;;
    esac
}

nmax_for() {
    case "$1" in
        pm_css_sat) echo "" ;;
        pm_css_matroid) echo "" ;;
        pm_css_graph_iso) echo "" ;;
        pm_css_classical) echo "" ;;
        pm_css_bruteforce) echo "" ;;
        pm_stb_sat) echo "" ;;
        pm_stb_graph_iso) echo "" ;;
        pm_stb_classical) echo "" ;;
        pm_stb_bruteforce) echo "" ;;
        pm_stb_aut) echo "" ;;
        lc_equ_sat) echo "" ;;
        lc_equ_kls) echo "" ;;
        lc_equ_graph_state) echo "" ;;
        lc_equ_graph_iso) echo "" ;;
        lc_equ_bruteforce) echo "" ;;
        lc_css_sat) echo "" ;;
        lc_css_kls) echo "" ;;
        lc_css_cliff_orbit) echo "" ;;
        lc_css_lc_orbit) echo "" ;;
        lc_css_bruteforce) echo "" ;;
    esac
}

mkdir -p results

case "$problem_type" in
    pm_css)
        algorithm_prefix="pm_css"
        algorithms=(sat matroid graph_iso classical bruteforce)
        ;;
    pm_stb)
        algorithm_prefix="pm_stb"
        algorithms=(sat graph_iso classical bruteforce aut)
        ;;
    lc_eq)
        algorithm_prefix="lc_equ"
        algorithms=(sat kls graph_state graph_iso bruteforce)
        ;;
    lc_css)
        algorithm_prefix="lc_css"
        algorithms=(sat kls cliff_orbit lc_orbit bruteforce)
        ;;
esac

if [[ "$random" == true ]]; then
    random_args=(--random)
    suffix="rdm"
else
    random_args=()
    suffix="known"
fi

run_algo() {
    local algo="$1"
    local algorithm_name="${algorithm_prefix}_${algo}"
    local output_base="results/${problem_type}_${algo}_${suffix}"
    local n_args=()
    local nmin
    local nmax

    nmin="$(nmin_for "$algorithm_name")"
    nmax="$(nmax_for "$algorithm_name")"

    if [[ -n "$nmin" ]]; then
        n_args+=(--nmin "$nmin")
    fi
    if [[ -n "$nmax" ]]; then
        n_args+=(--nmax "$nmax")
    fi

    echo "Starting ${algorithm_name}"
    python3 -m benchmarks.run \
        --stats \
        --algorithm "${algorithm_name}" \
        --timeout 5400 \
        --memory-limit "${memory_limit_gib}GiB" \
        --verbose \
        "${random_args[@]}" \
        "${n_args[@]}" \
        --output "${output_base}.csv" \
        >"${output_base}.log" \
        2>"${output_base}.err"
}

for algo in "${algorithms[@]}"; do
    while (( ${#pids[@]} >= max_jobs )); do
        wait "${pids[0]}" || status=$?
        pids=("${pids[@]:1}")
    done

    run_algo "$algo" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done

exit "$status"
