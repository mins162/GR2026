#!/usr/bin/env bash
# A/B: all optimizations WITHOUT tree-center (src/) vs paper baseline (baseline/).
# Usage: ./run_ab_no_treecenter.sh [design ...]
# Defaults: BENCH=/home/shkim/6_Internship/mskim/benchmarks/benchmarks, ARCH=sm_75 (TITAN RTX).
# Override from tcsh with: env BENCH=/other/path ARCH=sm_86 ./run_ab_no_treecenter.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BENCH="${BENCH:-/home/shkim/6_Internship/mskim/benchmarks/benchmarks}"
ARCH="${ARCH:-sm_75}"
DESIGNS=("$@")
[ ${#DESIGNS[@]} -eq 0 ] && DESIGNS=(mempool_tile_rank mempool_group mempool_cluster_ranking bsg_chip)

OUT="$ROOT/ab_results_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "== build =="
nvcc "$ROOT/src/main.cpp" -o "$ROOT/run/InstantGR" -std=c++17 -x cu -O3 -arch="$ARCH"
nvcc "$ROOT/baseline/src/main.cpp" -o "$ROOT/baseline/run/InstantGR" -std=c++17 -x cu -O3 -arch="$ARCH"
g++ -O3 -std=c++17 -o "$ROOT/run/evaluator" "$ROOT/run/evaluator.cpp"

for d in "${DESIGNS[@]}"; do
    cap="$BENCH/$d.cap" net="$BENCH/$d.net"

    echo "== $d : opt (no tree-center) =="
    (cd "$ROOT/run" && \
     INSTANTGR_TREE_CENTER=0 INSTANTGR_GPU_TREE_CENTER=0 \
     INSTANTGR_GPU_FLUTE=1 INSTANTGR_INCREMENTAL_VCOST=1 INSTANTGR_INCREMENTAL_PRESUM=1 \
     ./InstantGR -cap "$cap" -net "$net" -out "$OUT/$d.opt.out") \
     > "$OUT/$d.opt.log" 2>&1

    echo "== $d : baseline (paper) =="
    (cd "$ROOT/baseline/run" && \
     ./InstantGR -cap "$cap" -net "$net" -out "$OUT/$d.base.out") \
     > "$OUT/$d.base.log" 2>&1

    "$ROOT/run/evaluator" "$cap" "$net" "$OUT/$d.opt.out"  > "$OUT/$d.opt.eval"  2>&1
    "$ROOT/run/evaluator" "$cap" "$net" "$OUT/$d.base.out" > "$OUT/$d.base.eval" 2>&1
done

echo
echo "==================== SUMMARY ===================="
printf "%-26s %12s %12s %16s %16s\n" design opt_time base_time opt_cost base_cost
for d in "${DESIGNS[@]}"; do
    ot=$(awk '$1=="total"{print $2}' "$OUT/$d.opt.log"  | tail -1)
    bt=$(awk '$1=="total"{print $2}' "$OUT/$d.base.log" | tail -1)
    oc=$(awk '/^total cost/{print $3}' "$OUT/$d.opt.eval")
    bc=$(awk '/^total cost/{print $3}' "$OUT/$d.base.eval")
    printf "%-26s %12s %12s %16s %16s\n" "$d" "${ot:-?}" "${bt:-?}" "${oc:-?}" "${bc:-?}"
done
echo "logs/outputs: $OUT"
