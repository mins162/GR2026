#!/usr/bin/env bash
# A/B matrix: one binary, runtime toggles only, every contribution isolated.
#
# Two attribution views come out of the same set of runs:
#   - leave-one-out (`loo-*`) : opt minus one contribution -> that contribution's
#     marginal value at the final configuration (report table)
#   - ladder (`l1..l5`)       : base plus one contribution at a time, cumulative
#     (stacked-bar figure).  The rungs are ordered so that a knob is only turned
#     on after the knob it depends on: flute-overlap needs GPU-FLUTE.
# `l5-bgen` is bit-identical to a hypothetical `loo-tc`, so it serves both views
# and is only run once.
#
# Usage (tcsh):
#   env BENCH=/path/to/benchmarks ARCH=sm_86 ./tools/ab_matrix.sh
# Knobs: PLAN, OUT, ARCH, BENCH, KEEP_OUT, SKIP_BUILD, SKIP_EVAL.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${BENCH:-$ROOT/benchmarks}"
ARCH="${ARCH:-sm_86}"
OUT="${OUT:-$ROOT/ab_results_$(date +%m%d_%H%M%S)}"
KEEP_OUT="${KEEP_OUT:-base opt}"   # designs' .out kept for the congestion-heatmap pair
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# design:configset:repeats
PLAN="${PLAN:-mempool_group:full:3 bsg_chip:loo:2 mempool_tile_rank:anchor:2 nvdla:anchor:2}"

# nvcc lives in a home-installed toolkit on this server (docs/my-setup.md).
command -v nvcc >/dev/null 2>&1 || export PATH="$HOME/cuda-12.6/bin:$PATH"

# Knob order: gpu-flute, flute-overlap, gpu-batch-gen, incremental-vcost,
#             incremental-presum, incremental-commit, tree-center
cfg_bits() {
    case "$1" in
    base)     echo "0 0 0 0 0 0 0" ;;  # anchor: every contribution off
    opt)      echo "1 1 1 1 1 1 1" ;;  # anchor: shipping default
    loo-vcp)  echo "1 1 1 0 0 1 1" ;;  # -incremental vcost/presum
    loo-ovl)  echo "1 0 1 1 1 1 1" ;;  # -CPU/GPU overlap (GPU-FLUTE still on)
    loo-bgen) echo "1 1 0 1 1 1 1" ;;  # -GPU batch generation
    loo-flt)  echo "0 0 1 1 1 1 1" ;;  # -GPU-FLUTE entirely (algorithm + overlap)
    loo-cmt)  echo "1 1 1 1 1 0 1" ;;  # -incremental wire-demand commit
    l1-vcp)   echo "0 0 0 1 1 0 0" ;;
    l2-cmt)   echo "0 0 0 1 1 1 0" ;;
    l3-flt)   echo "1 0 0 1 1 1 0" ;;
    l4-ovl)   echo "1 1 0 1 1 1 0" ;;
    l5-bgen)  echo "1 1 1 1 1 1 0" ;;  # ladder rung 5 == leave-one-out of tree-center
    *) return 1 ;;
    esac
}

configset() {
    case "$1" in
    full)   echo "base opt loo-vcp loo-ovl loo-bgen loo-flt loo-cmt l1-vcp l2-cmt l3-flt l4-ovl l5-bgen" ;;
    loo)    echo "base opt loo-vcp loo-ovl loo-bgen loo-flt loo-cmt l5-bgen" ;;
    anchor) echo "base opt" ;;
    *) return 1 ;;
    esac
}

onoff() { [ "$1" = 1 ] && echo on || echo off; }

cfg_env() {
    read -r F O B V P C T <<<"$(cfg_bits "$1")"
    # Every knob is set explicitly on every run: a leftover `setenv` in the
    # tcsh session must never be able to silently flip one.
    # INSTANTGR_TREE_CENTER takes 'cpu' or '0', not 1/0 like the rest.
    echo "INSTANTGR_GPU_FLUTE=$F INSTANTGR_FLUTE_OVERLAP=$O INSTANTGR_GPU_BATCH_GEN=$B" \
         "INSTANTGR_INCREMENTAL_VCOST=$V INSTANTGR_INCREMENTAL_PRESUM=$P" \
         "INSTANTGR_INCREMENTAL_COMMIT=$C INSTANTGR_TREE_CENTER=$([ "$T" = 1 ] && echo cpu || echo 0)" \
         "INSTANTGR_GPU_TREE_CENTER=0"
}

# The requested configuration has to show up in the run's own output, or the
# two columns of a table are the same run.  GPU batch generation is absent from
# the `config:` line, so it is checked on the Generation row instead.
verify_log() {
    local log="$1" cfg="$2" line w
    read -r F O B V P C T <<<"$(cfg_bits "$cfg")"
    line="$(grep -m1 '^config: ' "$log")" || { echo "no config line"; return 1; }
    for w in "gpu-flute=$(onoff "$F")" "flute-overlap=$(onoff "$O")" \
             "incremental-vcost=$(onoff "$V")" "incremental-presum=$(onoff "$P")" \
             "incremental-commit=$(onoff "$C")" \
             "tree-center=$([ "$T" = 1 ] && echo cpu || echo off)"; do
        case "$line" in *"$w"*) ;; *) echo "expected '$w' in: $line"; return 1 ;; esac
    done
    w="Generation ($([ "$B" = 1 ] && echo GPU || echo CPU))"
    grep -qF "$w" "$log" || { echo "expected '$w' in the batch table"; return 1; }
    grep -q '^total ' "$log" || { echo "no total row (run did not finish)"; return 1; }
    return 0
}

gpu_busy() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
        tr -d ' ' | grep -c . || true
}

mkdir -p "$OUT"
RUNS="$OUT/runs.tsv"
printf 'design\tconfig\trep\tstatus\tgpu_busy\tlog\teval\n' >"$RUNS"

{
    echo "date       : $(date -Iseconds)"
    echo "host       : $(hostname)"
    echo "git HEAD   : $(git -C "$ROOT" rev-parse HEAD)"
    echo "git dirty  : $(git -C "$ROOT" status --short | tr '\n' ';')"
    echo "arch       : $ARCH"
    echo "bench      : $BENCH"
    echo "plan       : $PLAN"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "--- nvcc ---"; nvcc --version 2>&1
    echo "--- nvidia-smi ---"; nvidia-smi 2>&1
} >"$OUT/ENV.txt"

if [ "$SKIP_BUILD" != 1 ]; then
    echo "== build =="
    nvcc "$ROOT/src/main.cpp" -o "$ROOT/run/InstantGR.opt" -std=c++17 -x cu -O3 -arch="$ARCH" || exit 1
    g++ -O3 -std=c++17 -o "$ROOT/run/evaluator" "$ROOT/run/evaluator.cpp" || exit 1
fi

for entry in $PLAN; do
    IFS=: read -r design set reps <<<"$entry"
    cap="$BENCH/$design.cap" net="$BENCH/$design.net"
    if [ ! -r "$cap" ] || [ ! -r "$net" ]; then
        echo "!! $design: benchmark missing, skipped"
        continue
    fi
    configs="$(configset "$set")" || { echo "!! unknown config set '$set'"; continue; }

    # Design-major, then rep-major: a design's runs stay contiguous (the host
    # stages drift on a shared machine) while the config order rotates inside
    # each repeat, so drift is spread over every config instead of landing on one.
    for rep in $(seq 1 "$reps"); do
        for cfg in $configs; do
            tag="$design.$cfg.r$rep"
            log="$OUT/$tag.log"
            outfile="$OUT/$tag.out"
            busy="$(gpu_busy)"
            [ "$busy" != 0 ] && echo "!! $tag: $busy other compute process(es) on the GPU"

            echo "== $tag =="
            ( cd "$ROOT/run" && env $(cfg_env "$cfg") \
                ./InstantGR.opt -cap "$cap" -net "$net" -out "$outfile" ) >"$log" 2>&1
            rc=$?

            status=ok
            if [ $rc -ne 0 ]; then
                status="fail(rc=$rc)"
            elif ! msg="$(verify_log "$log" "$cfg")"; then
                status="mismatch"
                echo "!! $tag: $msg" | tee -a "$OUT/MISMATCH.txt"
            fi

            evalfile=""
            if [ "$status" = ok ] && [ "$SKIP_EVAL" != 1 ] && [ "$rep" = 1 ]; then
                evalfile="$OUT/$tag.eval"
                "$ROOT/run/evaluator" "$cap" "$net" "$outfile" >"$evalfile" 2>&1
            fi

            # Route files are large; only the pair the congestion heatmap needs
            # is kept, and only from the repeat the evaluator ran on.
            keep=0
            [ "$rep" = 1 ] && for k in $KEEP_OUT; do [ "$k" = "$cfg" ] && keep=1; done
            [ "$keep" = 0 ] && rm -f "$outfile"

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$design" "$cfg" "$rep" "$status" "$busy" "$log" "$evalfile" >>"$RUNS"
            grep -m1 '^total ' "$log" 2>/dev/null | sed "s/^/    $tag  /"
        done
    done
done

echo
echo "runs : $RUNS"
[ -s "$OUT/MISMATCH.txt" ] && echo "!! mismatches recorded in $OUT/MISMATCH.txt"
echo "next : python3 $ROOT/tools/ab_summary.py $OUT"
