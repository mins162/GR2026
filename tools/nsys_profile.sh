#!/usr/bin/env bash
# Nsight Systems profiling for the "does vcost / presum actually cost anything?"
# question.
#
# The numbers in docs/optimizations.md come from cudaEvent pairs that the code
# itself records.  Reading such a pair needs cudaEventSynchronize(), which
# drains the queue and changes the pipeline it measures, and it only ever sees
# the kernels the instrumentation happens to bracket.  nsys measures from the
# driver side: every kernel, its real GPU duration, and its launch count, with
# no synchronization added to the program.
#
# Usage (the shebang runs bash, so calling it from tcsh works as-is):
#   tools/nsys_profile.sh                          # default design, all configs
#   tools/nsys_profile.sh -d mempool_group
#   tools/nsys_profile.sh -d mempool_tile_rank -c incr -c full
#   tools/nsys_profile.sh --cpu                    # add CPU sampling
#
# Env: BENCH (benchmark dir), ARCH (sm_XX).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${BENCH:-/home/shkim/6_Internship/mskim/benchmarks/benchmarks}"
ARCH="${ARCH:-sm_75}"

DESIGNS=()
CONFIGS=()
CPU_SAMPLING=0
GPU_METRICS=0
SKIP_BUILD=0

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    cat <<'USAGE'

Options:
  -d DESIGN     benchmark name, repeatable (default: mempool_tile_rank)
  -c CONFIG     repeatable; one of:
                  incr   src/ with incremental vcost+presum on   (current opt)
                  full   src/ with both off -> full-grid rebuild (paper behavior)
                  paper  baseline/ binary, untouched paper source
                (default: incr full paper)
  --cpu         also sample the CPU (answers "is the host the bottleneck?")
  --gpu-metrics sample SM/DRAM utilization counters (needs profiling permission)
  --no-build    reuse existing binaries
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        -d) DESIGNS+=("$2"); shift 2;;
        -c) CONFIGS+=("$2"); shift 2;;
        --cpu) CPU_SAMPLING=1; shift;;
        --gpu-metrics) GPU_METRICS=1; shift;;
        --no-build) SKIP_BUILD=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done

[ ${#DESIGNS[@]} -eq 0 ] && DESIGNS=(mempool_tile_rank)
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(incr full paper)

# $CUDA/bin/nsys is often a wrapper that refuses to run when its version does
# not match the toolkit ("Error: Nsight Systems X hasn't been installed with
# CUDA Toolkit Y").  The real binary sits under /opt/nvidia/nsight-systems.
# Probe candidates newest-first and keep the first one that answers --version.
resolve_nsys() {
    local c
    local -a candidates=()
    if [ -n "${NSYS:-}" ]; then candidates+=("$NSYS"); fi
    candidates+=(nsys)
    while IFS= read -r c; do
        candidates+=("$c")
    done < <(ls -d /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
                   /opt/nvidia/nsight-systems-cli/*/target-linux-x64/nsys \
                   /usr/local/cuda*/nsight-systems-*/target-linux-x64/nsys \
                   2>/dev/null | sort -Vr)
    for c in "${candidates[@]}"; do
        command -v "$c" > /dev/null 2>&1 || continue
        "$c" --version > /dev/null 2>&1 || continue
        command -v "$c"
        return 0
    done
    return 1
}

NSYS_BIN="$(resolve_nsys)" || {
    echo "no working nsys found." >&2
    echo "  what is on PATH:" >&2
    command -v nsys >&2 || echo "    (nothing)" >&2
    command -v nsys > /dev/null 2>&1 && nsys --version >&2 || true
    echo "  installed copies:" >&2
    ls -d /opt/nvidia/nsight-systems*/*/target-linux-x64/nsys \
          /usr/local/cuda*/nsight-systems-*/target-linux-x64/nsys 2>/dev/null >&2 \
        || echo "    (none found; Nsight Systems is not installed)" >&2
    echo "  point at one explicitly with:  env NSYS=/full/path/to/nsys $0 ..." >&2
    exit 1
}
echo "nsys: $NSYS_BIN"

OUT="$ROOT/nsys_results_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "output: $OUT"
"$NSYS_BIN" --version | tee "$OUT/nsys-version.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.csv" 2>/dev/null || true
{ lscpu; echo; nproc; } > "$OUT/cpu.txt" 2>/dev/null || true

# ---------------------------------------------------------------- build ------
# The NVTX build is a separate binary: -DINSTANTGR_NVTX is the only difference,
# and it must not become the binary that gets timed for the headline numbers.
NVTX_FLAG="-DINSTANTGR_NVTX"
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "== build (nvtx) =="
    if ! nvcc "$ROOT/src/main.cpp" -o "$ROOT/run/InstantGR.nvtx" \
              -std=c++17 -x cu -O3 -arch="$ARCH" $NVTX_FLAG 2> "$OUT/build.nvtx.log"; then
        echo "   NVTX build failed (see $OUT/build.nvtx.log); falling back to a plain build."
        echo "   Per-kernel timings still work; only the range grouping is lost."
        NVTX_FLAG=""
        nvcc "$ROOT/src/main.cpp" -o "$ROOT/run/InstantGR.nvtx" \
             -std=c++17 -x cu -O3 -arch="$ARCH"
    fi
    if printf '%s\n' "${CONFIGS[@]}" | grep -qx paper; then
        echo "== build (paper baseline) =="
        nvcc "$ROOT/baseline/src/main.cpp" -o "$ROOT/baseline/run/InstantGR" \
             -std=c++17 -x cu -O3 -arch="$ARCH"
    fi
fi

# --------------------------------------------------------------- profile -----
# --sample=none / --cpuctxsw=none keep the collector off the host critical path;
# the GPU trace is what the vcost/presum question needs.  --cpu turns the host
# side back on for the separate "is the CPU the bottleneck?" question.
TRACE="cuda,nvtx"
SAMPLE_ARGS=(--sample=none --cpuctxsw=none)
if [ "$CPU_SAMPLING" -eq 1 ]; then
    TRACE="cuda,nvtx,osrt"
    SAMPLE_ARGS=(--sample=process-tree --cpuctxsw=process-tree)
fi
METRIC_ARGS=()
if [ "$GPU_METRICS" -eq 1 ]; then METRIC_ARGS=(--gpu-metrics-devices=cuda-visible); fi

run_config() {
    local design="$1" config="$2"
    local cap="$BENCH/$design.cap" net="$BENCH/$design.net"
    local tag="$design-$config" workdir bin
    local -a envs=()

    case "$config" in
        incr)  workdir="$ROOT/run"; bin=./InstantGR.nvtx
               envs=(INSTANTGR_GPU_FLUTE=1 INSTANTGR_INCREMENTAL_VCOST=1 INSTANTGR_INCREMENTAL_PRESUM=1);;
        full)  workdir="$ROOT/run"; bin=./InstantGR.nvtx
               envs=(INSTANTGR_GPU_FLUTE=1 INSTANTGR_INCREMENTAL_VCOST=0 INSTANTGR_INCREMENTAL_PRESUM=0);;
        paper) workdir="$ROOT/baseline/run"; bin=./InstantGR; envs=();;
        *) echo "unknown config: $config" >&2; return 2;;
    esac
    for f in "$cap" "$net"; do
        [ -r "$f" ] || { echo "missing benchmark file: $f" >&2; return 1; }
    done

    echo "== profile $tag =="
    ( cd "$workdir" && \
      env ${envs[@]+"${envs[@]}"} "$NSYS_BIN" profile \
          --trace="$TRACE" \
          "${SAMPLE_ARGS[@]}" \
          ${METRIC_ARGS[@]+"${METRIC_ARGS[@]}"} \
          --cuda-memory-usage=false \
          --force-overwrite=true \
          --stats=false \
          --output="$OUT/$tag" \
          "$bin" -cap "$cap" -net "$net" -out "$OUT/$tag.out" ) \
      > "$OUT/$tag.log" 2>&1 || {
        echo "   run failed; tail of $OUT/$tag.log:" >&2
        tail -20 "$OUT/$tag.log" >&2
        return 1
      }
    grep -E '^(config|grid|total)' "$OUT/$tag.log" || true
}

# nsys renamed its reports around 2022 (gpukernsum -> cuda_gpu_kern_sum).  Try
# the current name first and fall back, so this works on whatever the server has.
stats_csv() {
    local rep="$1" tag="$2"; shift 2
    for name in "$@"; do
        if "$NSYS_BIN" stats --report "$name" --format csv \
                      --output "$OUT/$tag" "$rep" > /dev/null 2>&1; then
            echo "$name"
            return 0
        fi
    done
    return 1
}

for design in "${DESIGNS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        tag="$design-$config"
        run_config "$design" "$config" || continue
        rep="$OUT/$tag.nsys-rep"
        [ -f "$rep" ] || rep="$OUT/$tag.qdrep"
        echo "   stats: $tag"
        stats_csv "$rep" "$tag" cuda_gpu_kern_sum gpukernsum   > /dev/null || echo "   (no kernel-summary report)"
        stats_csv "$rep" "$tag" cuda_api_sum      cudaapisum   > /dev/null || true
        stats_csv "$rep" "$tag" cuda_gpu_mem_time_sum gpumemtimesum > /dev/null || true
        if [ -n "$NVTX_FLAG" ]; then
            stats_csv "$rep" "$tag" nvtx_gpu_proj_sum nvtxppsum > /dev/null || true
            stats_csv "$rep" "$tag" nvtx_sum nvtxsum            > /dev/null || true
        fi
        if [ "$CPU_SAMPLING" -eq 1 ]; then
            stats_csv "$rep" "$tag" osrt_sum osrtsum > /dev/null || true
        fi
    done
done

echo
python3 "$ROOT/tools/nsys_summarize.py" "$OUT" | tee "$OUT/SUMMARY.txt"
echo
echo "raw traces + csv: $OUT"
echo "open a timeline with:  nsys-ui $OUT/<design>-<config>.nsys-rep"
