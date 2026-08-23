#pragma once
// Force-included (nvcc -include) into the paper baseline for profiling runs.
//
// baseline/src/main.cpp ends in quick_exit(0), which skips the atexit handler
// CUPTI flushes its activity buffers from.  src/ works around this inline
// (#ifdef INSTANTGR_NVTX in main.cpp); baseline/ is kept byte-identical to the
// paper, so redirect the call at build time instead.
//
// Confirmed on mempool_group: the same ~64s paper run traced without this shim
// reported 1305 compute_presum launches, with it 1466 -- which is exactly the
// 601 + 865 batches the run actually executes.  A 3s run came back empty
// altogether.  Truncation is silent, so the shim is not optional.
//
// <cstdlib> is pulled in first on purpose: defining the macro before the real
// declaration is parsed would mangle the declaration itself.
#include <cstdlib>
#include <cuda_runtime.h>

static inline void instantgr_profiling_exit(int code) {
    cudaDeviceSynchronize();
    exit(code);
}

#define quick_exit instantgr_profiling_exit
