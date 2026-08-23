#pragma once
// Force-included (nvcc -include) into the paper baseline for profiling runs.
//
// baseline/src/main.cpp ends in quick_exit(0), which skips the atexit handler
// CUPTI flushes its activity buffers from.  src/ works around this inline
// (#ifdef INSTANTGR_NVTX in main.cpp); baseline/ is kept byte-identical to the
// paper, so redirect the call at build time instead.
//
// This is insurance, not a diagnosed fix.  A ~3s paper run did come back with
// an empty trace and a ~66s one did not, which fits a buffer that had not been
// streamed out yet at exit -- but that was never confirmed, and the long run
// worked without this shim.  Keep it because a truncated trace is silent.
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
