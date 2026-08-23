#pragma once
// Force-included (nvcc -include) into the paper baseline for profiling runs.
//
// baseline/src/main.cpp ends in quick_exit(0).  That skips the atexit handler
// CUPTI flushes its activity buffers from, so nsys captures a trace with no
// kernels in it -- which is why the "paper" config came back empty.  src/ got
// the same fix inline (#ifdef INSTANTGR_NVTX in main.cpp), but baseline/ is
// kept byte-identical to the paper, so redirect the call at build time instead.
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
