#pragma once
#include <cstdint>

// NVTX ranges for Nsight Systems.  Build with -DINSTANTGR_NVTX to enable; every
// macro below expands to nothing otherwise, so the normal build is unchanged.
//
// Why NVTX instead of the cudaEvent timers in Lshape_route_detour.hpp: reading
// an event pair requires cudaEventSynchronize(), which drains the queue and so
// changes the pipeline it is measuring.  nvtxRangePush/Pop only write a marker
// into the trace and never synchronize, and `nsys stats -r nvtx_gpu_proj_sum`
// projects each range onto the GPU work that was launched inside it.  The GPU
// numbers therefore come from the driver, not from host wall clock around
// asynchronous launches.
//
// The per-kernel table (`nsys stats -r cuda_gpu_kern_sum`) needs no
// instrumentation at all and is the primary evidence; these ranges exist to
// group kernels into pipeline stages and to count batches (the instance count
// of the "S1/batch" and "S2/batch" ranges is the batch count).

#if defined(INSTANTGR_NVTX)

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#else
#error "INSTANTGR_NVTX set but no NVTX header found; expected <nvtx3/nvToolsExt.h> from the CUDA toolkit"
#endif

namespace instantgr_nvtx {

// Distinct colors so one Stage-2 batch is readable at a glance in the timeline.
enum Color : uint32_t {
    STAGE       = 0xff90a4ae,
    RIPUP       = 0xffef5350,
    UPDATE_COST = 0xffffa726,
    PRESUM      = 0xffffee58,
    FLT         = 0xffab47bc,
    BOTTOM_UP   = 0xff66bb6a,
    TRACEBACK   = 0xff26c6da,
    COMMIT      = 0xff7e57c2,
};

inline void push(const char *name, uint32_t color) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

struct Range {
    Range(const char *name, uint32_t color) { push(name, color); }
    ~Range() { nvtxRangePop(); }
    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
};

}// namespace instantgr_nvtx

#define INSTANTGR_NVTX_JOIN_(a, b) a##b
#define INSTANTGR_NVTX_JOIN(a, b) INSTANTGR_NVTX_JOIN_(a, b)

#define NVTX_PUSH(name, color) instantgr_nvtx::push(name, instantgr_nvtx::color)
#define NVTX_POP() nvtxRangePop()
#define NVTX_RANGE(name, color) \
    instantgr_nvtx::Range INSTANTGR_NVTX_JOIN(nvtx_range_, __LINE__)(name, instantgr_nvtx::color)

#else

#define NVTX_PUSH(name, color) ((void) 0)
#define NVTX_POP() ((void) 0)
#define NVTX_RANGE(name, color) ((void) 0)

#endif
