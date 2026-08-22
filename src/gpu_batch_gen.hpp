#pragma once

// GPU-accelerated batch generation.
//
// The arbitration is the one from the InstantGR journal (TCAD Jan 2026,
// Section III-D): candidates stamp their marks into a batch map with atomicMin
// so the higher-priority net (the smaller index in the caller's ordering) wins
// a contested cell, the nets that still own all of their own points join the
// batch, and the losers retry.  The paper drives that arbitration one batch at
// a time, with every unplaced net competing for the batch being built.  That
// costs one full mark commit per net per batch, and mempool_group needs 602
// batches in stage 1 and 869 in stage 2, so a net that ends up late pays for
// hundreds of commits.  Measured on an A/B run it came out slower than the
// sequential CPU first-fit it replaces (stage 2: 12.9s against 3.5s).
//
// What makes the CPU version fast is that a *failed* attempt is cheap: it tests
// the net's points against a batch's occupancy map and stops at the first hit.
// This implementation keeps that structure and parallelizes it:
//
//   window     a ring of open batches keeps its occupancy bitmap live, so a net
//              can still join an older batch instead of only the newest one;
//   wavefront  a bounded slice of the priority order is in flight at a time,
//              which keeps the speculation ratio (nets committing per net
//              assigned) near one instead of near the batch count;
//   pick       each net scans the open batches from where it last stopped -- a
//              batch that blocked it stays blocked -- and takes the first one
//              its points are free in, so a failed attempt costs a few bitmap
//              reads and no commit;
//   commit     the paper's atomicMin arbitration, run once per net per round
//              over the batch it picked;
//   check      a net that owns all of its points joins that batch, stamps its
//              marks into the batch bitmap and leaves the wavefront; the others
//              retry next round, and fresh nets refill the wavefront.
//
// A net that fits in no open batch opens a new one.  When the ring is full the
// oldest batch is retired; nets never scan backwards, so a retired batch is one
// no net in flight could still join.
//
// The marks (horizontal / vertical RSMT segments plus the four-cell cross that
// mark_3x3() stamps around every point) and the conflict predicate (a net's own
// points against the batch map) are the ones the CPU path uses, so the batches
// stay interchangeable and nothing downstream changes.  Nets assigned in the
// same round can co-occupy a cell when the lower-priority one only crosses it
// with a segment, which is what the CPU path allows as well; the invariant both
// keep is that no net's points are covered by a higher-priority member of its
// batch.

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_batch {

constexpr int kEmpty = INT_MAX;   // no candidate owns the cell
constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 8192;

// Wavefront bounds.  It is retuned every round from the assignment rate, so
// these only have to be wide enough to bracket the batch sizes of both stages
// (stage 1 averages ~4k nets per batch, stage 2 ~230).  The floor keeps the
// tail of a run from running rounds too small to be worth their launch.
constexpr int kMinWavefront = 1024;
constexpr int kMaxWavefront = 1 << 18;
constexpr int kInitialWavefront = 1 << 14;

// Open batches to keep live.  More of them means fewer batches retired early,
// at one bitmap per batch; the ring is sized from this budget and the grid.
constexpr size_t kRingBudgetBytes = 1ull << 30;
constexpr int kMinRingSlots = 8;
constexpr int kMaxRingSlots = 4096;

enum : unsigned char {
  kActive = 0,    // still in the wavefront
  kAssigned = 1,  // joined a batch this round
};

// One net's grid footprint, in the caller's priority order.  Segment endpoints
// use the router's 2-D encoding pos = x * Y + y, so a horizontal segment walks
// pos by Y and a vertical one by 1.
struct HostMarks {
  std::vector<int> h_off, h_lo, h_hi;
  std::vector<int> v_off, v_lo, v_hi;
  std::vector<int> p_off, p_pos;
};

struct DeviceMarks {
  const int *h_off, *h_lo, *h_hi;
  const int *v_off, *v_lo, *v_hi;
  const int *p_off, *p_pos;
};

struct Result {
  std::vector<int> batch_of_net;  // priority index -> batch, -1 if unplaced
  int batch_count = 0;
  int rounds = 0;                 // wavefront rounds
  long long commits = 0;          // nets that committed, summed over rounds
  int ring_slots = 0;             // batches kept open at once
  int retired = 0;                // batches closed because the ring was full
  int wavefront = 0;              // wavefront size the run settled on
};

inline void check(cudaError_t status, const char *operation) {
  if(status != cudaSuccess) {
    throw std::runtime_error(std::string("GPU batch generation ") + operation +
                             ": " + cudaGetErrorString(status));
  }
}

inline int blocks_for(int items) {
  const int blocks = (items + kBlockSize - 1) / kBlockSize;
  return std::max(1, std::min(kMaxBlocks, blocks));
}

// Every cell the net occupies in a batch map.  This is mark_3x3() plus the two
// segment walks of the CPU path, kept in one place so commit, stamp and release
// cannot drift apart.  mark_3x3()'s y loop stops at y == _y, so the cross is
// (x, y-1), (x-1, y), (x, y) and (x+1, y) -- not a full 3x3 block.
template <class Fn>
__device__ inline void visit_marks(const DeviceMarks &marks, int net, int X, int Y, Fn fn) {
  for(int i = marks.h_off[net]; i < marks.h_off[net + 1]; i++)
    for(int pos = marks.h_lo[i]; pos <= marks.h_hi[i]; pos += Y) fn(pos);
  for(int i = marks.v_off[net]; i < marks.v_off[net + 1]; i++)
    for(int pos = marks.v_lo[i]; pos <= marks.v_hi[i]; pos++) fn(pos);
  for(int i = marks.p_off[net]; i < marks.p_off[net + 1]; i++) {
    const int pos = marks.p_pos[i];
    const int x = pos / Y, y = pos % Y;
    if(y > 0) fn(pos - 1);
    fn(pos);
    if(x > 0) fn(pos - Y);
    if(x + 1 < X) fn(pos + Y);
  }
}

__device__ inline bool bit_test(const unsigned *map, int pos) {
  return (map[pos >> 5] >> (pos & 31)) & 1u;
}

__global__ void fill_kernel(int *data, long long count, int value) {
  for(long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < count;
      i += (long long)blockDim.x * gridDim.x)
    data[i] = value;
}

// First fit over the open batches.  A batch that blocked this net can only have
// gained marks since, so the scan resumes where it stopped instead of starting
// over; that is what keeps a failed attempt down to a few bitmap reads.  The
// newest batch is always an empty one, so the scan always ends somewhere.
__global__ void pick_kernel(const int *active, int active_cnt, DeviceMarks marks,
                            const unsigned *slot_maps, long long words, int ring,
                            int slot_base, int live_cnt, const int *slot_size, int cap,
                            int *scan_from, int *pick) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    const int net = active[i];
    const int last = slot_base + live_cnt;
    int chosen = -1;
    int batch = scan_from[net];
    if(batch < slot_base) batch = slot_base;   // its earlier batches were closed
    for(; batch < last; batch++) {
      const int slot = batch % ring;
      if(slot_size[slot] >= cap) continue;
      const unsigned *map = slot_maps + slot * words;
      bool free_here = true;
      for(int k = marks.p_off[net]; k < marks.p_off[net + 1]; k++)
        if(bit_test(map, marks.p_pos[k])) { free_here = false; break; }
      if(free_here) { chosen = batch; break; }
    }
    scan_from[net] = chosen < 0 ? last : chosen;
    pick[i] = chosen;
  }
}

// The paper's commit step.  Nets that picked different batches share the owner
// map, so one can take a cell the other wanted; the loser simply retries next
// round, which is cheaper than keeping an owner map per open batch.
__global__ void commit_kernel(const int *active, int active_cnt, DeviceMarks marks,
                              int X, int Y, int *owner) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    const int net = active[i];
    visit_marks(marks, net, X, Y, [&] (int pos) { atomicMin(owner + pos, net); });
  }
}

// Check step.  Only the net's own points are tested, exactly as the CPU
// has_conflict() does: a segment may run over a lower-priority net's cells
// without that net losing its place in the batch.
__global__ void check_kernel(const int *active, int active_cnt, const int *pick,
                             DeviceMarks marks, const int *owner, int ring, int cap,
                             int *slot_size, int *batch_of_net, unsigned char *status) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    const int net = active[i];
    bool clean = true;
    for(int k = marks.p_off[net]; k < marks.p_off[net + 1]; k++)
      if(owner[marks.p_pos[k]] != net) { clean = false; break; }
    unsigned char state = kActive;
    if(clean && pick[i] >= 0) {
      // The size cap is a memory guard on the batch kernels downstream, and it
      // sits far above the batch sizes these designs produce.  Losing the race
      // for the last few places only sends a net to another batch.
      const int taken = atomicAdd(slot_size + pick[i] % ring, 1);
      if(taken < cap) {
        batch_of_net[net] = pick[i];
        state = kAssigned;
      }
    }
    status[i] = state;
  }
}

__global__ void stamp_kernel(const int *active, int active_cnt, const int *pick,
                             const unsigned char *status, DeviceMarks marks, int X, int Y,
                             unsigned *slot_maps, long long words, int ring) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    if(status[i] != kAssigned) continue;
    const int net = active[i];
    unsigned *map = slot_maps + (pick[i] % ring) * words;
    visit_marks(marks, net, X, Y, [&] (int pos) {
      atomicOr(map + (pos >> 5), 1u << (pos & 31));
    });
  }
}

// The owner map is scratch for one round: an assigned net's marks live in its
// batch bitmap from here on, so every net in the round drops its cells.  A cell
// holds this net's index only if this net won it, so the store cannot race.
__global__ void release_kernel(const int *active, int active_cnt, DeviceMarks marks,
                               int X, int Y, int *owner) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    const int net = active[i];
    visit_marks(marks, net, X, Y, [&] (int pos) {
      if(owner[pos] == net) owner[pos] = kEmpty;
    });
  }
}

template <class T>
struct DeviceArray {
  T *ptr = nullptr;
  size_t count = 0;

  DeviceArray() = default;
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator= (const DeviceArray &) = delete;
  ~DeviceArray() { cudaFree(ptr); }

  void alloc(size_t n, const char *what) {
    count = n;
    if(n == 0) return;
    check(cudaMalloc(&ptr, n * sizeof(T)), what);
  }
  void upload(const std::vector<T> &host, const char *what) {
    alloc(host.size(), what);
    if(host.empty()) return;
    check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what);
  }
};

// Returns false and leaves out untouched when the GPU cannot take the job (out
// of memory on a large grid, most likely); the caller then runs the CPU path.
inline bool generate(const HostMarks &marks, int net_cnt, int X, int Y, int max_batch_size,
                     Result &out) {
  if(net_cnt <= 0) return false;
  const long long cell_count = (long long)X * Y;
  const long long words = (cell_count + 31) / 32;
  try {
    DeviceArray<int> h_off, h_lo, h_hi, v_off, v_lo, v_hi, p_off, p_pos;
    h_off.upload(marks.h_off, "h_off upload");
    h_lo.upload(marks.h_lo, "h_lo upload");
    h_hi.upload(marks.h_hi, "h_hi upload");
    v_off.upload(marks.v_off, "v_off upload");
    v_lo.upload(marks.v_lo, "v_lo upload");
    v_hi.upload(marks.v_hi, "v_hi upload");
    p_off.upload(marks.p_off, "p_off upload");
    p_pos.upload(marks.p_pos, "p_pos upload");
    const DeviceMarks device_marks{h_off.ptr, h_lo.ptr, h_hi.ptr,
                                   v_off.ptr, v_lo.ptr, v_hi.ptr,
                                   p_off.ptr, p_pos.ptr};

    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory query");
    const size_t budget = std::min<size_t>(kRingBudgetBytes, free_bytes / 8);
    const int ring = (int)std::max<size_t>(
        kMinRingSlots, std::min<size_t>(kMaxRingSlots, budget / (words * sizeof(unsigned))));

    DeviceArray<unsigned> slot_maps;
    slot_maps.alloc((size_t)ring * words, "batch bitmap alloc");
    DeviceArray<int> slot_size;
    slot_size.alloc(ring, "batch size alloc");

    DeviceArray<int> owner;
    owner.alloc(cell_count, "owner map alloc");
    fill_kernel<<<kMaxBlocks, kBlockSize>>>(owner.ptr, cell_count, kEmpty);
    check(cudaGetLastError(), "owner map fill");

    DeviceArray<int> batch_of_net, scan_from, pending;
    batch_of_net.alloc(net_cnt, "batch id alloc");
    fill_kernel<<<blocks_for(net_cnt), kBlockSize>>>(batch_of_net.ptr, net_cnt, -1);
    scan_from.alloc(net_cnt, "scan cursor alloc");
    check(cudaMemset(scan_from.ptr, 0, (size_t)net_cnt * sizeof(int)), "scan cursor clear");

    // Unplaced nets in priority order.  The wavefront is the prefix starting at
    // head, so narrowing the wavefront hands the tail straight back to the next
    // round instead of leaving stale nets in flight.  The host keeps the list:
    // compacting it there costs one small copy each way, where a device-side
    // stream compaction would allocate scratch storage every round.
    std::vector<int> pending_host(net_cnt);
    for(int i = 0; i < net_cnt; i++) pending_host[i] = i;
    pending.upload(pending_host, "pending upload");

    const int wavefront_cap = std::min(net_cnt, kMaxWavefront);
    DeviceArray<int> pick;
    DeviceArray<unsigned char> status;
    pick.alloc(wavefront_cap, "pick alloc");
    status.alloc(wavefront_cap, "status alloc");
    std::vector<unsigned char> status_host(wavefront_cap);
    std::vector<int> kept_host(wavefront_cap);

    const int cap = std::max(1, std::min(net_cnt, max_batch_size));
    int wavefront = std::min(net_cnt, kInitialWavefront);
    int pending_cnt = net_cnt, head = 0, slot_base = 0, live_cnt = 0, stalled = 0;

    // The newest batch is kept empty so that pick_kernel always has somewhere to
    // put a net that fits nowhere else; that removes a device-to-host round trip
    // from the middle of every round.
    const auto open_batch = [&] {
      if(live_cnt == ring) {
        // The ring is full, so the oldest batch is closed to make room.  Nets
        // that arrive later can no longer land in it, which can cost a few
        // extra batches; the ring is sized to make that rare.
        slot_base++;
        live_cnt--;
        out.retired++;
      }
      const int slot = (slot_base + live_cnt) % ring;
      check(cudaMemset(slot_maps.ptr + (size_t)slot * words, 0, words * sizeof(unsigned)),
            "batch bitmap clear");
      check(cudaMemset(slot_size.ptr + slot, 0, sizeof(int)), "batch size clear");
      live_cnt++;
    };
    open_batch();

    int newest_size = 0;
    while(pending_cnt > 0) {
      const int active_cnt = std::min(wavefront, pending_cnt);
      int *const active = pending.ptr + head;

      pick_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
          active, active_cnt, device_marks, slot_maps.ptr, words, ring, slot_base, live_cnt,
          slot_size.ptr, cap, scan_from.ptr, pick.ptr);
      commit_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
          active, active_cnt, device_marks, X, Y, owner.ptr);
      check_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
          active, active_cnt, pick.ptr, device_marks, owner.ptr, ring, cap, slot_size.ptr,
          batch_of_net.ptr, status.ptr);
      stamp_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
          active, active_cnt, pick.ptr, status.ptr, device_marks, X, Y, slot_maps.ptr, words,
          ring);
      release_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
          active, active_cnt, device_marks, X, Y, owner.ptr);
      check(cudaGetLastError(), "commit-check round");

      check(cudaMemcpy(status_host.data(), status.ptr, active_cnt, cudaMemcpyDeviceToHost),
            "status download");
      check(cudaMemcpy(&newest_size, slot_size.ptr + (slot_base + live_cnt - 1) % ring,
                       sizeof(int), cudaMemcpyDeviceToHost), "batch size download");

      int kept = 0;
      for(int i = 0; i < active_cnt; i++)
        if(status_host[i] == kActive) kept_host[kept++] = pending_host[head + i];
      const int assigned = active_cnt - kept;
      // The highest-priority net of a round wins every cell it asks for, so a
      // round without an assignment means it lost the race for the last places
      // of a full batch.  That resolves once the next pick sees the batch full;
      // several in a row would mean the invariant broke, so bail to the CPU.
      stalled = assigned == 0 ? stalled + 1 : 0;
      if(stalled > 4) throw std::runtime_error("rounds stopped assigning nets");
      if(kept > 0) {
        std::copy(kept_host.begin(), kept_host.begin() + kept, pending_host.begin() + head + assigned);
        check(cudaMemcpy(pending.ptr + head + assigned, kept_host.data(), kept * sizeof(int),
                         cudaMemcpyHostToDevice), "pending upload"); 
      }
      head += assigned;
      pending_cnt -= assigned;
      out.commits += active_cnt;
      out.rounds++;
      if(newest_size > 0 && pending_cnt > 0) open_batch();

      // Retune the wavefront to the batch size this design produces.  The round
      // count does not depend on it -- how fast batches fill is set by the
      // conflicts, not by how many nets are in flight -- so a wavefront wider
      // than the design needs only buys speculation that loses.
      if(assigned * 8 < wavefront) wavefront = std::max(kMinWavefront, wavefront / 2);
      else if(assigned * 2 > wavefront) wavefront = std::min(wavefront_cap, wavefront * 2);
    }

    out.batch_of_net.resize(net_cnt);
    check(cudaMemcpy(out.batch_of_net.data(), batch_of_net.ptr, net_cnt * sizeof(int),
                     cudaMemcpyDeviceToHost), "batch id download");
    // The newest batch is kept empty on purpose; it is only a real batch once a
    // net has landed in it.
    out.batch_count = slot_base + live_cnt - (newest_size == 0 ? 1 : 0);
    out.ring_slots = ring;
    out.wavefront = wavefront;
    return true;
  } catch(const std::exception &error) {
    fprintf(stderr, "[gpu-batch-gen] %s; falling back to the CPU path\n", error.what());
    cudaGetLastError();
    out = Result();
    return false;
  }
}

}
