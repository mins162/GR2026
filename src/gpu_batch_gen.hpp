#pragma once

// GPU-accelerated batch generation (InstantGR journal version, TCAD Jan 2026,
// Section III-D).
//
// The CPU algorithm in generate_batches_rsmt() walks the nets in priority
// order and drops each one into the first batch whose occupancy map is clear
// of that net's points.  The scan is sequential by construction: where a net
// lands depends on every higher-priority net placed before it.
//
// The journal keeps the same geometry and replaces the scheduling with a
// priority-based conflict resolution scheme that fills one batch at a time:
//
//   commit   every candidate stamps its segments and points into the batch map
//            with atomicMin, so the higher-priority net (the smaller index in
//            the caller's ordering) wins a contested cell;
//   check    a candidate that still owns all of its own points is assigned to
//            the batch;
//   ruleout  a candidate whose point is covered by a net already assigned to
//            the batch can never join it, so it is pushed to a later batch;
//   repeat   the commit-check cycle until no candidate is left, then open the
//            next batch with the nets that were ruled out.
//
// Only the scheduling is new.  The marks (horizontal / vertical RSMT segments
// plus the four-cell cross that mark_3x3() stamps around every point) and the
// conflict predicate (a net's own points against the batch map) are the ones
// the CPU path uses, so the resulting batches are interchangeable with the CPU
// ones and the rest of the router does not change.
//
// The one behavioral difference is inherent to the paper's algorithm: its
// rule-out step compares a candidate against every net already assigned to the
// batch, while the CPU first-fit compares it only against the nets placed
// before it.  A net that a *lower*-priority net displaces is therefore pushed
// to the next batch here but kept by the CPU path, so the batch composition is
// "similar" (the paper's wording) rather than identical.  Both satisfy the
// invariant the router relies on: inside a batch, no net's points are covered
// by a higher-priority member's marks.

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

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

// Per-candidate state inside one commit-check cycle.
enum : unsigned char {
  kActive = 0,    // still competing for the current batch
  kAssigned = 1,  // owns all of its points, joins the current batch
  kDeferred = 2,  // blocked by an assigned net, moves to a later batch
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
  int commit_check_rounds = 0;    // summed over all batches
  int max_rounds_per_batch = 0;
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

// Every cell the net occupies in the batch map.  This is mark_3x3() plus the
// two segment walks of the CPU path, kept in one place so commit, stamp and
// release cannot drift apart.  mark_3x3()'s y loop stops at y == _y, so the
// cross is (x, y-1), (x-1, y), (x, y) and (x+1, y) — not a full 3x3 block.
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

// Rule-out step.  A candidate whose point is already covered by a net assigned
// to this batch has no chance left here, so it is handed to the next batch
// before it wastes a commit.
__global__ void prune_kernel(const int *active, int active_cnt, DeviceMarks marks,
                             const unsigned *assigned_map, unsigned char *status) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    const int net = active[i];
    unsigned char state = kActive;
    for(int k = marks.p_off[net]; k < marks.p_off[net + 1]; k++)
      if(bit_test(assigned_map, marks.p_pos[k])) { state = kDeferred; break; }
    status[i] = state;
  }
}

// Commit step.  atomicMin resolves a contested cell in favor of the net with
// the higher priority, which is what lets the check below run without knowing
// the order in which the candidates were processed.
__global__ void commit_kernel(const int *active, int active_cnt, const unsigned char *status,
                              DeviceMarks marks, int X, int Y, int *owner) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    if(status[i] != kActive) continue;
    const int net = active[i];
    visit_marks(marks, net, X, Y, [&] (int pos) { atomicMin(owner + pos, net); });
  }
}

// Check step.  Only the net's own points are tested, exactly as the CPU
// has_conflict() does: a segment may run over a lower-priority net's cells
// without that net losing its place in the batch.
__global__ void check_kernel(const int *active, int active_cnt, DeviceMarks marks,
                             const int *owner, unsigned char *status, int *batch_of_net,
                             int batch_id) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    if(status[i] != kActive) continue;
    const int net = active[i];
    bool clean = true;
    for(int k = marks.p_off[net]; k < marks.p_off[net + 1]; k++)
      if(owner[marks.p_pos[k]] != net) { clean = false; break; }
    if(clean) {
      status[i] = kAssigned;
      batch_of_net[net] = batch_id;
    }
  }
}

// An assigned net publishes all of its marks, including the cells it lost to a
// higher-priority candidate that was later ruled out.  The bitmap, not the
// owner map, is what blocks the remaining candidates from this batch.
__global__ void stamp_kernel(const int *active, int active_cnt, const unsigned char *status,
                             DeviceMarks marks, int X, int Y, unsigned *assigned_map) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    if(status[i] != kAssigned) continue;
    const int net = active[i];
    visit_marks(marks, net, X, Y, [&] (int pos) {
      atomicOr(assigned_map + (pos >> 5), 1u << (pos & 31));
    });
  }
}

// Un-commit: a candidate that did not make it drops its cells so the next
// cycle starts from the marks of the assigned nets only.  A cell holds this
// net's index only if this net won it, so the plain store cannot race.
__global__ void release_kernel(const int *active, int active_cnt, const unsigned char *status,
                               DeviceMarks marks, int X, int Y, int *owner) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < active_cnt; i += blockDim.x * gridDim.x) {
    if(status[i] != kActive) continue;
    const int net = active[i];
    visit_marks(marks, net, X, Y, [&] (int pos) {
      if(owner[pos] == net) owner[pos] = kEmpty;
    });
  }
}

// Both maps go back to empty at the end of a batch.  Clearing by the batch's
// own marks keeps the cost proportional to the work done, instead of memsetting
// the whole grid once per batch.
__global__ void reset_kernel(const int *assigned, int assigned_cnt, DeviceMarks marks,
                             int X, int Y, int *owner, unsigned *assigned_map) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < assigned_cnt; i += blockDim.x * gridDim.x) {
    const int net = assigned[i];
    visit_marks(marks, net, X, Y, [&] (int pos) {
      if(owner[pos] == net) owner[pos] = kEmpty;
      atomicAnd(assigned_map + (pos >> 5), ~(1u << (pos & 31)));
    });
  }
}

// A batch that reaches the net-count cap mid-cycle keeps the highest-priority
// nets of that cycle and gives the overshoot back to the next batch.  The nets
// stay in the reset list: their marks are on the maps and have to come off.
__global__ void unassign_kernel(const int *assigned, int overshoot, int *batch_of_net) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < overshoot; i += blockDim.x * gridDim.x)
    batch_of_net[assigned[i]] = -1;
}

struct IsStatus {
  unsigned char wanted;
  __host__ __device__ bool operator() (unsigned char state) const { return state == wanted; }
};

struct IsUnassigned {
  const int *batch_of_net;
  __host__ __device__ bool operator() (int net) const { return batch_of_net[net] < 0; }
};

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
  thrust::device_ptr<T> begin() const { return thrust::device_pointer_cast(ptr); }
};

// Returns false and leaves out untouched when the GPU cannot take the job (out
// of memory on a large grid, most likely); the caller then runs the CPU path.
inline bool generate(const HostMarks &marks, int net_cnt, int X, int Y, int max_batch_size,
                     Result &out) {
  if(net_cnt <= 0) return false;
  const long long cell_count = (long long)X * Y;
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

    DeviceArray<int> owner;
    owner.alloc(cell_count, "owner map alloc");
    fill_kernel<<<blocks_for(kMaxBlocks * kBlockSize), kBlockSize>>>(owner.ptr, cell_count, kEmpty);
    check(cudaGetLastError(), "owner map fill");

    DeviceArray<unsigned> assigned_map;
    assigned_map.alloc((cell_count + 31) / 32, "assigned map alloc");
    check(cudaMemset(assigned_map.ptr, 0, assigned_map.count * sizeof(unsigned)), "assigned map clear");

    DeviceArray<int> batch_of_net;
    batch_of_net.alloc(net_cnt, "batch id alloc");
    fill_kernel<<<blocks_for(net_cnt), kBlockSize>>>(batch_of_net.ptr, net_cnt, -1);
    check(cudaGetLastError(), "batch id fill");

    const int batch_cap = (int)std::min<long long>(net_cnt, max_batch_size);
    DeviceArray<int> remaining, remaining_next, active, active_next, assigned_list;
    DeviceArray<unsigned char> status;
    remaining.alloc(net_cnt, "remaining alloc");
    remaining_next.alloc(net_cnt, "remaining scratch alloc");
    active.alloc(net_cnt, "candidate alloc");
    active_next.alloc(net_cnt, "candidate scratch alloc");
    assigned_list.alloc(net_cnt, "assigned list alloc");
    status.alloc(net_cnt, "status alloc");
    thrust::sequence(thrust::device, remaining.begin(), remaining.begin() + net_cnt);

    int remaining_cnt = net_cnt, batch_id = 0;
    while(remaining_cnt > 0) {
      // Every net still unplaced competes for the batch, as in the paper.  The
      // CPU cap on the number of nets per batch is applied to the assignments
      // instead of the candidates, so a batch is not starved of nets that sit
      // far down the priority order.
      check(cudaMemcpy(active.ptr, remaining.ptr, remaining_cnt * sizeof(int),
                       cudaMemcpyDeviceToDevice), "candidate seed");
      int active_cnt = remaining_cnt, assigned_cnt = 0, reset_cnt = 0, rounds = 0;

      while(active_cnt > 0) {
        if(rounds == 0)
          check(cudaMemset(status.ptr, kActive, active_cnt), "status reset");
        else
          prune_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
              active.ptr, active_cnt, device_marks, assigned_map.ptr, status.ptr);
        commit_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
            active.ptr, active_cnt, status.ptr, device_marks, X, Y, owner.ptr);
        check_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
            active.ptr, active_cnt, device_marks, owner.ptr, status.ptr, batch_of_net.ptr, batch_id);
        stamp_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
            active.ptr, active_cnt, status.ptr, device_marks, X, Y, assigned_map.ptr);
        release_kernel<<<blocks_for(active_cnt), kBlockSize>>>(
            active.ptr, active_cnt, status.ptr, device_marks, X, Y, owner.ptr);
        check(cudaGetLastError(), "commit-check cycle");

        const auto assigned_end = thrust::copy_if(
            thrust::device, active.begin(), active.begin() + active_cnt, status.begin(),
            assigned_list.begin() + assigned_cnt, IsStatus{kAssigned});
        const int newly_assigned = (int)(assigned_end - (assigned_list.begin() + assigned_cnt));
        const auto active_end = thrust::copy_if(
            thrust::device, active.begin(), active.begin() + active_cnt, status.begin(),
            active_next.begin(), IsStatus{kActive});
        const int next_cnt = (int)(active_end - active_next.begin());

        assigned_cnt += newly_assigned;
        reset_cnt = assigned_cnt;
        rounds++;
        // A full batch is closed, exactly as the CPU path stops offering it to
        // further nets.  The overshoot is the tail of the last cycle, which is
        // the least valuable part of it: the cycle assigns in priority order.
        if(assigned_cnt >= batch_cap) {
          const int overshoot = assigned_cnt - batch_cap;
          if(overshoot > 0) {
            unassign_kernel<<<blocks_for(overshoot), kBlockSize>>>(
                assigned_list.ptr + batch_cap, overshoot, batch_of_net.ptr);
            check(cudaGetLastError(), "batch cap trim");
          }
          assigned_cnt = batch_cap;
          break;
        }
        // The highest-priority candidate wins every cell it asks for, so it is
        // always assigned or ruled out.  Stopping here would mean that
        // invariant broke; leave the rest to the next batch rather than spin.
        if(newly_assigned == 0 && next_cnt == active_cnt) break;
        std::swap(active.ptr, active_next.ptr);
        active_cnt = next_cnt;
      }

      if(assigned_cnt == 0)
        throw std::runtime_error("batch " + std::to_string(batch_id) + " accepted no net");
      reset_kernel<<<blocks_for(reset_cnt), kBlockSize>>>(
          assigned_list.ptr, reset_cnt, device_marks, X, Y, owner.ptr, assigned_map.ptr);
      check(cudaGetLastError(), "batch reset");

      out.commit_check_rounds += rounds;
      out.max_rounds_per_batch = std::max(out.max_rounds_per_batch, rounds);
      batch_id++;

      const auto remaining_end = thrust::copy_if(
          thrust::device, remaining.begin(), remaining.begin() + remaining_cnt,
          remaining_next.begin(), IsUnassigned{batch_of_net.ptr});
      remaining_cnt = (int)(remaining_end - remaining_next.begin());
      std::swap(remaining.ptr, remaining_next.ptr);
    }

    out.batch_of_net.resize(net_cnt);
    check(cudaMemcpy(out.batch_of_net.data(), batch_of_net.ptr, net_cnt * sizeof(int),
                     cudaMemcpyDeviceToHost), "batch id download");
    out.batch_count = batch_id;
    return true;
  } catch(const std::exception &error) {
    fprintf(stderr, "[gpu-batch-gen] %s; falling back to the CPU path\n", error.what());
    cudaGetLastError();
    out = Result();
    return false;
  }
}

}
