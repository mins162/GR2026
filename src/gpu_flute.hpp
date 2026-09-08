#pragma once

// GPU-FLUTE (Guo, Gu and Lin, ICCAD 2022) integration.
//
// The original FLUTE implementation in flute.hpp is depth-first and allocates
// a temporary Tree at every recursive call.  That is a poor fit for CUDA.  This
// file keeps FLUTE's lookup-table representation and replaces the recursive
// execution order with the paper's levelized representation:
//
//   Hanan grids -> break levels -> low-degree LUT -> reverse merge levels.
//
// A level owns contiguous metadata, pin and tree buffers.  Children are
// assigned by an exclusive scan, so a CUDA thread can generate a subnet without
// allocating memory or communicating with another thread.  The code is kept in
// a header because InstantGR is built as one CUDA translation unit.

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include <climits>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_flute {

constexpr int kLutDegree = DEGREE;
constexpr int kBlockSize = 128;  // GPU-FLUTE paper uses 128 threads/block.
constexpr int kMaxAccuracy = 16; // Paper evaluates A up to 12.
constexpr int kThrustSortDegree = 128;

inline void check(cudaError_t status, const char* operation);

struct Branch {
  int x;
  int y;
  int n;
};

// Physical-coordinate result for the optional GPU tree-center pass.  The host
// maps this back to its finalized RSMT node after downloading the FLUTE tree.
struct TreeCenter {
  int x;
  int y;
};

// Device form of flute::csoln.  Do not use pointers in this structure: the
// host LUT is flattened before the first kernel launch.
struct Solution {
  unsigned char parent;
  unsigned char seg[12];
  unsigned char row[kLutDegree - 2];
  unsigned char col[kLutDegree - 2];
  unsigned char neighbor[2 * kLutDegree - 2];
};

struct Lut {
  Solution* solutions = nullptr;
  int* offsets = nullptr;  // [degree * MGROUP + permutation]
  int* counts = nullptr;   // number of solutions for the same key
  int solution_count = 0;

  void release() {
    cudaFree(solutions);
    cudaFree(offsets);
    cudaFree(counts);
    solutions = nullptr;
    offsets = nullptr;
    counts = nullptr;
    solution_count = 0;
  }
};

struct Profile {
  double host_wall_seconds = 0.0;
  double h2d_api_seconds = 0.0;
  double d2h_api_seconds = 0.0;
  double device_pipeline_seconds = 0.0;
  double tree_center_device_seconds = 0.0;
  size_t h2d_bytes = 0;
  size_t d2h_bytes = 0;
};

// A Hanan-grid subnet.  xs, ys, and s reside in the same level-wide buffers.
// child_begin/count are retained for the reverse merge pass.
struct Net {
  int degree;
  int pin_offset;
  int acc;
  int child_begin;
  int child_count;
  int break_kind;  // own outgoing split: 0 leaf, 1 direct, 2 candidates.
  int merge_kind;  // immutable relation to this node's sibling: 1/2/3.
  int break_pos;
  int tree_length;
};

struct Level {
  int net_count = 0;
  int pin_count = 0;
  Net* nets = nullptr;
  int* xs = nullptr;
  int* ys = nullptr;
  int* s = nullptr;
  Branch* trees = nullptr;
  int* tree_offsets = nullptr;

  void release() {
    cudaFree(nets);
    cudaFree(xs);
    cudaFree(ys);
    cudaFree(s);
    cudaFree(trees);
    cudaFree(tree_offsets);
    nets = nullptr;
    xs = ys = s = nullptr;
    trees = nullptr;
    tree_offsets = nullptr;
    net_count = pin_count = 0;
  }
};

// Scratch is indexed by the final root-tree branch buffer.  It deliberately
// stays device-resident: the center pass runs before root_result.trees is
// copied back to the host.
struct TreeCenterWorkspace {
  int branch_count = 0;
  int* hash = nullptr;
  int* canonical = nullptr;
  int* degree = nullptr;
  int* offset = nullptr;
  int* write = nullptr;
  int* alive = nullptr;
  int* queue_a = nullptr;
  int* queue_b = nullptr;
  int* adjacency = nullptr;

  void allocate(int count) {
    branch_count = count;
    check(cudaMalloc(&hash, sizeof(int) * count * 2), "allocate tree-center hash");
    check(cudaMalloc(&canonical, sizeof(int) * count), "allocate tree-center canonical ids");
    check(cudaMalloc(&degree, sizeof(int) * count), "allocate tree-center degrees");
    check(cudaMalloc(&offset, sizeof(int) * count), "allocate tree-center offsets");
    check(cudaMalloc(&write, sizeof(int) * count), "allocate tree-center writes");
    check(cudaMalloc(&alive, sizeof(int) * count), "allocate tree-center live flags");
    check(cudaMalloc(&queue_a, sizeof(int) * count), "allocate tree-center queue A");
    check(cudaMalloc(&queue_b, sizeof(int) * count), "allocate tree-center queue B");
    check(cudaMalloc(&adjacency, sizeof(int) * count * 2), "allocate tree-center adjacency");
  }

  void release() {
    cudaFree(hash); cudaFree(canonical); cudaFree(degree); cudaFree(offset);
    cudaFree(write); cudaFree(alive); cudaFree(queue_a); cudaFree(queue_b);
    cudaFree(adjacency);
    hash = canonical = degree = offset = write = alive = queue_a = queue_b = adjacency = nullptr;
    branch_count = 0;
  }
};

// The initial Hanan-grid layer is constructed from InstantGR's device pin
// database.  `net_ids` indexes the router nets, while `pin_acc_num` and
// `pins` are the existing compact CSR representation.  x_coords/y_coords are
// physical track coordinates (not grid indices), matching flute::flute().
//
// One thread owns one root net.  This O(d^2) initialization is intentional:
// high-degree nets are the GPU-FLUTE input; low-degree root nets stay on the
// existing CPU FLUTE path.  All repeated break/merge work afterwards is fully
// level-parallel and does not return to the CPU.
__global__ void make_hanan_kernel(const int* net_ids, int net_count,
                                  const int* pin_acc_num, const int* pins,
                                  const int* x_coords, const int* y_coords,
                                  int grid_x, int grid_y, Net* roots, int* xs, int* ys,
                                  int* permutations, int accuracy) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= net_count) return;

  Net& net = roots[id];
  const int start = pin_acc_num[net_ids[id]];
  const int degree = pin_acc_num[net_ids[id] + 1] - start;
  net.degree = degree;
  net.acc = accuracy;
  net.pin_offset = start;  // root buffers use the same packed pin offsets.
  net.child_begin = -1;
  net.child_count = 0;
  net.break_kind = 0;
  net.merge_kind = 0;
  net.break_pos = -1;
  net.tree_length = 0;

  int* nx = xs + start;
  int* ny = ys + start;
  int* ns = permutations + start;
  for (int i = 0; i < degree; ++i) {
    const int pin = pins[start + i];
    // Router pins are 3-D IDs.  FLUTE operates on their 2-D projection.
    nx[i] = x_coords[(pin / grid_y) % grid_x];
    ny[i] = y_coords[pin % grid_y];
    ns[i] = i;
  }

  // Large roots are sorted by Thrust after this kernel.  Keeping the small
  // path below preserves FLUTE's exact selection-sort tie behavior where it
  // matters for the normal 10..128-pin GPU batch.
  if (degree > kThrustSortDegree) return;

  // Same x ordering as flute::flute(): x ascending, y descending on ties.
  for (int i = 0; i + 1 < degree; ++i) {
    int best = i;
    for (int j = i + 1; j < degree; ++j) {
      const int best_pin = pins[start + ns[best]];
      const int test_pin = pins[start + ns[j]];
      const int best_y = y_coords[best_pin % grid_y];
      const int test_y = y_coords[test_pin % grid_y];
      if (nx[j] < nx[best] || (nx[j] == nx[best] && test_y > best_y)) best = j;
    }
    if (best != i) {
      const int tx = nx[i]; nx[i] = nx[best]; nx[best] = tx;
      const int ts = ns[i]; ns[i] = ns[best]; ns[best] = ts;
    }
  }

  // flute::flute() selection-sorts y on the already x-sorted point-pointer
  // array.  Keep that exact starting order and its strict-y comparison:
  // equal-y entries are therefore reordered only by the same swap history as
  // the CPU reference, not by an added tie-break rule.
  for (int i = 0; i < degree; ++i) {
    const int pin = pins[start + ns[i]];
    ny[i] = y_coords[pin % grid_y];
    ns[i] = i;  // x rank; it moves together with ny during y selection-sort.
  }
  for (int i = 0; i + 1 < degree; ++i) {
    int best = i;
    for (int j = i + 1; j < degree; ++j) {
      if (ny[j] < ny[best]) {
        best = j;
      }
    }
    if (best != i) {
      const int ty = ny[i]; ny[i] = ny[best]; ny[best] = ty;
      const int ts = ns[i]; ns[i] = ns[best]; ns[best] = ts;
    }
  }

}

__global__ void make_x_sort_keys_kernel(const int* xs, const int* ys,
                                        unsigned long long* keys, int count) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count) return;
  // Ascending x, then descending y: exactly FLUTE's x-sort comparator.
  keys[id] = (static_cast<unsigned long long>(static_cast<unsigned int>(xs[id])) << 32) |
             static_cast<unsigned int>(INT_MAX - ys[id]);
}

__global__ void finish_x_sort_kernel(const unsigned long long* keys, int* xs,
                                     int* ranks, int count) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count) return;
  xs[id] = static_cast<int>(keys[id] >> 32);
  ranks[id] = id;
}

inline void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("GPU-FLUTE ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

__device__ inline unsigned int tree_center_hash(int x, int y) {
  return static_cast<unsigned int>(x) * 0x9e3779b1u ^
         static_cast<unsigned int>(y) * 0x85ebca6bu;
}

// One block owns one completed high-degree FLUTE tree.  First, equal-coordinate
// FLUTE branches are contracted exactly as gpu_flute_tree_to_graph() does on
// the host.  Two block-parallel BFS traversals then find a diameter and its
// middle node, which minimizes the maximum RSMT edge-hop depth.
__global__ void tree_center_kernel(const Branch* trees, const int* tree_offsets,
                                   int tree_count, int* hash, int* canonical,
                                   int* degree, int* offset, int* write, int* alive,
                                   int* queue_a, int* queue_b, int* adjacency,
                                   TreeCenter* centers) {
  const int tree_id = blockIdx.x;
  if(tree_id >= tree_count) return;
  const int base = tree_offsets[tree_id];
  const int branch_count = tree_offsets[tree_id + 1] - base;
  const int hash_base = base * 2;
  const int hash_count = branch_count * 2;

  __shared__ int frontier_count;
  __shared__ int next_frontier_count;
  __shared__ int round;
  __shared__ int is_contracted_tree;
  __shared__ int bfs_start;

  for(int i = threadIdx.x; i < hash_count; i += blockDim.x) hash[hash_base + i] = -1;
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x) {
    degree[base + i] = 0;
    offset[base + i] = -1;
    write[base + i] = 0;
    alive[base + i] = 0;
  }
  __syncthreads();

  // Parallel coordinate deduplication.  A representative is an absolute
  // branch index, so non-representatives naturally consume no graph storage.
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x) {
    const Branch branch = trees[base + i];
    int slot = tree_center_hash(branch.x, branch.y) % hash_count;
    while(true) {
      const int prior = atomicCAS(hash + hash_base + slot, -1, base + i);
      if(prior == -1) {
        canonical[base + i] = base + i;
        break;
      }
      const Branch prior_branch = trees[prior];
      if(prior_branch.x == branch.x && prior_branch.y == branch.y) {
        canonical[base + i] = prior;
        break;
      }
      ++slot;
      if(slot == hash_count) slot = 0;
    }
  }
  __syncthreads();

  // Build the contracted undirected adjacency list in a per-tree global slice.
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x) {
    const int parent = trees[base + i].n;
    if(parent < 0 || parent >= branch_count) continue;
    const int u = canonical[base + i];
    const int v = canonical[base + parent];
    if(u != v) {
      atomicAdd(degree + u, 1);
      atomicAdd(degree + v, 1);
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    int adjacency_count = 0;
    int node_count = 0;
    for(int i = 0; i < branch_count; ++i) if(canonical[base + i] == base + i) {
      offset[base + i] = adjacency_count;
      adjacency_count += degree[base + i];
      alive[base + i] = 1;
      ++node_count;
    }
    frontier_count = 0;
    next_frontier_count = 0;
    round = 0;
    // The host conversion runs Kruskal after coordinate contraction.  When
    // contraction creates a cycle/parallel edge, this raw device graph is not
    // the exact host RSMT.  Do not select an unsafe root in that rare case.
    is_contracted_tree = adjacency_count == 2 * (node_count - 1);
  }
  __syncthreads();
  if(!is_contracted_tree) {
    if(threadIdx.x == 0) centers[tree_id] = {-1, -1};
    return;
  }
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x)
    if(canonical[base + i] == base + i) write[base + i] = 0;
  __syncthreads();
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x) {
    const int parent = trees[base + i].n;
    if(parent < 0 || parent >= branch_count) continue;
    const int u = canonical[base + i];
    const int v = canonical[base + parent];
    if(u != v) {
      adjacency[2 * base + offset[u] + atomicAdd(write + u, 1)] = v;
      adjacency[2 * base + offset[v] + atomicAdd(write + v, 1)] = u;
    }
  }
  __syncthreads();
  // BFS #1: arbitrary node -> one diameter endpoint.  `degree` is reused as
  // distance after adjacency construction; `write` retains static edge counts.
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x)
    if(canonical[base + i] == base + i) degree[base + i] = -1;
  __syncthreads();
  if(threadIdx.x == 0) {
    bfs_start = 0;
    while(canonical[base + bfs_start] != base + bfs_start) ++bfs_start;
    degree[base + bfs_start] = 0;
    queue_a[base] = bfs_start;
    frontier_count = 1;
    round = 0;
  }
  __syncthreads();
  while(frontier_count > 0) {
    if(threadIdx.x == 0) next_frontier_count = 0;
    __syncthreads();
    int* current_queue = (round & 1) ? queue_b : queue_a;
    int* next_queue = (round & 1) ? queue_a : queue_b;
    const int current_count = frontier_count;
    for(int q = threadIdx.x; q < current_count; q += blockDim.x) {
      const int u = base + current_queue[base + q];
      for(int edge = offset[u]; edge < offset[u] + write[u]; ++edge) {
        const int v = adjacency[2 * base + edge];
        if(atomicCAS(degree + v, -1, degree[u] + 1) == -1)
          next_queue[base + atomicAdd(&next_frontier_count, 1)] = v - base;
      }
    }
    __syncthreads();
    if(threadIdx.x == 0) { frontier_count = next_frontier_count; ++round; }
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    for(int i = 0; i < branch_count; ++i)
      if(canonical[base + i] == base + i && degree[base + i] > degree[base + bfs_start])
        bfs_start = i;
  }
  __syncthreads();

  // BFS #2: diameter endpoint -> opposite endpoint, retaining parents for
  // the middle-node walk back along the diameter.
  for(int i = threadIdx.x; i < branch_count; i += blockDim.x)
    if(canonical[base + i] == base + i) { degree[base + i] = -1; alive[base + i] = -1; }
  __syncthreads();
  if(threadIdx.x == 0) {
    degree[base + bfs_start] = 0;
    alive[base + bfs_start] = base + bfs_start;
    queue_a[base] = bfs_start;
    frontier_count = 1;
    round = 0;
  }
  __syncthreads();
  while(frontier_count > 0) {
    if(threadIdx.x == 0) next_frontier_count = 0;
    __syncthreads();
    int* current_queue = (round & 1) ? queue_b : queue_a;
    int* next_queue = (round & 1) ? queue_a : queue_b;
    const int current_count = frontier_count;
    for(int q = threadIdx.x; q < current_count; q += blockDim.x) {
      const int u = base + current_queue[base + q];
      for(int edge = offset[u]; edge < offset[u] + write[u]; ++edge) {
        const int v = adjacency[2 * base + edge];
        if(atomicCAS(degree + v, -1, degree[u] + 1) == -1) {
          alive[v] = u;
          next_queue[base + atomicAdd(&next_frontier_count, 1)] = v - base;
        }
      }
    }
    __syncthreads();
    if(threadIdx.x == 0) { frontier_count = next_frontier_count; ++round; }
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    int endpoint = bfs_start;
    for(int i = 0; i < branch_count; ++i)
      if(canonical[base + i] == base + i && degree[base + i] > degree[endpoint]) endpoint = i;
    for(int step = 0; step < degree[endpoint] / 2; ++step) endpoint = alive[base + endpoint] - base;
    centers[tree_id] = {trees[base + endpoint].x, trees[base + endpoint].y};
  }
}

inline void profiled_memcpy(void* dst, const void* src, size_t bytes,
                            cudaMemcpyKind kind, const char* operation,
                            Profile* profile) {
  const auto start = std::chrono::steady_clock::now();
  check(cudaMemcpy(dst, src, bytes, kind), operation);
  if (profile == nullptr) return;
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  if (kind == cudaMemcpyHostToDevice) {
    profile->h2d_api_seconds += seconds;
    profile->h2d_bytes += bytes;
  } else if (kind == cudaMemcpyDeviceToHost) {
    profile->d2h_api_seconds += seconds;
    profile->d2h_bytes += bytes;
  }
}

inline int blocks(int n) { return (n + kBlockSize - 1) / kBlockSize; }

// CUDA's segmented sort is unnecessary for the normal small-root path.  For
// a rare large root, however, a per-net Thrust radix sort avoids the O(d^2)
// selection sort used solely to reproduce FLUTE's small-net tie behavior.
inline void thrust_sort_large_root_hanan(const std::vector<int>& net_ids,
                                         const std::vector<int>& pin_offsets,
                                         int* xs, int* ys, int* s) {
  int max_degree = 0;
  for (int net_id : net_ids)
    max_degree = max(max_degree, pin_offsets[net_id + 1] - pin_offsets[net_id]);
  if (max_degree <= kThrustSortDegree) return;

  unsigned long long* keys = nullptr;
  check(cudaMalloc(&keys, sizeof(unsigned long long) * max_degree),
        "allocate Thrust Hanan keys");
  for (int net_id : net_ids) {
    const int offset = pin_offsets[net_id];
    const int degree = pin_offsets[net_id + 1] - offset;
    if (degree <= kThrustSortDegree) continue;
    int* net_xs = xs + offset;
    int* net_ys = ys + offset;
    int* net_s = s + offset;
    make_x_sort_keys_kernel<<<blocks(degree), kBlockSize>>>(net_xs, net_ys, keys, degree);
    check(cudaGetLastError(), "construct Thrust Hanan keys");

    auto key_begin = thrust::device_pointer_cast(keys);
    auto value_begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(net_ys), thrust::device_pointer_cast(net_s)));
    thrust::sort_by_key(thrust::device, key_begin, key_begin + degree, value_begin);
    finish_x_sort_kernel<<<blocks(degree), kBlockSize>>>(keys, net_xs, net_s, degree);
    check(cudaGetLastError(), "finish Thrust x sort");
    // Stable y sort keeps x order for equal-y pins.  It is deterministic and
    // produces a legal Hanan permutation for large roots.
    thrust::stable_sort_by_key(thrust::device,
                               thrust::device_pointer_cast(net_ys),
                               thrust::device_pointer_cast(net_ys) + degree,
                               thrust::device_pointer_cast(net_s));
    check(cudaGetLastError(), "finish Thrust y sort");
  }
  cudaFree(keys);
}

// Pack flute.hpp's host LUT once.  Its original pointer-of-pointer layout is
// not valid on the device.  A negative offset denotes an unused permutation.
inline Lut upload_lut(Profile* profile = nullptr) {
  Lut lut;
  const int offset_count = (kLutDegree + 1) * MGROUP;
  std::vector<int> host_offsets(offset_count, -1);
  std::vector<int> host_counts(offset_count, 0);
  std::vector<Solution> host_solutions;

  for (int degree = 4; degree <= kLutDegree; ++degree) {
    for (int group = 0; group < numgrp[degree]; ++group) {
      const int count = numsoln[degree][group];
      if (count == 0 || LUT[degree][group] == nullptr) continue;
      host_offsets[degree * MGROUP + group] =
          static_cast<int>(host_solutions.size());
      host_counts[degree * MGROUP + group] = count;
      for (int i = 0; i < count; ++i) {
        const flute::csoln& source = LUT[degree][group][i];
        Solution target{};
        target.parent = source.parent;
        for (int j = 0; j < 12; ++j) target.seg[j] = source.seg[j];
        for (int j = 0; j < kLutDegree - 2; ++j) {
          target.row[j] = source.row[j];
          target.col[j] = source.col[j];
        }
        for (int j = 0; j < 2 * kLutDegree - 2; ++j)
          target.neighbor[j] = source.neighbor[j];
        host_solutions.push_back(target);
      }
    }
  }

  lut.solution_count = static_cast<int>(host_solutions.size());
  check(cudaMalloc(&lut.offsets, sizeof(int) * host_offsets.size()),
        "allocate LUT offsets");
  profiled_memcpy(lut.offsets, host_offsets.data(), sizeof(int) * host_offsets.size(),
                  cudaMemcpyHostToDevice, "upload LUT offsets", profile);
  check(cudaMalloc(&lut.counts, sizeof(int) * host_counts.size()),
        "allocate LUT counts");
  profiled_memcpy(lut.counts, host_counts.data(), sizeof(int) * host_counts.size(),
                  cudaMemcpyHostToDevice, "upload LUT counts", profile);
  check(cudaMalloc(&lut.solutions, sizeof(Solution) * host_solutions.size()),
        "allocate LUT solutions");
  profiled_memcpy(lut.solutions, host_solutions.data(),
                  sizeof(Solution) * host_solutions.size(), cudaMemcpyHostToDevice,
                  "upload LUT solutions", profile);
  return lut;
}

// The flatten + upload above costs ~0.2 s regardless of net count, which is
// more than the whole CPU FLUTE loop on small designs.  Upload once per
// process and keep the LUT resident (~130 MB) instead of per call.
inline const Lut& resident_lut(Profile* profile = nullptr) {
  static const Lut lut = upload_lut(profile);
  return lut;
}

__device__ inline int abs_i(int value) { return value < 0 ? -value : value; }

__device__ inline int group_count(int degree) {
  // Device code cannot dereference flute.hpp's host global numgrp[].
  switch (degree) {
    case 4: return 6;
    case 5: return 30;
    case 6: return 180;
    case 7: return 1260;
    case 8: return 10080;
    case 9: return 90720;
    default: return 0;
  }
}

__device__ inline int tree_wirelength(const Branch* tree, int degree) {
  int length = 0;
  for (int i = 0; i < 2 * degree - 2; ++i) {
    const Branch& a = tree[i];
    const Branch& b = tree[a.n];
    length += abs_i(a.x - b.x) + abs_i(a.y - b.y);
  }
  return length;
}

// Exact device equivalent of flute::flutes_LD().  Input arrays are already in
// Hanan-grid order.  This is deliberately separate from the high-degree pass:
// every leaf is independently evaluated by one GPU thread.
__device__ inline void solve_low_degree(const Net& net, const int* xs,
                                        const int* ys, const int* s,
                                        const Solution* solutions,
                                        const int* offsets, const int* counts,
                                        Branch* out) {
  const int d = net.degree;
  const int* nx = xs + net.pin_offset;
  const int* ny = ys + net.pin_offset;
  const int* ns = s + net.pin_offset;
  if (d == 2) {
    out[0] = {nx[ns[0]], ny[0], 1};
    out[1] = {nx[ns[1]], ny[1], 1};
    return;
  }
  if (d == 3) {
    out[0] = {nx[ns[0]], ny[0], 3};
    out[1] = {nx[ns[1]], ny[1], 3};
    out[2] = {nx[ns[2]], ny[2], 3};
    out[3] = {nx[1], ny[1], 3};
    return;
  }

  int permutation = 0;
  if (ns[0] < ns[2]) ++permutation;
  if (ns[1] < ns[2]) ++permutation;
  for (int i = 3; i < d; ++i) {
    int pi = ns[i];
    for (int j = d - 1; j > i; --j)
      if (ns[j] < ns[i]) --pi;
    permutation = pi + (i + 1) * permutation;
  }

  int delta[2 * kLutDegree - 2]{};
  bool hflip = false;
  const int groups = group_count(d);
  if (permutation >= groups) {
    hflip = true;
    permutation = 2 * groups - 1 - permutation;
  }
  for (int i = 1; i <= d - 3; ++i) {
    delta[i] = ny[i + 1] - ny[i];
    delta[d - 1 + i] = hflip ? nx[d - 1 - i] - nx[d - 2 - i]
                               : nx[i + 1] - nx[i];
  }

  const int lut_offset = offsets[d * MGROUP + permutation];
  const Solution* candidate = solutions + lut_offset;
  int values[MPOWV + 1];
  const int base_length = nx[d - 1] - nx[0] + ny[d - 1] - ny[0];
  int best_length = base_length;
  for (int i = 0; candidate->seg[i] != 0; ++i)
    best_length += delta[candidate->seg[i]];
  // Match flute::flutes_LD(): l[0] is the unaugmented bounding-box length,
  // LUT[0] becomes l[1], and csoln::parent can legally refer to either.
  values[0] = base_length;
  values[1] = best_length;
  const Solution* best = candidate;
  const int count = counts[d * MGROUP + permutation];
  for (int j = 2; j <= count; ++j) {
    ++candidate;
    int value = values[candidate->parent];
    for (int i = 0; candidate->seg[i] != 0; ++i) value += delta[candidate->seg[i]];
    for (int i = 10; candidate->seg[i] != 0; --i) value -= delta[candidate->seg[i]];
    if (value < best_length) {
      best_length = value;
      best = candidate;
    }
    values[j] = value;
  }

  out[0] = {nx[ns[0]], ny[0], 0};
  out[1] = {nx[ns[1]], ny[1], 0};
  for (int i = 2; i < d - 2; ++i)
    out[i] = {nx[ns[i]], ny[i], static_cast<int>(best->neighbor[i])};
  out[d - 2] = {nx[ns[d - 2]], ny[d - 2], 0};
  out[d - 1] = {nx[ns[d - 1]], ny[d - 1], 0};
  if (hflip) {
    out[0].n = best->neighbor[ns[1] < ns[0] ? 1 : 0];
    out[1].n = best->neighbor[ns[1] < ns[0] ? 0 : 1];
    out[d - 2].n = best->neighbor[ns[d - 1] < ns[d - 2] ? d - 1 : d - 2];
    out[d - 1].n = best->neighbor[ns[d - 1] < ns[d - 2] ? d - 2 : d - 1];
    for (int i = d; i < 2 * d - 2; ++i)
      out[i] = {nx[d - 1 - best->col[i - d]], ny[best->row[i - d]],
                static_cast<int>(best->neighbor[i])};
  } else {
    out[0].n = best->neighbor[ns[0] < ns[1] ? 1 : 0];
    out[1].n = best->neighbor[ns[0] < ns[1] ? 0 : 1];
    out[d - 2].n = best->neighbor[ns[d - 2] < ns[d - 1] ? d - 1 : d - 2];
    out[d - 1].n = best->neighbor[ns[d - 2] < ns[d - 1] ? d - 2 : d - 1];
    for (int i = d; i < 2 * d - 2; ++i)
      out[i] = {nx[best->col[i - d]], ny[best->row[i - d]],
                static_cast<int>(best->neighbor[i])};
  }
}

__global__ void solve_leaf_kernel(Net* nets, int net_count, const int* xs,
                                  const int* ys, const int* s,
                                  const Solution* solutions, const int* offsets,
                                  const int* counts,
                                  const int* tree_offsets, Branch* trees) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= net_count || nets[id].degree > kLutDegree) return;
  solve_low_degree(nets[id], xs, ys, s, solutions, offsets, counts,
                   trees + tree_offsets[id]);
  nets[id].tree_length = tree_wirelength(trees + tree_offsets[id], nets[id].degree);
}

__device__ inline int inverse_s(const int* s, int d, int x_rank) {
  for (int i = 0; i < d; ++i)
    if (s[i] == x_rank) return i;
  return -1;
}

// Storage-backed replica of the per-recursion score inputs in
// flute::flutes_MD(): the inverse permutation si[], penalty[], distx[] and
// disty[] are computed once per subnet in O(d), so every candidate score
// afterwards is O(1).  The original storage-free translation recomputed
// dist_y per candidate through inverse_s() linear scans instead, which is
// O(d^2) per candidate and O(d^3) per subnet on a single thread -- fine for
// the degree <= 51 nets the GPU-FLUTE paper evaluates, but a 2153-pin net
// (bsg_chip) turned that into a ~100s kernel.  Float operation order below
// deliberately mirrors flute::flutes_MD() so the trees stay identical.
__device__ inline void precompute_break_arrays(const Net& net, const int* all_xs,
                                               const int* all_ys, const int* all_s,
                                               int* all_si, float* all_penalty,
                                               float* all_distx, float* all_disty) {
  const int d = net.degree;
  const int* xs = all_xs + net.pin_offset;
  const int* ys = all_ys + net.pin_offset;
  const int* s = all_s + net.pin_offset;
  int* si = all_si + net.pin_offset;
  float* penalty = all_penalty + net.pin_offset;
  float* distx = all_distx + net.pin_offset;
  float* disty = all_disty + net.pin_offset;

  for (int r = 0; r < d; ++r) si[s[r]] = r;

  // penalty[]: identical accumulation sequence to flute::flutes_MD().  The
  // middle ranks are written twice by the x loop; the overwrite equals the
  // reference running sum because the first write is always the 0.0f start.
  const float ccc = fmaxf(0.41f - 0.005f * d, 0.1f);
  const float dx = ccc * (xs[d - 2] - xs[1]) / (d - 3);
  const float dy = ccc * (ys[d - 2] - ys[1]) / (d - 3);
  float pnlty = 0.0f;
  for (int r = d / 2; r >= 2; --r, pnlty += dx) {
    penalty[r] = pnlty;
    penalty[d - 1 - r] = pnlty;
  }
  penalty[1] = pnlty; penalty[d - 2] = pnlty;
  penalty[0] = pnlty; penalty[d - 1] = pnlty;
  pnlty = dy;
  for (int r = d / 2 - 1; r >= 2; --r, pnlty += dy) {
    penalty[s[r]] += pnlty;
    penalty[s[d - 1 - r]] += pnlty;
  }
  penalty[s[1]] += pnlty; penalty[s[d - 2]] += pnlty;
  penalty[s[0]] += pnlty; penalty[s[d - 1]] += pnlty;

  // distx[]/disty[]: two running min/max sweeps, exactly flutes_MD().  Only
  // positions in [lb, ub] are ever queried by candidates; entries the suffix
  // sweep reaches beyond ub take 0 as their prefix instead of reading the
  // unwritten slot.
  const int lb = max((d - net.acc) / 5, 2);
  const int ub = d - 1 - lb;
  const int xydiff = (xs[d - 1] - xs[0]) - (ys[d - 1] - ys[0]);
  int mins = min(s[0], s[1]), maxs = max(s[0], s[1]);
  int minsi = min(si[0], si[1]), maxsi = max(si[0], si[1]);
  for (int r = 2; r <= ub; ++r) {
    mins = min(mins, s[r]); maxs = max(maxs, s[r]);
    distx[r] = static_cast<float>(xs[maxs] - xs[mins]);
    minsi = min(minsi, si[r]); maxsi = max(maxsi, si[r]);
    disty[r] = static_cast<float>(ys[maxsi] - ys[minsi] + xydiff);
  }
  mins = min(s[d - 2], s[d - 1]); maxs = max(s[d - 2], s[d - 1]);
  minsi = min(si[d - 2], si[d - 1]); maxsi = max(si[d - 2], si[d - 1]);
  for (int r = d - 3; r >= lb; --r) {
    mins = min(mins, s[r]); maxs = max(maxs, s[r]);
    const float prefix_x = r <= ub ? distx[r] : 0.0f;
    distx[r] = prefix_x + xs[maxs] - xs[mins];
    minsi = min(minsi, si[r]); maxsi = max(maxsi, si[r]);
    const float prefix_y = r <= ub ? disty[r] : 0.0f;
    disty[r] = prefix_y + ys[maxsi] - ys[minsi];
  }
}

// One candidate score in O(1) over the precomputed arrays.  The final
// expression keeps flute::flutes_MD()'s exact left-to-right float order.
__device__ inline float break_score_pre(const Net& net, const int* all_xs,
                                        const int* all_ys, const int* all_s,
                                        const int* all_si, const float* all_penalty,
                                        const float* all_distx, const float* all_disty,
                                        int candidate) {
  const int d = net.degree;
  const int* xs = all_xs + net.pin_offset;
  const int* ys = all_ys + net.pin_offset;
  const int* s = all_s + net.pin_offset;
  const int* si = all_si + net.pin_offset;
  const float* penalty = all_penalty + net.pin_offset;
  const float* distx = all_distx + net.pin_offset;
  const float* disty = all_disty + net.pin_offset;
  const int lb = max((d - net.acc) / 5, 2);
  const int p = candidate / 2 + lb;
  constexpr float aa = 0.6f;
  constexpr float bb = 0.3f;
  const float ddd = 4.8f / (d - 1);
  if ((candidate & 1) == 0) {  // break in x
    const int sip = si[p];
    const float side = (sip <= 1) ? aa * (ys[2] - ys[1]) :
                       (sip >= d - 2) ? aa * (ys[d - 2] - ys[d - 3]) :
                       bb * (ys[sip + 1] - ys[sip - 1]);
    return (xs[p + 1] - xs[p - 1]) - penalty[p] - side - ddd * disty[p];
  }
  const int x_rank = s[p];  // break in y
  const float side = (x_rank <= 1) ? aa * (xs[2] - xs[1]) :
                     (x_rank >= d - 2) ? aa * (xs[d - 2] - xs[d - 3]) :
                     bb * (xs[x_rank + 1] - xs[x_rank - 1]);
  return (ys[p + 1] - ys[p - 1]) - penalty[x_rank] - side - ddd * distx[p];
}

__device__ inline int candidate_count(const Net& net) {
  const int d = net.degree;
  const int lb = max((d - net.acc) / 5, 2);
  const int ub = d - 1 - lb;
  const int total = 2 * (ub - lb + 1);
  return net.acc >= total ? total - 1 : net.acc;
}

__device__ inline int child_accuracy(int acc) {
  return acc <= 3 ? 1 : acc / 2;
}

// Returns the two provably optimal FLUTE decompositions checked before the
// heuristic score loop.  Values 1 and 2 correspond to the two early-return
// branches in flute::flutes_MD(); zero means use candidate breaks.
__device__ inline int direct_break(const Net& net, const int* all_s, int* split) {
  const int d = net.degree;
  const int* s = all_s + net.pin_offset;
  if (s[0] < s[d - 1]) {
    int ms = max(s[0], s[1]);
    for (int i = 2; i <= ms; ++i) ms = max(ms, s[i]);
    if (ms <= d - 3) { *split = ms; return 1; }
  } else {
    int ms = min(s[0], s[1]);
    for (int i = 2; i <= d - 1 - ms; ++i) ms = min(ms, s[i]);
    if (ms >= 2) { *split = ms; return 2; }
  }
  return 0;
}

// Algorithm 2 in the paper.  The output arrays have N+1 entries: index zero
// is initialized by the caller and entries [1,N] are exclusively scanned.
__global__ void estimate_break_kernel(Net* level, int count, const int* xs,
                                      const int* ys, const int* s,
                                      int* next_net_sizes, int* next_pin_sizes,
                                      int* has_high_degree) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count) return;
  Net& net = level[id];
  if (net.degree <= kLutDegree) {
    next_net_sizes[id + 1] = 0;
    next_pin_sizes[id + 1] = 0;
    return;
  }
  int direct_split = 0;
  if (direct_break(net, s, &direct_split)) {
    next_net_sizes[id + 1] = 2;
    next_pin_sizes[id + 1] = net.degree + 2;
    atomicExch(has_high_degree, 1);
    return;
  }
  const int candidates = candidate_count(net);
  // Every heuristic FLUTE candidate produces two overlapping subnets with
  // d+1 pins.  The direct optimal cases returned above use d+2 pins instead.
  next_net_sizes[id + 1] = 2 * candidates;
  next_pin_sizes[id + 1] = candidates * (net.degree + 1);
  atomicExch(has_high_degree, 1);
}

__device__ inline int kth_candidate(const Net& net, const int* xs,
                                    const int* ys, const int* s,
                                    const int* score_si, const float* score_penalty,
                                    const float* score_distx, const float* score_disty,
                                    int wanted) {
  const int total = 2 * ((net.degree - 1 - max((net.degree - net.acc) / 5, 2)) -
                         max((net.degree - net.acc) / 5, 2) + 1);
  // FLUTE repeatedly extracts the highest score.  The old direct translation
  // recomputed every candidate's rank against every other one: O(d^2) score
  // evaluations for one subnet.  Keep only the first (wanted + 1) entries;
  // accuracy is small (A=3 here), so this is O(A*d) and preserves FLUTE's
  // score-descending, lower-index-first tie order exactly.
  float top_scores[kMaxAccuracy];
  int top_ids[kMaxAccuracy];
  const int keep = wanted + 1;
  for (int i = 0; i < keep; ++i) {
    top_scores[i] = -3.402823466e+38F;
    top_ids[i] = INT_MAX;
  }
  for (int candidate = 0; candidate < total; ++candidate) {
    const float score = break_score_pre(net, xs, ys, s, score_si, score_penalty,
                                        score_distx, score_disty, candidate);
    int insert = keep;
    for (int i = 0; i < keep; ++i) {
      if (score > top_scores[i] ||
          (score == top_scores[i] && candidate < top_ids[i])) {
        insert = i;
        break;
      }
    }
    if (insert == keep) continue;
    for (int i = keep - 1; i > insert; --i) {
      top_scores[i] = top_scores[i - 1];
      top_ids[i] = top_ids[i - 1];
    }
    top_scores[insert] = score;
    top_ids[insert] = candidate;
  }
  return top_ids[wanted];
}

// Algorithm 3 in the paper for heuristic FLUTE candidates.  Exact FLUTE
// x/y split construction is preserved; only the execution order changes.
__global__ void break_kernel(Net* level, int count, const int* net_offsets,
                             const int* pin_offsets, const int* xs,
                             const int* ys, const int* s, Net* next_level,
                             int* next_xs, int* next_ys, int* next_s,
                             int* score_si, float* score_penalty,
                             float* score_distx, float* score_disty) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count || level[id].degree <= kLutDegree) return;
  Net& parent = level[id];
  int direct_split = 0;
  const int direct_kind = direct_break(parent, s, &direct_split);
  const int selected = direct_kind ? 1 : candidate_count(parent);
  // Heuristic candidates are the only consumers of break scores; the two
  // provably optimal direct breaks never evaluate them.  Subnet pin ranges
  // are disjoint within a level, so the scratch slices race with nobody.
  if (!direct_kind)
    precompute_break_arrays(parent, xs, ys, s, score_si, score_penalty,
                            score_distx, score_disty);
  parent.child_begin = net_offsets[id];
  parent.child_count = selected * 2;
  parent.break_kind = direct_kind ? 1 : 2;
  const int* px = xs + parent.pin_offset;
  const int* py = ys + parent.pin_offset;
  const int* ps = s + parent.pin_offset;
  const int d = parent.degree;
  const int lb = max((d - parent.acc) / 5, 2);

  for (int choice = 0; choice < selected; ++choice) {
    const int candidate = direct_kind ? 0 :
        kth_candidate(parent, xs, ys, s, score_si, score_penalty,
                      score_distx, score_disty, choice);
    const int split = direct_kind ? direct_split : candidate / 2 + lb;
    const bool split_x = direct_kind ? (direct_kind == 1) : (candidate & 1) == 0;
    const int left_id = net_offsets[id] + 2 * choice;
    const int right_id = left_id + 1;
    const int left_offset = pin_offsets[id] + (direct_kind ? 0 : choice * (d + 1));
    const int left_degree = direct_kind == 1 ? split + 2 :
                            direct_kind == 2 ? d + 1 - split : split + 1;
    const int right_degree = direct_kind == 1 ? d - split :
                             direct_kind == 2 ? split + 1 : d - split;
    const int right_offset = left_offset + left_degree;
    const int next_acc = direct_kind ? parent.acc : child_accuracy(parent.acc);
    const int merge_kind = direct_kind ? 1 : (split_x ? 2 : 3);
    next_level[left_id] = {left_degree, left_offset, next_acc,
                           -1, 0, 0, merge_kind, split, 0};
    next_level[right_id] = {right_degree, right_offset, next_acc,
                            -1, 0, 0, merge_kind, split, 0};

    if (direct_kind == 1) {
      for (int i = 0; i <= split; ++i) {
        next_xs[left_offset + i] = px[i];
        next_ys[left_offset + i] = py[i];
        next_s[left_offset + i] = ps[i];
      }
      next_xs[left_offset + split + 1] = px[split];
      next_ys[left_offset + split + 1] = py[split];
      next_s[left_offset + split + 1] = split + 1;
      next_xs[right_offset] = px[split];
      next_ys[right_offset] = py[split];
      next_s[right_offset] = 0;
      for (int i = 1; i < right_degree; ++i) {
        next_xs[right_offset + i] = px[split + i];
        next_ys[right_offset + i] = py[split + i];
        next_s[right_offset + i] = ps[split + i] - split;
      }
      continue;
    }
    if (direct_kind == 2) {
      next_xs[left_offset] = px[split];
      next_ys[left_offset] = py[0];
      next_s[left_offset] = ps[0] - split + 1;
      for (int i = 1; i <= d - 1 - split; ++i) {
        next_xs[left_offset + i] = px[i + split - 1];
        next_ys[left_offset + i] = py[i];
        next_s[left_offset + i] = ps[i] - split + 1;
      }
      next_xs[left_offset + d - split] = px[d - 1];
      next_ys[left_offset + d - split] = py[d - 1 - split];
      next_s[left_offset + d - split] = 0;
      next_s[right_offset] = split;
      for (int i = 0; i < right_degree; ++i) {
        next_xs[right_offset + i] = px[i];
        next_ys[right_offset + i] = py[i + d - 1 - split];
        if (i) next_s[right_offset + i] = ps[i + d - 1 - split];
      }
      continue;
    }

    if (split_x) {
      for (int i = 0; i <= split; ++i) next_xs[left_offset + i] = px[i];
      for (int i = 0; i < right_degree; ++i) next_xs[right_offset + i] = px[split + i];
      int l = 0, r = 0;
      for (int i = 0; i < d; ++i) {
        if (ps[i] < split) {
          next_ys[left_offset + l] = py[i]; next_s[left_offset + l++] = ps[i];
        } else if (ps[i] > split) {
          next_ys[right_offset + r] = py[i]; next_s[right_offset + r++] = ps[i] - split;
        } else {
          next_ys[left_offset + l] = py[i]; next_s[left_offset + l++] = split;
          next_ys[right_offset + r] = py[i]; next_s[right_offset + r++] = 0;
        }
      }
    } else {
      for (int i = 0; i <= split; ++i) next_ys[left_offset + i] = py[i];
      for (int i = 0; i < right_degree; ++i) next_ys[right_offset + i] = py[split + i];
      int l = 0, r = 0;
      for (int x = 0; x < d; ++x) {
        const int y = inverse_s(ps, d, x);
        if (y < split) {
          next_xs[left_offset + l] = px[x]; next_s[left_offset + y] = l++;
        } else if (y > split) {
          next_xs[right_offset + r] = px[x]; next_s[right_offset + y - split] = r++;
        } else {
          next_xs[left_offset + l] = px[x]; next_s[left_offset + split] = l++;
          next_xs[right_offset + r] = px[x]; next_s[right_offset] = r++;
        }
      }
    }
  }
}

__device__ inline void merge_h(const Branch* t1, int d1, const Branch* t2,
                               int d2, const int* s, Branch* out) {
  const int d = d1 + d2 - 1;
  const int offset1 = d2 - 1;
  const int offset2 = 2 * d1 - 3;
  const int split = d1 - 1;
  int n1 = 0, n2 = 0, nn1 = 0, nn2 = 0, common_y = 0;
  for (int i = 0; i < d; ++i) {
    if (s[i] < split) {
      out[i] = {t1[n1].x, t1[n1].y, t1[n1].n + offset1};
      ++n1;
    } else if (s[i] > split) {
      out[i] = {t2[n2].x, t2[n2].y, t2[n2].n + offset2};
      ++n2;
    } else {
      out[i] = {t2[n2].x, t2[n2].y, t2[n2].n + offset2};
      nn1 = n1++; nn2 = n2++; common_y = i;
    }
  }
  for (int i = d; i <= d + d1 - 3; ++i)
    out[i] = {t1[i - offset1].x, t1[i - offset1].y,
              t1[i - offset1].n + offset1};
  for (int i = d + d1 - 2; i <= 2 * d - 4; ++i)
    out[i] = {t2[i - offset2].x, t2[i - offset2].y,
              t2[i - offset2].n + offset2};

  const int extra = 2 * d - 3;
  const int coord1 = t1[t1[nn1].n].y;
  const int coord2 = t2[t2[nn2].n].y;
  int y = t2[nn2].y;
  if (y > max(coord1, coord2)) y = max(coord1, coord2);
  if (y < min(coord1, coord2)) y = min(coord1, coord2);
  out[extra] = {t2[nn2].x, y, out[common_y].n};
  out[common_y].n = extra;

  int prev = extra;
  int cur = t1[nn1].n + offset1;
  int next = out[cur].n;
  while (cur != next) {
    out[cur].n = prev;
    prev = cur;
    cur = next;
    next = out[cur].n;
  }
  out[cur].n = prev;
}

__device__ inline void merge_d(const Branch* t1, int d1, const Branch* t2,
                               int d2, Branch* out) {
  const int d = d1 + d2 - 2;
  const int offset1 = d2 - 2;
  const int offset2 = 2 * d1 - 4;
  for (int i = 0; i <= d1 - 2; ++i)
    out[i] = {t1[i].x, t1[i].y, t1[i].n + offset1};
  for (int i = d1 - 1; i < d; ++i)
    out[i] = {t2[i - d1 + 2].x, t2[i - d1 + 2].y,
              t2[i - d1 + 2].n + offset2};
  for (int i = d; i <= d + d1 - 3; ++i)
    out[i] = {t1[i - offset1].x, t1[i - offset1].y,
              t1[i - offset1].n + offset1};
  for (int i = d + d1 - 2; i <= 2 * d - 3; ++i)
    out[i] = {t2[i - offset2].x, t2[i - offset2].y,
              t2[i - offset2].n + offset2};
  int prev = t2[0].n + offset2;
  int cur = t1[d1 - 1].n + offset1;
  int next = out[cur].n;
  while (cur != next) {
    out[cur].n = prev;
    prev = cur;
    cur = next;
    next = out[cur].n;
  }
  out[cur].n = prev;
}

__device__ inline void merge_v(const Branch* t1, int d1, const Branch* t2,
                               int d2, Branch* out) {
  const int d = d1 + d2 - 1;
  const int offset1 = d2 - 1;
  const int offset2 = 2 * d1 - 3;
  for (int i = 0; i <= d1 - 2; ++i)
    out[i] = {t1[i].x, t1[i].y, t1[i].n + offset1};
  for (int i = d1 - 1; i < d; ++i)
    out[i] = {t2[i - d1 + 1].x, t2[i - d1 + 1].y,
              t2[i - d1 + 1].n + offset2};
  for (int i = d; i <= d + d1 - 3; ++i)
    out[i] = {t1[i - offset1].x, t1[i - offset1].y,
              t1[i - offset1].n + offset1};
  for (int i = d + d1 - 2; i <= 2 * d - 4; ++i)
    out[i] = {t2[i - offset2].x, t2[i - offset2].y,
              t2[i - offset2].n + offset2};

  const int extra = 2 * d - 3;
  const int coord1 = t1[t1[d1 - 1].n].x;
  const int coord2 = t2[t2[0].n].x;
  int x = t2[0].x;
  if (x > max(coord1, coord2)) x = max(coord1, coord2);
  if (x < min(coord1, coord2)) x = min(coord1, coord2);
  out[extra] = {x, t2[0].y, out[d1 - 1].n};
  out[d1 - 1].n = extra;

  int prev = extra;
  int cur = t1[d1 - 1].n + offset1;
  int next = out[cur].n;
  while (cur != next) {
    out[cur].n = prev;
    prev = cur;
    cur = next;
    next = out[cur].n;
  }
  out[cur].n = prev;
}

// Reverse level pass (Algorithm 4).  Scratch space is one fixed-size output
// per FLUTE candidate, allowing all candidate trees to be evaluated before
// copying only the shortest one into the parent-level tree buffer.
__global__ void merge_level_kernel(Net* level, int count, const int* xs,
                                   const int* s, const Net* next_level,
                                   const int* next_tree_offsets,
                                   const Branch* next_trees,
                                   const int* candidate_offsets,
                                   Branch* candidate_trees,
                                   const int* tree_offsets, Branch* trees) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count || level[id].degree <= kLutDegree) return;
  Net& parent = level[id];
  const int choices = parent.child_count / 2;
  const int capacity = 2 * parent.degree - 2;
  int best_choice = 0;
  int best_length = INT_MAX;
  for (int choice = 0; choice < choices; ++choice) {
    const int left = parent.child_begin + 2 * choice;
    const int right = left + 1;
    const Net& left_net = next_level[left];
    const Net& right_net = next_level[right];
    Branch* out = candidate_trees + candidate_offsets[id] + choice * capacity;
    const Branch* left_tree = next_trees + next_tree_offsets[left];
    const Branch* right_tree = next_trees + next_tree_offsets[right];
    if (left_net.merge_kind == 1)
      merge_d(left_tree, left_net.degree, right_tree, right_net.degree, out);
    else if (left_net.merge_kind == 2)
      merge_h(left_tree, left_net.degree, right_tree, right_net.degree,
              s + parent.pin_offset, out);
    else
      merge_v(left_tree, left_net.degree, right_tree, right_net.degree, out);
    const int length = tree_wirelength(out, parent.degree);
    if (length < best_length) {
      best_length = length;
      best_choice = choice;
    }
  }
  const Branch* best = candidate_trees + candidate_offsets[id] +
                       best_choice * capacity;
  Branch* target = trees + tree_offsets[id];
  for (int i = 0; i < capacity; ++i) target[i] = best[i];
  parent.tree_length = best_length;
}

__global__ void tree_size_kernel(const Net* nets, int count, int* sizes) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) sizes[id + 1] = 2 * nets[id].degree - 2;
}

__global__ void candidate_size_kernel(const Net* nets, int count, int* sizes) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) {
    const Net& net = nets[id];
    sizes[id + 1] = net.degree <= kLutDegree ? 0 :
        (net.child_count / 2) * (2 * net.degree - 2);
  }
}

inline int inclusive_scan_sizes(int* device_sizes, int count, Profile* profile = nullptr) {
  check(cudaMemset(device_sizes, 0, sizeof(int)), "initialize scan origin");
  thrust::device_ptr<int> begin(device_sizes + 1);
  thrust::inclusive_scan(thrust::device, begin, begin + count, begin);
  int total = 0;
  profiled_memcpy(&total, device_sizes + count, sizeof(int), cudaMemcpyDeviceToHost,
                  "read scan total", profile);
  return total;
}

inline int allocate_tree_storage(Level* level, Profile* profile = nullptr) {
  check(cudaMalloc(&level->tree_offsets, sizeof(int) * (level->net_count + 1)),
        "allocate tree offsets");
  check(cudaMemset(level->tree_offsets, 0, sizeof(int) * (level->net_count + 1)),
        "initialize tree offsets");
  tree_size_kernel<<<blocks(level->net_count), kBlockSize>>>(
      level->nets, level->net_count, level->tree_offsets);
  check(cudaGetLastError(), "compute tree sizes");
  const int branch_count = inclusive_scan_sizes(level->tree_offsets, level->net_count, profile);
  check(cudaMalloc(&level->trees, sizeof(Branch) * branch_count),
        "allocate tree buffer");
  return branch_count;
}

struct Result {
  std::vector<int> net_ids;
  std::vector<int> tree_offsets;
  std::vector<Branch> trees;
  // Populated only when the reversible GPU tree-center option is enabled.
  std::vector<TreeCenter> tree_centers;
  // Optional root Hanan snapshot used only by the host validation mode.
  std::vector<int> root_xs;
  std::vector<int> root_ys;
  std::vector<int> root_s;
  Profile profile;
};

// Runs the GPU-FLUTE path for the supplied high-degree router nets.  Root nets
// of degree <= kLutDegree deliberately remain outside this API and continue to
// use the original CPU FLUTE call site.
inline Result solve_high_degree(const std::vector<int>& host_net_ids,
                                const std::vector<int>& host_pin_acc_num,
                                const int* pin_acc_num, const int* pins,
                                const int* x_coords, const int* y_coords,
                                int grid_x, int grid_y, int accuracy = 3,
                                bool capture_root_hanan = false,
                                bool collect_profile = false,
                                bool compute_tree_centers = false) {
  Result result;
  Profile* profile = collect_profile ? &result.profile : nullptr;
  const auto host_start = std::chrono::steady_clock::now();
  result.net_ids = host_net_ids;
  if (host_net_ids.empty()) return result;

  int* root_ids = nullptr;
  check(cudaMalloc(&root_ids, sizeof(int) * host_net_ids.size()),
        "allocate root ids");
  profiled_memcpy(root_ids, host_net_ids.data(), sizeof(int) * host_net_ids.size(),
                  cudaMemcpyHostToDevice, "upload root ids", profile);

  std::vector<Level> levels;
  levels.emplace_back();
  Level& root = levels.back();
  root.net_count = static_cast<int>(host_net_ids.size());
  // Root pin offsets intentionally match InstantGR's existing CSR offsets;
  // this avoids an extra input compaction before the first GPU kernel.
  root.pin_count = host_pin_acc_num.back();
  check(cudaMalloc(&root.nets, sizeof(Net) * root.net_count), "allocate root metadata");
  check(cudaMalloc(&root.xs, sizeof(int) * root.pin_count), "allocate root xs");
  check(cudaMalloc(&root.ys, sizeof(int) * root.pin_count), "allocate root ys");
  check(cudaMalloc(&root.s, sizeof(int) * root.pin_count), "allocate root s");
  cudaEvent_t device_start = nullptr;
  cudaEvent_t device_end = nullptr;
  if (profile != nullptr) {
    check(cudaEventCreate(&device_start), "create profile start event");
    check(cudaEventCreate(&device_end), "create profile end event");
    check(cudaEventRecord(device_start), "record profile start event");
  }
  make_hanan_kernel<<<blocks(root.net_count), kBlockSize>>>(
      root_ids, root.net_count, pin_acc_num, pins, x_coords, y_coords, grid_x, grid_y,
      root.nets, root.xs, root.ys, root.s, accuracy);
  check(cudaGetLastError(), "construct root Hanan grids");
  thrust_sort_large_root_hanan(host_net_ids, host_pin_acc_num,
                               root.xs, root.ys, root.s);
  allocate_tree_storage(&root, profile);

  while (true) {
    Level& current = levels.back();
    int* net_offsets = nullptr;
    int* pin_offsets = nullptr;
    int* high = nullptr;
    check(cudaMalloc(&net_offsets, sizeof(int) * (current.net_count + 1)),
          "allocate subnet offsets");
    check(cudaMalloc(&pin_offsets, sizeof(int) * (current.net_count + 1)),
          "allocate subnet pin offsets");
    check(cudaMemset(net_offsets, 0, sizeof(int) * (current.net_count + 1)),
          "initialize subnet offsets");
    check(cudaMemset(pin_offsets, 0, sizeof(int) * (current.net_count + 1)),
          "initialize subnet pin offsets");
    check(cudaMalloc(&high, sizeof(int)), "allocate high-degree flag");
    check(cudaMemset(high, 0, sizeof(int)), "initialize high-degree flag");
    estimate_break_kernel<<<blocks(current.net_count), kBlockSize>>>(
        current.nets, current.net_count, current.xs, current.ys, current.s,
        net_offsets, pin_offsets, high);
    check(cudaGetLastError(), "estimate next GPU-FLUTE level");
    int has_high = 0;
    profiled_memcpy(&has_high, high, sizeof(int), cudaMemcpyDeviceToHost,
                    "read high-degree flag", profile);
    if (!has_high) {
      cudaFree(net_offsets); cudaFree(pin_offsets); cudaFree(high);
      break;
    }
    const int next_nets = inclusive_scan_sizes(net_offsets, current.net_count, profile);
    const int next_pins = inclusive_scan_sizes(pin_offsets, current.net_count, profile);
    cudaFree(high);
    Level next;
    next.net_count = next_nets;
    next.pin_count = next_pins;
    check(cudaMalloc(&next.nets, sizeof(Net) * next.net_count), "allocate subnet metadata");
    check(cudaMalloc(&next.xs, sizeof(int) * next.pin_count), "allocate subnet xs");
    check(cudaMalloc(&next.ys, sizeof(int) * next.pin_count), "allocate subnet ys");
    check(cudaMalloc(&next.s, sizeof(int) * next.pin_count), "allocate subnet s");
    // Score scratch lives only across this level's break: the merge pass
    // reads trees and s, never scores.  Sliced by the same pin offsets as
    // xs/ys/s, so one allocation of the level's pin count covers every net.
    int* score_si = nullptr;
    float* score_penalty = nullptr;
    float* score_distx = nullptr;
    float* score_disty = nullptr;
    check(cudaMalloc(&score_si, sizeof(int) * current.pin_count),
          "allocate score si scratch");
    check(cudaMalloc(&score_penalty, sizeof(float) * current.pin_count),
          "allocate score penalty scratch");
    check(cudaMalloc(&score_distx, sizeof(float) * current.pin_count),
          "allocate score distx scratch");
    check(cudaMalloc(&score_disty, sizeof(float) * current.pin_count),
          "allocate score disty scratch");
    break_kernel<<<blocks(current.net_count), kBlockSize>>>(
        current.nets, current.net_count, net_offsets, pin_offsets, current.xs,
        current.ys, current.s, next.nets, next.xs, next.ys, next.s,
        score_si, score_penalty, score_distx, score_disty);
    check(cudaGetLastError(), "construct next GPU-FLUTE level");
    cudaFree(net_offsets);
    cudaFree(pin_offsets);
    cudaFree(score_si);
    cudaFree(score_penalty);
    cudaFree(score_distx);
    cudaFree(score_disty);
    allocate_tree_storage(&next, profile);
    levels.push_back(next);
  }

  const Lut& lut = resident_lut(profile);
  for (int level_id = static_cast<int>(levels.size()) - 1; level_id >= 0; --level_id) {
    Level& level = levels[level_id];
    solve_leaf_kernel<<<blocks(level.net_count), kBlockSize>>>(
        level.nets, level.net_count, level.xs, level.ys, level.s,
        lut.solutions, lut.offsets, lut.counts, level.tree_offsets, level.trees);
    check(cudaGetLastError(), "solve GPU-FLUTE LUT leaves");
    if (level_id == static_cast<int>(levels.size()) - 1) continue;

    int* candidate_offsets = nullptr;
    check(cudaMalloc(&candidate_offsets, sizeof(int) * (level.net_count + 1)),
          "allocate candidate offsets");
    check(cudaMemset(candidate_offsets, 0, sizeof(int) * (level.net_count + 1)),
          "initialize candidate offsets");
    candidate_size_kernel<<<blocks(level.net_count), kBlockSize>>>(
        level.nets, level.net_count, candidate_offsets);
    check(cudaGetLastError(), "compute candidate tree sizes");
    const int candidate_branches = inclusive_scan_sizes(candidate_offsets, level.net_count, profile);
    Branch* candidate_trees = nullptr;
    check(cudaMalloc(&candidate_trees, sizeof(Branch) * candidate_branches),
          "allocate candidate trees");
    Level& next = levels[level_id + 1];
    merge_level_kernel<<<blocks(level.net_count), kBlockSize>>>(
        level.nets, level.net_count, level.xs, level.s, next.nets,
        next.tree_offsets, next.trees, candidate_offsets, candidate_trees,
        level.tree_offsets, level.trees);
    check(cudaGetLastError(), "merge GPU-FLUTE level");
    cudaFree(candidate_trees);
    cudaFree(candidate_offsets);
  }

  Level& root_result = levels.front();
  if (profile != nullptr) {
    check(cudaEventRecord(device_end), "record profile end event");
    check(cudaEventSynchronize(device_end), "synchronize profile end event");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, device_start, device_end),
          "read profile event time");
    profile->device_pipeline_seconds = milliseconds / 1000.0;
    cudaEventDestroy(device_start);
    cudaEventDestroy(device_end);
  }
  result.tree_offsets.resize(root_result.net_count + 1);
  profiled_memcpy(result.tree_offsets.data(), root_result.tree_offsets,
                  sizeof(int) * result.tree_offsets.size(), cudaMemcpyDeviceToHost,
                  "download root tree offsets", profile);
  if(compute_tree_centers) {
    const int branch_count = result.tree_offsets.back();
    TreeCenterWorkspace workspace;
    workspace.allocate(branch_count);
    TreeCenter* device_centers = nullptr;
    check(cudaMalloc(&device_centers, sizeof(TreeCenter) * root_result.net_count),
          "allocate tree centers");
    cudaEvent_t center_start = nullptr;
    cudaEvent_t center_end = nullptr;
    if(profile != nullptr) {
      check(cudaEventCreate(&center_start), "create tree-center profile start");
      check(cudaEventCreate(&center_end), "create tree-center profile end");
      check(cudaEventRecord(center_start), "record tree-center profile start");
    }
    tree_center_kernel<<<root_result.net_count, kBlockSize>>>(
        root_result.trees, root_result.tree_offsets, root_result.net_count,
        workspace.hash, workspace.canonical, workspace.degree, workspace.offset,
        workspace.write, workspace.alive, workspace.queue_a, workspace.queue_b,
        workspace.adjacency, device_centers);
    check(cudaGetLastError(), "compute GPU tree centers");
    if(profile != nullptr) {
      check(cudaEventRecord(center_end), "record tree-center profile end");
      check(cudaEventSynchronize(center_end), "synchronize tree-center profile end");
      float milliseconds = 0.0f;
      check(cudaEventElapsedTime(&milliseconds, center_start, center_end),
            "read tree-center profile event time");
      profile->tree_center_device_seconds = milliseconds / 1000.0;
      cudaEventDestroy(center_start);
      cudaEventDestroy(center_end);
    }
    result.tree_centers.resize(root_result.net_count);
    profiled_memcpy(result.tree_centers.data(), device_centers,
                    sizeof(TreeCenter) * result.tree_centers.size(), cudaMemcpyDeviceToHost,
                    "download GPU tree centers", profile);
    cudaFree(device_centers);
    workspace.release();
  }
  result.trees.resize(result.tree_offsets.back());
  profiled_memcpy(result.trees.data(), root_result.trees,
                  sizeof(Branch) * result.trees.size(), cudaMemcpyDeviceToHost,
                  "download GPU-FLUTE trees", profile);
  if (capture_root_hanan) {
    result.root_xs.resize(root_result.pin_count);
    result.root_ys.resize(root_result.pin_count);
    result.root_s.resize(root_result.pin_count);
    profiled_memcpy(result.root_xs.data(), root_result.xs,
                    sizeof(int) * result.root_xs.size(), cudaMemcpyDeviceToHost,
                    "download root Hanan xs", profile);
    profiled_memcpy(result.root_ys.data(), root_result.ys,
                    sizeof(int) * result.root_ys.size(), cudaMemcpyDeviceToHost,
                    "download root Hanan ys", profile);
    profiled_memcpy(result.root_s.data(), root_result.s,
                    sizeof(int) * result.root_s.size(), cudaMemcpyDeviceToHost,
                    "download root Hanan permutation", profile);
  }
  for (Level& level : levels) level.release();
  cudaFree(root_ids);
  if (profile != nullptr)
    profile->host_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - host_start).count();
  return result;
}

}  // namespace gpu_flute
