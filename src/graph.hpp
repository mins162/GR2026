
#pragma once
#include "global.h"
#include <fcntl.h>
#include <unistd.h>
#include "database_cuda.hpp"

namespace graph {

//declaration
void update_cost();
void output(char file_name[]);
__global__ void compute_presum();

//implementation

__device__ double wire_segment_cost(int layer, int xmin, int xmax, int ymin, int ymax) {
    return presum[IDX(layer, xmax, ymax)] - presum[IDX(layer, xmin, ymin)];
}

__device__ double of_cost_scaled(float capacity, float demand) {
    if(capacity > 0.001) return __expf(min(0.5 * (demand - capacity), of_cost_scale * 0.5 * (demand - capacity)));
    if(demand > 0) return __expf(min(1.5 * demand, of_cost_scale * 1.5 * demand));
    return 0;
}
__device__ double of_cost(float capacity, float demand) {
    if(capacity > 0.001) return __expf(0.5 * (demand - capacity));
    if(demand > 0) return __expf(1.5 * demand);
    return 0;
}

__device__ double incremental_of_cost(int l, int x, int y, double incre_demand) {
    int idx = l * X * Y + x * Y + y;
    return min(1e12, (of_cost_scaled(capacity[idx], demand[idx] + incre_demand) - of_cost_scaled(capacity[idx], demand[idx])) * unit_length_short_costs[l]);
}

#ifdef INSTANTGR_LEGACY_PRESUM
// A/B scaffold: build with -DINSTANTGR_LEGACY_PRESUM to get the pre-fusion
// behavior for timing comparisons.  Delete this block once the fusion is
// accepted.
__global__ void update_wcost_cuda_ispd24() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y, dir = l & 1 ^ DIR;
    wcost[idx] = incremental_of_cost(l, x, y, 1);
    if(dir == 0 && x < X - 1) wcost[idx] += unit_length_wire_cost * x_edge_len[x];
    if(dir == 1 && y < Y - 1) wcost[idx] += unit_length_wire_cost * y_edge_len[y];
}
#endif

// Wire cost of one grid edge.  This used to be materialized into a wcost[]
// array by its own full-grid kernel, but the only reader was compute_presum()
// on the very next line of every caller, so the array is now skipped and the
// cost is computed inside the scan.  Keep the float return type: the old code
// stored into a float array, and rounding there decides ties in the DP.
__device__ float wire_cost(int l, int x, int y, int dir) {
    float cost = incremental_of_cost(l, x, y, 1);
    if(dir == 0 && x < X - 1) cost += unit_length_wire_cost * x_edge_len[x];
    if(dir == 1 && y < Y - 1) cost += unit_length_wire_cost * y_edge_len[y];
    return cost;
}



// Reserve n consecutive slots in the dirty list.  Returns -1 when the list is
// full; the caller then skips recording and update_cost() falls back to a full
// rebuild, so a full list costs performance but never correctness.
__device__ int reserve_dirty_cells(int n) {
    if(!incremental_vcost_on || dirty_overflow) return -1;
    const int pos = atomicAdd(&dirty_cell_cnt, n);
    // Two limits, both handled as overflow: the list is physically full, or the
    // batch touches so much of the grid that rebuilding everything is cheaper
    // than chasing a scattered list (each entry also refreshes a neighbor).
    if(pos + n > dirty_cell_limit) {
        dirty_overflow = true;
        return -1;
    }
    return pos;
}

__device__ void mark_dirty_track(int l, int x, int y) {
    if(!incremental_presum_on) return;
    track_dirty[l * XY + ((l & 1 ^ DIR) ? x : y)] = true;
}

__device__ void mark_dirty_cell(int idx) {
    const int pos = reserve_dirty_cells(1);
    if(pos >= 0) dirty_cells[pos] = idx;
    const int l = idx / X / Y;
    mark_dirty_track(l, idx / Y % X, idx % Y);
}

__global__ void reset_dirty_state() {
    const int stride = gridDim.x * blockDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < L * XY; i += stride)
        track_dirty[i] = false;
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        dirty_cell_cnt = 0;
        dirty_overflow = false;
    }
}

__device__ const double via_weight = 1;
__device__ void update_single_vcost_ispd24(int l, int x, int y, float *vcost) {
    int idx = IDX(l, x, y);
    if(l & 1 ^ DIR) {
        if(y == 0) 
            vcost[idx] = incremental_of_cost(l, x, y, 1);
        else if(y == Y - 1)
            vcost[idx] = incremental_of_cost(l, x, y - 1, 1);
        else
            vcost[idx] = incremental_of_cost(l, x, y, 0.5) + incremental_of_cost(l, x, y - 1, 0.5);
    } else {
        if(x == 0)
            vcost[idx] = incremental_of_cost(l, x, y, 1);
        else if(x == X - 1)
            vcost[idx] = incremental_of_cost(l, x - 1, y, 1);
        else
            vcost[idx] = incremental_of_cost(l, x, y, 0.5) + incremental_of_cost(l, x - 1, y, 0.5);
    }
    vcost[idx] += unit_via_cost;
}


__global__ void compute_presum_general(int *to_sum) {
    extern __shared__ int sum2[];
    if(threadIdx.x == 0) sum2[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum2[y + 1] = to_sum[IDX(l, x, y)];
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum2[idx] += sum2[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) to_sum[IDX(l, x, y)] = sum2[y];
    } else {
        int y = idx2track[blockIdx.x] % XY;
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum2[x + 1] = to_sum[IDX(l, x, y)];
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum2[idx] += sum2[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) to_sum[IDX(l, x, y)] = sum2[x];
    }
}

__global__ void update_vcost_ispd24() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < L * X * Y) update_single_vcost_ispd24(idx / X / Y, idx / Y % X, idx % Y, vcost);
}

// One grid-stride kernel that picks its own work: the dirty list when it is
// intact, the whole grid otherwise.  Deciding on the device keeps update_cost()
// free of a host synchronization, which the per-batch pipeline cannot afford.
const int VCOST_GRID_BLOCKS = 8192;

__global__ void update_vcost_full(float *out) {
    const int stride = gridDim.x * blockDim.x;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < L * X * Y; idx += stride)
        update_single_vcost_ispd24(idx / X / Y, idx / Y % X, idx % Y, out);
}

__global__ void update_vcost_selective(float *out) {
    const int stride = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(dirty_overflow || !incremental_vcost_on) {
        for(int idx = tid; idx < L * X * Y; idx += stride)
            update_single_vcost_ispd24(idx / X / Y, idx / Y % X, idx % Y, out);
        return;
    }
    const int cnt = min(dirty_cell_cnt, dirty_cell_limit);
    for(int i = tid; i < cnt; i += stride) {
        const int idx = dirty_cells[i];
        const int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
        // demand[l][x][y] feeds the via cost of this cell and of the next cell
        // along the layer's routing direction.
        update_single_vcost_ispd24(l, x, y, out);
        if(l & 1 ^ DIR) {
            if(y + 1 < Y) update_single_vcost_ispd24(l, x, y + 1, out);
        } else {
            if(x + 1 < X) update_single_vcost_ispd24(l, x + 1, y, out);
        }
    }
}
__device__ void atomic_add_unit_demand_wire_segment(int l, int minx, int maxx, int miny, int maxy, int stamp, int K = 1) {
    assert(minx == maxx || miny == maxy);

    atomicAdd(pre_demand + IDX(l, minx, miny), K);
    if(minx == maxx) {
        atomicAdd(pre_demand + IDX(l, minx, maxy), -K);
        timestamp[IDX(l, minx, maxy)] = stamp;
    }
    if(miny == maxy) {
        atomicAdd(pre_demand + IDX(l, maxx, miny), -K);
        timestamp[IDX(l, maxx, miny)] = stamp;
    }
}
__global__ void commit_all_edge(int stamp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if((l & 1 ^ DIR) == 0 && x + 1 < X && pre_demand[IDX(l, x + 1, y)] > 0) {
        demand[IDX(l, x, y)] += pre_demand[IDX(l, x + 1, y)];
        mark_dirty_cell(IDX(l, x, y));
        timestamp[IDX(l, x, y)] = stamp;
        atomicAdd(&total_wirelength, pre_demand[IDX(l, x + 1, y)] * x_edge_len[x]);
    }
    if((l & 1 ^ DIR) == 1 && y + 1 < Y && pre_demand[IDX(l, x, y + 1)] > 0) {
        demand[IDX(l, x, y)] += pre_demand[IDX(l, x, y + 1)];
        mark_dirty_cell(IDX(l, x, y));
        timestamp[IDX(l, x, y)] = stamp;
        atomicAdd(&total_wirelength, pre_demand[IDX(l, x, y + 1)] * y_edge_len[y]);
    }
}
void batch_wire_update(int stamp) {
    compute_presum_general<<<all_track_cnt, THREAD_NUM, sizeof(int) * XY>>> (pre_demand);
    commit_all_edge<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> (stamp);
    cudaMemset(pre_demand, 0, sizeof(int) * L * X * Y);
}


// Mirrors the device-side flag so the dispatch below needs no synchronization.
bool incremental_vcost_host() {
    static const bool on = [] {
        const char *value = getenv("INSTANTGR_INCREMENTAL_VCOST");
        return value == nullptr || string(value) != "0";
    }();
    return on;
}

bool incremental_vcost_validate() {
    static const bool on = [] {
        const char *value = getenv("INSTANTGR_INCREMENTAL_VCOST_VALIDATE");
        return value != nullptr && string(value) == "1";
    }();
    return on;
}

__managed__ unsigned long long vcost_mismatch_cnt = 0;
long long vcost_validation_passes = 0;

__global__ void compare_vcost(const float *reference) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    if(vcost[idx] != reference[idx]) atomicAdd(&vcost_mismatch_cnt, 1ULL);
}

void report_vcost_validation(const char *stage) {
    if(!incremental_vcost_validate()) return;
    cudaDeviceSynchronize();
    printf("%s incremental vcost validation: %llu mismatching cells over %lld full-grid checks (%lld cells each)\n",
           stage, vcost_mismatch_cnt, vcost_validation_passes, (long long) L * X * Y);
}

void update_cost_ispd24() {
    // Wire cost is no longer materialized here; compute_presum() derives it per
    // track through wire_cost().  Via cost stays, since the DP reads vcost[].
#ifdef INSTANTGR_LEGACY_PRESUM
    update_wcost_cuda_ispd24<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
#endif
    if(!incremental_vcost_host()) {
        update_vcost_ispd24<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
        return;
    }
    update_vcost_selective<<<VCOST_GRID_BLOCKS, THREAD_NUM>>> (vcost);
    if(incremental_vcost_validate()) {
        // Rebuild the whole grid into a shadow buffer and compare.  This makes
        // the run far slower; it only answers whether the dirty list is complete.
        static float *reference = [] {
            float *buffer = nullptr;
            cudaMalloc(&buffer, (size_t) L * X * Y * sizeof(float));
            return buffer;
        }();
        update_vcost_full<<<VCOST_GRID_BLOCKS, THREAD_NUM>>> (reference);
        compare_vcost<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> (reference);
        vcost_validation_passes++;
    }
    // The dirty state is NOT reset here: compute_presum() consumes the same
    // epoch.  Every caller must invoke finish_cost_refresh() after the presum
    // launch to validate (optionally) and reset for the next batch.
}
bool incremental_presum_validate() {
    static const bool on = [] {
        const char *value = getenv("INSTANTGR_INCREMENTAL_PRESUM_VALIDATE");
        return value != nullptr && string(value) == "1";
    }();
    return on;
}

__managed__ unsigned long long presum_mismatch_cnt = 0;
long long presum_validation_passes = 0;

__global__ void compute_presum_reference(double *out) {
    extern __shared__ double sum[];
    if(threadIdx.x == 0) sum[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum[y + 1] = wire_cost(l, x, y, 1);
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) out[l * X * Y + x * Y + y] = sum[y];
    } else {
        int y = idx2track[blockIdx.x] % XY;
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum[x + 1] = wire_cost(l, x, y, 0);
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) out[l * X * Y + x * Y + y] = sum[x];
    }
}

__global__ void compare_presum(const double *reference) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    if(presum[idx] != reference[idx]) atomicAdd(&presum_mismatch_cnt, 1ULL);
}

void report_presum_validation(const char *stage) {
    if(!incremental_presum_validate()) return;
    cudaDeviceSynchronize();
    printf("%s incremental presum validation: %llu mismatching cells over %lld full-grid checks (%lld cells each)\n",
           stage, presum_mismatch_cnt, presum_validation_passes, (long long) L * X * Y);
}

// Must run once after every update_cost() + compute_presum() pair: it closes
// the dirty epoch that both consumers read.  Resetting any earlier would leave
// presum scanning with cleared flags.
void finish_cost_refresh() {
    if(incremental_presum_validate()) {
        static double *reference = [] {
            double *buffer = nullptr;
            cudaMalloc(&buffer, (size_t) L * X * Y * sizeof(double));
            return buffer;
        }();
        compute_presum_reference<<<all_track_cnt, THREAD_NUM, sizeof(double) * XY>>> (reference);
        compare_presum<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> (reference);
        presum_validation_passes++;
    }
    reset_dirty_state<<<64, THREAD_NUM>>> ();
}

void update_cost() {
    update_cost_ispd24();
}

__global__ void add_all_overflow_cost() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if(((l & 1 ^ DIR) == 0 && x + 1 < X) || ((l & 1 ^ DIR) == 1 && y + 1 < Y)) {
        double c = unit_length_short_costs[l] * of_cost(capacity[idx], demand[idx]);
        atomicAdd(&total_overflow_cost, c);
        atomicAdd(&layer_overflow_cost[l], c);
    }
}

void report_score() {
    printf("\n----------------------------------------------------------------------\n");
    total_overflow_cost = 0;
    for(int i = 0; i < L; i++) layer_overflow_cost[i] = 0;
    add_all_overflow_cost<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
    cudaDeviceSynchronize();
    printf("             WL            Via                  OF                                             Total\n%15.0f%15.0f%20.0f%50.0f\n", 
        total_wirelength * db::unit_length_wire_cost, total_via_count * db::unit_via_cost, total_overflow_cost, 
        total_wirelength * db::unit_length_wire_cost + total_via_count * db::unit_via_cost + total_overflow_cost);
    printf("----------------------------------------------------------------------\n\n");
    report_vcost_validation("cumulative");
    report_presum_validation("cumulative");
}

void finish_nets(vector<int> &finished_nets) {
    for(auto _net_id : finished_nets) {
        int _ = nets[_net_id].original_net_id;
        if(--db::nets[_].unfinished_subnet_count == 0) nets2output.push(_);

    }
}

void output_nets() {
    auto write_int = [&] (int num) {
        if(num == 0) {
            putc_unlocked('0', out_file);
        } else {
            static char temp[20];
            int len = 0;
            while(num) temp[len++] = num % 10, num /= 10;
            for(int i = len - 1; i >= 0; i--) putc_unlocked('0' + temp[i], out_file);
        }
    };
    auto write_single_route = [&] (int x0, int y0, int l0, int x1, int y1, int l1) {
        write_int(x0);
        putc_unlocked(' ', out_file);
        write_int(y0);
        putc_unlocked(' ', out_file);
        write_int(l0);
        putc_unlocked(' ', out_file);
        write_int(x1);
        putc_unlocked(' ', out_file);
        write_int(y1);
        putc_unlocked(' ', out_file);
        write_int(l1);
        putc_unlocked('\n', out_file);
    };
    auto write_route = [&] (int pos0, int pos1) {
        if(pos0 > pos1) swap(pos0, pos1);
        write_single_route(pos0 / Y % X, pos0 % Y, pos0 / X / Y, pos1 / Y % X, pos1 % Y, pos1 / X / Y);
    };

    while(!nets2output.empty()) {
        int _ = nets2output.front();
        nets2output.pop();
        for(auto e : db::nets[_].name) putc_unlocked(e, out_file);
        putc_unlocked('\n', out_file);
        putc_unlocked('(', out_file);
        putc_unlocked('\n', out_file);
        for(auto pin : db::nets[_].pins) if(pin < X * Y) write_route(pin, pin + X * Y);
        for(int i = 0; i < db::nets[_].extra_routes.size(); i += 2) 
            write_route(db::nets[_].extra_routes[i], db::nets[_].extra_routes[i + 1]);
        for(auto net_id : db::nets[_].subnets) {
            int *net_routes = routes + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            for(int j = 1; j < net_routes[0]; j += 2) write_route(net_routes[j] + X * Y, net_routes[j + 1] + X * Y);
        }
        putc_unlocked(')', out_file);
        putc_unlocked('\n', out_file);
    }
}

void output(char file_name[]) {
    double output_start_time = elapsed_time();
    FILE *file = fopen(file_name, "w");
    int char_cur = 0;
    const int CHAR_COUNT = 1000000000;
    static char out[CHAR_COUNT];

    int write_count = 0;
    vector<int> is_pin(X * Y, -1);
    static int *routes_cpu = new int[PIN_NUM * ROUTE_PER_PIN];
    cudaMemcpy(routes_cpu, routes, sizeof(int) * PIN_NUM * ROUTE_PER_PIN, cudaMemcpyDeviceToHost);

    auto write_int = [&] (int num) {
        if(num == 0)
            out[char_cur++] = '0';
        else {
            static char temp[20];
            int len = 0;
            while(num) temp[len++] = num % 10, num /= 10;
            for(int i = len - 1; i >= 0; i--) out[char_cur++] = temp[i] + '0';
        }
    };
    auto write_single_route = [&] (int x0, int y0, int l0, int x1, int y1, int l1) {
        write_int(x0);
        out[char_cur++] = ' ';
        write_int(y0);
        out[char_cur++] = ' ';
        write_int(l0);
        out[char_cur++] = ' ';
        write_int(x1);
        out[char_cur++] = ' ';
        write_int(y1);
        out[char_cur++] = ' ';
        write_int(l1);
        out[char_cur++] = '\n';
    };
    auto write_route = [&] (int pos0, int pos1) {
        if(pos0 > pos1) swap(pos0, pos1);
        write_single_route(pos0 / Y % X, pos0 % Y, pos0 / X / Y, pos1 / Y % X, pos1 % Y, pos1 / X / Y);
    };

    for(int _ = 0; _ < db::nets.size(); _++) {
        if(char_cur * 1.1 > CHAR_COUNT) {
            fwrite(out, sizeof(char), char_cur, file), char_cur = 0;
            write_count++;
        }
        for(auto e : db::nets[_].name) out[char_cur++] = e;
        out[char_cur++] = '\n';
        out[char_cur++] = '(';
        out[char_cur++] = '\n';
        for(auto pin : db::nets[_].pins) if(pin < X * Y) is_pin[pin] = _;
        for(int i = 0; i < db::nets[_].extra_routes.size(); i += 2) 
            write_route(db::nets[_].extra_routes[i], db::nets[_].extra_routes[i + 1]);
        for(auto net_id : db::nets[_].subnets) {
            auto &net = nets[net_id];
            int *net_routes = routes_cpu + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            for(int j = 1; j < net_routes[0]; j += 2) {
                int pos0 = min(net_routes[j], net_routes[j + 1]), pos1 = max(net_routes[j], net_routes[j + 1]);
                int l0 = pos0 / X / Y + 1, x0 = pos0 / Y % X, y0 = pos0 % Y;
                int l1 = pos1 / X / Y + 1, x1 = pos1 / Y % X, y1 = pos1 % Y;
                if(l0 == 1 && l0 < l1 && is_pin[x0 * Y + y0] == _) l0 = 0, is_pin[x0 * Y + y0] = -1;
                write_single_route(x0, y0, l0, x1, y1, l1);
            }
        }
        for(auto pin : db::nets[_].pins) 
            if(pin < X * Y && is_pin[pin] == _) write_route(pin, pin + X * Y);
        out[char_cur++] = ')';
        out[char_cur++] = '\n';
    }
    
    double fwrite_start_time = elapsed_time();
    fwrite(out, sizeof(char), char_cur, file), char_cur = 0;
    //write(file, out, char_cur);
    fclose(file);
    printf("    write calls: %d\n", ++write_count);
    printf("    fwrite time: %.2f\n", elapsed_time() - fwrite_start_time);
    output_time = elapsed_time() - output_start_time;
}

__global__ void compute_presum() {
    extern __shared__ double sum[];
    if(incremental_presum_on && !dirty_overflow && !track_dirty[idx2track[blockIdx.x]])
        return;// demand unchanged on this track since its last scan
    if(threadIdx.x == 0) sum[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
#ifdef INSTANTGR_LEGACY_PRESUM
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum[y + 1] = wcost[l * X * Y + x * Y + y];
#else
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum[y + 1] = wire_cost(l, x, y, 1);
#endif
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) presum[l * X * Y + x * Y + y] = sum[y];
    } else {
        int y = idx2track[blockIdx.x] % XY;
#ifdef INSTANTGR_LEGACY_PRESUM
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum[x + 1] = wcost[l * X * Y + x * Y + y];
#else
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum[x + 1] = wire_cost(l, x, y, 0);
#endif
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) presum[l * X * Y + x * Y + y] = sum[x];
    }
}

__global__ void mark_overflow_edges(int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if(((l & 1 ^ DIR) == 0 && x + 1 < X) || ((l & 1 ^ DIR) == 1 && y + 1 < Y))
        of_edge_sum[idx] = (capacity[idx] + threshold <= demand[idx] ? 1 : 0);
}

__global__ void mark_overflow_nets() {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= NET_NUM) return;
    is_of_net[net_id] = false;
    int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN, of_count = 0, pin_cnt = pin_acc_num[net_id + 1] - pin_acc_num[net_id];
    for(int i = 1; i < net_routes[0]; i += 2) {
        int l = net_routes[i] / X / Y, x0 = net_routes[i] / Y % X, y0 = net_routes[i] % Y;
        int x1 = net_routes[i + 1] / Y % X, y1 = net_routes[i + 1] % Y;
        if(x0 != x1) of_count += of_edge_sum[IDX(l, max(x0, x1), y0)] - of_edge_sum[IDX(l, min(x0, x1), y0)];
        if(y0 != y1) of_count += of_edge_sum[IDX(l, x0, max(y0, y1))] - of_edge_sum[IDX(l, x0, min(y0, y1))];
        if(of_count > 12) {
            is_of_net[net_id] = true;
            break;
        }
    }
}

__global__ void commit_wire_demand(int net_cnt, int net_offset, int stamp, int K = 1) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= net_cnt) return;
    net_id = net_ids[net_id + net_offset];
    int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    double wirelength = 0;
    //commit wires
    for(int i = 1; i < net_routes[0]; i += 2) if(net_routes[i] / X / Y == net_routes[i + 1] / X / Y) {
        int l = net_routes[i] / X / Y;
        int x0 = net_routes[i] / Y % X, y0 = net_routes[i] % Y;
        int x1 = net_routes[i + 1] / Y % X, y1 = net_routes[i + 1] % Y;
        // One reservation per segment: a per-edge atomic on the shared counter
        // would contend far more than the demand updates themselves.
        int dirty_pos = reserve_dirty_cells(abs(x1 - x0) + abs(y1 - y0));
        if(x0 != x1) mark_dirty_track(l, x0, y0);
        if(y0 != y1) mark_dirty_track(l, x0, y0);
        if(x0 < x1) for(int x = x0; x < x1; x++) atomicAdd(demand + IDX(l, x, y0), K), wirelength += x_edge_len[x], timestamp[IDX(l, x, y0)] = stamp,
                        dirty_pos >= 0 ? dirty_cells[dirty_pos + x - x0] = IDX(l, x, y0) : 0;
        if(x1 < x0) for(int x = x1; x < x0; x++) atomicAdd(demand + IDX(l, x, y0), K), wirelength += x_edge_len[x], timestamp[IDX(l, x, y0)] = stamp,
                        dirty_pos >= 0 ? dirty_cells[dirty_pos + x - x1] = IDX(l, x, y0) : 0;
        if(y0 < y1) for(int y = y0; y < y1; y++) atomicAdd(demand + IDX(l, x0, y), K), wirelength += y_edge_len[y], timestamp[IDX(l, x0, y)] = stamp,
                        dirty_pos >= 0 ? dirty_cells[dirty_pos + y - y0] = IDX(l, x0, y) : 0;
        if(y1 < y0) for(int y = y1; y < y0; y++) atomicAdd(demand + IDX(l, x0, y), K), wirelength += y_edge_len[y], timestamp[IDX(l, x0, y)] = stamp,
                        dirty_pos >= 0 ? dirty_cells[dirty_pos + y - y1] = IDX(l, x0, y) : 0;
        timestamp[IDX(l, x0, y0)] = timestamp[IDX(l, x0, y1)] = timestamp[IDX(l, x1, y0)] = timestamp[IDX(l, x1, y1)] = stamp;
    }
    atomicAdd(&total_wirelength, wirelength * K);
}

__global__ void commit_via_demand(int net_cnt, int net_offset, int stamp, int K = 1) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= net_cnt) return;
    net_id = net_ids[net_id + net_offset];
    int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN, via_count = 0;
    //commit vias
    for(int i = 1; i < net_routes[0]; i += 2) if(net_routes[i] / X / Y != net_routes[i + 1] / X / Y) {
        int x = net_routes[i] / Y % X, y = net_routes[i] % Y, l0 = net_routes[i] / X / Y, l1 = net_routes[i + 1] / X / Y;
        int minl = min(l0, l1), maxl = max(l0, l1);
        via_count += maxl - minl;
        for(int l = minl; l < maxl; l++) if(timestamp[IDX(l, x, y)] < stamp) {
            if(l & 1 ^ DIR) {
                if(y == 0) {
                    atomicAdd(demand + IDX(l, x, y), K);
                    mark_dirty_cell(IDX(l, x, y));
                } else if(y == Y - 1) {
                    atomicAdd(demand + IDX(l, x, y - 1), K);
                    mark_dirty_cell(IDX(l, x, y - 1));
                } else {
                    atomicAdd(demand + IDX(l, x, y - 1), 0.5 * K);
                    atomicAdd(demand + IDX(l, x, y), 0.5 * K);
                    mark_dirty_cell(IDX(l, x, y - 1));
                    mark_dirty_cell(IDX(l, x, y));
                }
            } else {
                if(x == 0) {
                    atomicAdd(demand + IDX(l, x, y), K);
                    mark_dirty_cell(IDX(l, x, y));
                } else if(x == X - 1) {
                    atomicAdd(demand + IDX(l, x - 1, y), K);
                    mark_dirty_cell(IDX(l, x - 1, y));
                } else {
                    atomicAdd(demand + IDX(l, x - 1, y), 0.5 * K);
                    atomicAdd(demand + IDX(l, x, y), 0.5 * K);
                    mark_dirty_cell(IDX(l, x - 1, y));
                    mark_dirty_cell(IDX(l, x, y));
                }
            }
            //timestamp[IDX(l, x, y)] = stamp;
        }
    }
    atomicAdd(&total_via_count, via_count * K);
}

__global__ void extract_congestionView() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int x = idx / Y % X, y = idx % Y;
    double resource = capacity[idx] - demand[idx];
    if(resource < 0)
    {
        congestion[x * Y + y] = true;
        congestion_xsum[x * Y + y] = max(congestion_xsum[x * Y + y], -resource);
        congestion_ysum[x * Y + y] = max(congestion_ysum[x * Y + y], -resource);
    }
}

__global__ void extract_congestionView_ysum() {
    extern __shared__ float sum3[];
    if(threadIdx.x == 0) sum3[0] = 0;
    int x = blockIdx.x; // % XY;
    for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum3[y + 1] = congestion_ysum[x*Y+y];
    __syncthreads();
    for(int d = 0; (1 << d) < Y; d++) {
        for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
            if(idx >> d & 1) sum3[idx] += sum3[(idx >> d << d) - 1];
        __syncthreads();
    }
    for(int y = threadIdx.x; y < Y; y += blockDim.x) congestion_ysum[x*Y+y] = sum3[y];
}

__global__ void extract_congestionView_xsum() {
    extern __shared__ float sum3[];
    if(threadIdx.x == 0) sum3[0] = 0;
    int y = blockIdx.x;// % XY;
    for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum3[x + 1] = congestion_xsum[x*Y+y];
    __syncthreads();
    for(int d = 0; (1 << d) < X; d++) {
        for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
            if(idx >> d & 1) sum3[idx] += sum3[(idx >> d << d) - 1];
        __syncthreads();
    }
    for(int x = threadIdx.x; x < X; x += blockDim.x) congestion_xsum[x*Y+y] = sum3[x];
}

bool is_of_net_cpu[db::MAX_NET_NUM];

pair<vector<int>, vector<int>> ripup(int of_threshold) {
    assert(NET_NUM <= db::MAX_NET_NUM);
    
    cudaMemset(of_edge_sum, 0, sizeof(int) * L * X * Y);
    mark_overflow_edges<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> (of_threshold);
    compute_presum_general<<<all_track_cnt, THREAD_NUM, sizeof(int) * XY>>> (of_edge_sum);
    mark_overflow_nets<<<BLOCK_NUM(NET_NUM), THREAD_NUM>>> ();
    cudaMemcpy(&is_of_net_cpu, is_of_net, sizeof(bool) * NET_NUM, cudaMemcpyDeviceToHost);
    vector<int> of_nets, no_of_nets;
    for(int i = 0; i < NET_NUM; i++) 
        (is_of_net_cpu[i] ? of_nets : no_of_nets).emplace_back(i);
    return make_pair(move(of_nets), move(no_of_nets));
}

}