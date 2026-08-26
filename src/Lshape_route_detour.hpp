#include "graph.hpp"
#include "nvtx_profile.hpp"
#include <omp.h>
#include <atomic>
#include <unistd.h>
#include <queue>
#include <cstdlib>

namespace cudb {
bool cpu_tree_center_enabled();
int cpu_tree_center_min_degree();
void reset_cpu_tree_center_profile();
void print_cpu_tree_center_profile(const char *stage);
}

#define INF_LAYER 20
#define MAX_LAYER 10
#define MIN_ROUTE_LAYER 1
#define MAX_DEPTH 5000
__managed__ double *cost_edges;
__managed__ int *best_change;
__managed__ int edge_cnt;

namespace Lshape_route_detour {

//declaration
void Lshape_route_detour(vector<int> &nets2route);
__managed__ int *macroBorder;
int *macroBorder_cpu;
int cntCongested = 0;
int totalEdgeNum = 0;

__managed__ int *node_cnt_sum, *nodes, *par_nodes, *from, *layer_range;
int node_cnt_estimate;
int parent_cnt_estimate;
__managed__ int *child_num; 
__managed__ int *child_num_sum; 
__managed__ int *in_degree;
__managed__ int *currentChildIDX;

__managed__ int *par_num;
__managed__ int *par_num_sum;
__managed__ int *locks;
__managed__ double *childCosts;
__managed__ int *childCosts_road;

__managed__ int *best_path;

__managed__ int *layer_output;
__managed__ double *costs;
__managed__ int *fixed_layers;
__managed__ int *node_net_idx;
__managed__ int *node_net_idx2;

__managed__ int *lock_gpu;

__managed__ int *node_depth;
__managed__ int *net_depth;

__managed__ int *batch_depth;
__managed__ int *depth_node;
__managed__ int *depth_node_cnt;

bool *congestionView_cpu;
float *congestionView_xsum_cpu;
float *congestionView_ysum_cpu;
int *node_cnt_sum_cpu, *node_depth_cpu, *net_depth_cpu, *batch_depth_cnt_cpu, *depth_node_cnt_cpu, *depth_node_cpu,
    *nodes_cpu, *node_net_idx_cpu, *node_net_idx2_cpu, *child_num_cpu, *child_num_sum_cpu, *par_num_cpu, *par_num_sum_cpu,
    *par_nodes_cpu, *currentChildIDX_cpu, *depthID2nodeSequence;

// Lshape_route_detour() is called once per outer routing batch.  Each call's
// maximum augmented-DAG depth is a serial GPU level chain; collect all chains
// and report one Stage-2 critical-path summary after the outer loop.
vector<int> stage2_depth_levels;
double stage2_host_prep_seconds = 0, stage2_gpu_route_seconds = 0;

bool augmented_dag_profile_enabled() {
    const char *value = getenv("INSTANTGR_AUGMENTED_DAG_PROFILE");
    return value != nullptr && string(value) == "1";
}

struct AugmentedDagGpuProfile {
    bool enabled = false;
    cudaEvent_t start{}, end{};
    double remove_and_cost_seconds = 0.0;
    // remove_and_cost_seconds is the sum of the three sub-stages below; they are
    // timed separately because that bucket dominates Stage 2 and we need to know
    // which of the full-grid kernels inside it is responsible.
    double ripup_seconds = 0.0;
    double update_cost_seconds = 0.0;
    double presum_seconds = 0.0;
    double bottom_up_seconds = 0.0;
    double traceback_seconds = 0.0;
    double commit_seconds = 0.0;
    long long level_node_visits = 0;
};

AugmentedDagGpuProfile augmented_dag_gpu_profile;

// Sub-stage timing for the remove+cost bucket.  These only record markers into
// the stream and are read after the single synchronization that the bucket
// already performed, so splitting the bucket does not add a synchronization and
// does not perturb the pipeline.
cudaEvent_t remove_and_cost_marks[4];

void record_remove_and_cost_mark(int i) {
    if(augmented_dag_gpu_profile.enabled) cudaEventRecord(remove_and_cost_marks[i]);
}

void accumulate_remove_and_cost_marks() {
    if(!augmented_dag_gpu_profile.enabled) return;
    cudaEventSynchronize(remove_and_cost_marks[3]);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, remove_and_cost_marks[0], remove_and_cost_marks[1]);
    augmented_dag_gpu_profile.ripup_seconds += ms / 1000.0;
    cudaEventElapsedTime(&ms, remove_and_cost_marks[1], remove_and_cost_marks[2]);
    augmented_dag_gpu_profile.update_cost_seconds += ms / 1000.0;
    cudaEventElapsedTime(&ms, remove_and_cost_marks[2], remove_and_cost_marks[3]);
    augmented_dag_gpu_profile.presum_seconds += ms / 1000.0;
    cudaEventElapsedTime(&ms, remove_and_cost_marks[0], remove_and_cost_marks[3]);
    augmented_dag_gpu_profile.remove_and_cost_seconds += ms / 1000.0;
}

void start_augmented_dag_gpu_timer() {
    if(augmented_dag_gpu_profile.enabled)
        cudaEventRecord(augmented_dag_gpu_profile.start);
}

void stop_augmented_dag_gpu_timer(double &seconds) {
    if(!augmented_dag_gpu_profile.enabled) return;
    cudaEventRecord(augmented_dag_gpu_profile.end);
    cudaEventSynchronize(augmented_dag_gpu_profile.end);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, augmented_dag_gpu_profile.start,
                         augmented_dag_gpu_profile.end);
    seconds += milliseconds / 1000.0;
}

void print_augmented_dag_gpu_profile() {
    if(!augmented_dag_gpu_profile.enabled) return;
    const double total = augmented_dag_gpu_profile.remove_and_cost_seconds +
                         augmented_dag_gpu_profile.bottom_up_seconds +
                         augmented_dag_gpu_profile.traceback_seconds +
                         augmented_dag_gpu_profile.commit_seconds;
    const auto pct = [&] (double seconds) { return total > 0 ? 100.0 * seconds / total : 0.0; };
    printf("Stage 2 augmented-DAG GPU profile (profile run): remove+cost=%.3fs (%.1f%%), "
           "bottom-up=%.3fs (%.1f%%), traceback=%.3fs (%.1f%%), "
           "commit=%.3fs (%.1f%%), total=%.3fs, level-node-visits=%lld\n"
           "                    remove+cost split: ripup-commit=%.3fs (%.1f%%), "
           "update_cost=%.3fs (%.1f%%), compute_presum=%.3fs (%.1f%%)\n",
           augmented_dag_gpu_profile.remove_and_cost_seconds,
           pct(augmented_dag_gpu_profile.remove_and_cost_seconds),
           augmented_dag_gpu_profile.bottom_up_seconds,
           pct(augmented_dag_gpu_profile.bottom_up_seconds),
           augmented_dag_gpu_profile.traceback_seconds,
           pct(augmented_dag_gpu_profile.traceback_seconds),
           augmented_dag_gpu_profile.commit_seconds,
           pct(augmented_dag_gpu_profile.commit_seconds), total,
           augmented_dag_gpu_profile.level_node_visits,
           augmented_dag_gpu_profile.ripup_seconds,
           pct(augmented_dag_gpu_profile.ripup_seconds),
           augmented_dag_gpu_profile.update_cost_seconds,
           pct(augmented_dag_gpu_profile.update_cost_seconds),
           augmented_dag_gpu_profile.presum_seconds,
           pct(augmented_dag_gpu_profile.presum_seconds));
}

// Set INSTANTGR_BATCH_COVERAGE=1 to report how much of the grid each Stage-2
// batch actually touches.  update_vcost and compute_presum rebuild the whole
// L*X*Y grid once per batch, so this fraction is the ceiling on what a
// coverage-limited rebuild could save.
bool batch_coverage_enabled() {
    const char *value = getenv("INSTANTGR_BATCH_COVERAGE");
    return value != nullptr && string(value) == "1";
}

void report_batch_coverage(const vector<vector<int>> &batches) {
    if(!batch_coverage_enabled() || batches.empty()) return;
    const double coverage_start = elapsed_time();
    // Stamp arrays instead of clearing: a cell belongs to batch b if its stamp
    // equals b + 1.
    vector<int> cell_stamp(X * Y, 0), x_stamp(X, 0), y_stamp(Y, 0);
    vector<double> cell_fraction, track_fraction;
    cell_fraction.reserve(batches.size());
    track_fraction.reserve(batches.size());
    long long cell_sum = 0, x_sum = 0, y_sum = 0;
    for(int b = 0; b < batches.size(); b++) {
        const int stamp = b + 1;
        long long cells = 0, xs = 0, ys = 0;
        for(int net_id : batches[b]) {
            const auto &target = nets[net_id];
            if(target.node_index_cnt == 0) continue;
            // Bounding box of the augmented DAG: an upper bound on the region
            // whose costs this net can query.
            int minx = X, maxx = 0, miny = Y, maxy = 0;
            for(int i = 0; i < target.node_index_cnt; i++) {
                const int pos = target.nodes_cpu[i] % (X * Y);
                const int x = pos / Y, y = pos % Y;
                minx = min(minx, x); maxx = max(maxx, x);
                miny = min(miny, y); maxy = max(maxy, y);
            }
            for(int x = minx; x <= maxx; x++) {
                if(x_stamp[x] != stamp) { x_stamp[x] = stamp; xs++; }
                for(int y = miny; y <= maxy; y++)
                    if(cell_stamp[x * Y + y] != stamp) { cell_stamp[x * Y + y] = stamp; cells++; }
            }
            for(int y = miny; y <= maxy; y++)
                if(y_stamp[y] != stamp) { y_stamp[y] = stamp; ys++; }
        }
        cell_sum += cells; x_sum += xs; y_sum += ys;
        cell_fraction.emplace_back((double) cells / ((double) X * Y));
        // A track is one row or one column, so track coverage is the fraction
        // of distinct rows and columns the batch spans.
        track_fraction.emplace_back((double) (xs + ys) / (X + Y));
    }
    // Histogram before sorting is irrelevant, but keep the buckets next to the
    // percentiles so one run answers both "how skewed" and "how many where".
    const int bucket_cnt = 20;
    vector<int> cell_buckets(bucket_cnt, 0), track_buckets(bucket_cnt, 0);
    for(double f : cell_fraction) cell_buckets[min(bucket_cnt - 1, (int) (f * bucket_cnt))]++;
    for(double f : track_fraction) track_buckets[min(bucket_cnt - 1, (int) (f * bucket_cnt))]++;
    sort(cell_fraction.begin(), cell_fraction.end());
    sort(track_fraction.begin(), track_fraction.end());
    const auto pct = [&] (const vector<double> &v, double q) {
        const int i = min((int) v.size() - 1, max(0, (int) ceil(q * v.size()) - 1));
        return 100.0 * v[i];
    };
    printf("Stage 2 batch coverage over %zu batches (grid %dx%d): "
           "cells avg=%.2f%% p50=%.2f%% p90=%.2f%% max=%.2f%%, "
           "tracks avg=%.2f%% p50=%.2f%% p90=%.2f%% max=%.2f%%, measured in %.2fs\n",
           batches.size(), X, Y,
           100.0 * cell_sum / ((double) X * Y * batches.size()),
           pct(cell_fraction, 0.50), pct(cell_fraction, 0.90), pct(cell_fraction, 1.0),
           100.0 * (x_sum + y_sum) / ((double) (X + Y) * batches.size()),
           pct(track_fraction, 0.50), pct(track_fraction, 0.90), pct(track_fraction, 1.0),
           elapsed_time() - coverage_start);
    printf("Stage 2 coverage histogram, batches per 5%% bucket from 0-5%% to 95-100%%\n");
    printf("  cells :");
    for(int b = 0; b < bucket_cnt; b++) printf(" %d", cell_buckets[b]);
    printf("\n  tracks:");
    for(int b = 0; b < bucket_cnt; b++) printf(" %d", track_buckets[b]);
    printf("\n");
}

void reset_stage2_depth_schedule() {
    stage2_depth_levels.clear();
}

void print_stage2_depth_schedule() {
    if(!LOG || stage2_depth_levels.empty()) return;
    long long total_levels = 0;
    for(int levels : stage2_depth_levels) total_levels += levels;
    vector<int> sorted_levels = stage2_depth_levels;
    sort(sorted_levels.begin(), sorted_levels.end());
    const auto percentile = [&] (double q) {
        const int count = static_cast<int>(sorted_levels.size());
        const int index = min(count - 1, max(0, static_cast<int>(ceil(q * count)) - 1));
        return sorted_levels[index];
    };
    const bool cpu_center = cudb::cpu_tree_center_enabled();
    printf("Stage 2 augmented-DAG critical path: root=%s, degree-threshold=%d, route-batches=%zu, "
           "serialized depth phases=%lld, p50/p90/p99/max=%d/%d/%d/%d, "
           "node-kernel launches=%lld, tree-kernel launches=%lld, "
           "tree synchronizations=%lld, pre-batch synchronizations=%zu\n",
           cpu_center ? "cpu-tree-center" : "legacy",
           cpu_center ? cudb::cpu_tree_center_min_degree() : -1,
           stage2_depth_levels.size(), total_levels,
           percentile(0.50), percentile(0.90), percentile(0.99), sorted_levels.back(),
           total_levels, total_levels, total_levels, stage2_depth_levels.size());
}

__device__ void atomicMinDouble(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) {
            break;
        }
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__global__ void init_min_child_costs(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    childCosts[index] = INF;
}

__global__ void init_road(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    childCosts_road[index] = 200000000;
}

__global__ void init_costs(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>limit)
    {
        return;
    }
    costs[index] = INF;
}

__global__ void Lshape_route_node_cuda(int shift, int end_shift) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_idx = depth_node[node_sequence];
    int parent_num_cur = par_num_sum[node_idx+1]-par_num_sum[node_idx];
    int fixed_layer_low = 1 + nodes[node_idx] / X / Y;
    int x = nodes[node_idx] / Y % X, y = nodes[node_idx] % Y;
    int fixed_layer_high = fixed_layer_low==10?0:fixed_layer_low;
    int cur_child_num = child_num_sum[node_idx+1]-child_num_sum[node_idx];

    int *cur_best_path = best_path + child_num_sum[node_idx] * MAX_LAYER;
    double *cur_childCosts = childCosts + child_num_sum[node_idx] * MAX_LAYER;
    int *cur_childCosts_road = childCosts_road + child_num_sum[node_idx] * MAX_LAYER;
    double minChildCosts[6];
    int bestPaths[6];
    for (int lowLayerIndex = MIN_ROUTE_LAYER; lowLayerIndex <= fixed_layer_low; lowLayerIndex++) {
        for(int cid=0; cid<cur_child_num; cid++)
        {
            minChildCosts[cid] = INF;
        }
        double via_cost = 0;
        for (int layerIndex = lowLayerIndex; layerIndex < (L+1); layerIndex++) {
            if(layerIndex>lowLayerIndex)
            {   
                // min value of lowLayerIndex is 1 
                via_cost += vcost[IDX(layerIndex - 2, x, y)];
            }
            // int min_layer = 10;
            for (int childIndex = 0; childIndex < cur_child_num; childIndex++) {
                double cur_child_cost = cur_childCosts[childIndex * MAX_LAYER + layerIndex];
                if (cur_child_cost < minChildCosts[childIndex]) {
                    minChildCosts[childIndex] = cur_child_cost;
                    bestPaths[childIndex] = cur_childCosts_road[childIndex * MAX_LAYER + layerIndex] * MAX_LAYER + layerIndex;
                }
            }
            if (layerIndex >= fixed_layer_high) {
                double cost = via_cost;
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cost += minChildCosts[childIndex];
                }
                if (cost<INF && cost < costs[node_idx*MAX_LAYER+layerIndex]) {
                    costs[node_idx*MAX_LAYER+layerIndex] = cost;
                    for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                    {
                        cur_best_path[childIndex * MAX_LAYER + layerIndex] = bestPaths[childIndex];
                    }
                }
            }
        }
        for (int layerIndex = (L+1) - 2; layerIndex >= lowLayerIndex; layerIndex--) {//
            if (costs[node_idx*MAX_LAYER+layerIndex + 1] < costs[node_idx*MAX_LAYER+layerIndex]) {
                costs[node_idx*MAX_LAYER+layerIndex] = costs[node_idx*MAX_LAYER+layerIndex + 1];
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cur_best_path[childIndex * MAX_LAYER + layerIndex] = cur_best_path[childIndex * MAX_LAYER + layerIndex+1];
                }
            }
        }
    }
    int node_x = nodes[node_idx] / Y % X, node_y = nodes[node_idx] % Y; 

    for(int par_id = 0; par_id < parent_num_cur; par_id++)
    {
        int parent_IDX = par_nodes[par_num_sum[node_idx] + par_id];
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_idx] + par_id]%10;
        double *parent_childCosts = childCosts + child_num_sum[parent_IDX] * MAX_LAYER;
        int *parent_childCosts_road = childCosts_road + child_num_sum[parent_IDX] * MAX_LAYER;
        int px = nodes[parent_IDX] / Y % X, py = nodes[parent_IDX] % Y; 
        assert(px==node_x||py==node_y);
        for(int layer = MIN_ROUTE_LAYER; layer<MAX_LAYER; layer++)
        {
            if((layer & 1 ^ DIR) == 1 && node_y != py) continue;
            if((layer & 1 ^ DIR) == 0 && node_x != px) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + layer;
            double cost = costs[node_idx * MAX_LAYER + layer] + graph::wire_segment_cost(layer-1, min(node_x, px), max(node_x, px), min(node_y, py), max(node_y, py));
            int shift_modify = child_num_sum[parent_IDX] * MAX_LAYER + index_;
            atomicMinDouble(&parent_childCosts[index_], cost);
            if(parent_childCosts[index_]==cost)
            {
                parent_childCosts_road[index_] = node_idx;
            }
        }
    }
}

__global__ void get_routing_tree_cuda(int shift, int end_shift, int depth, int stamp) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_id = depth_node[node_sequence];
    int net_id = node_net_idx[node_id];
    int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    int *cur_best_path = best_path + child_num_sum[node_id] * MAX_LAYER;

    int l = nodes[node_id] / Y / X;
    int cur_x = nodes[node_id] / Y % X, cur_y = nodes[node_id] % Y;
    if(par_num_sum[node_id+1]-par_num_sum[node_id]==0)
    {
        int min_layer = 0;
        double min_cost = costs[node_id * MAX_LAYER];
        for(int layer = 1; layer < MAX_LAYER; layer++)
        {
            if(costs[node_id * MAX_LAYER + layer] < min_cost)
            {
                min_cost = costs[ node_id * MAX_LAYER + layer];
                min_layer = layer;
            }
        }
        layer_output[node_id] = min_layer;
        net_routes[0] = 1;
    } else{
        int par_layer = -1;
        int par_idx = -1;
        int par_sequence = -1;
        for(int par_id = 0; par_id< par_num_sum[node_id+1]-par_num_sum[node_id]; par_id++)
        {
            int par_node = par_nodes[par_num_sum[node_id]+par_id];
            if(layer_output[par_node]>=0)
            {
                par_idx = par_node;
                par_sequence = par_id;
                par_layer = layer_output[par_node];
                int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
                int *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
                int path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
                int child_idx = path / MAX_LAYER;
                if(child_idx == node_id)
                {
                    layer_output[node_id] = path % MAX_LAYER;
                    int px = nodes[par_idx] / Y % X, py = nodes[par_idx] % Y;
                    assert(px==cur_x||py==cur_y); 
                    if(px==cur_x && cur_y!=py)
                    {
                        graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, px, px, min(py,cur_y), max(py,cur_y), stamp);
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(layer_output[node_id] - 1, px, min(py,cur_y));
                        net_routes[idd1+1] = IDX(layer_output[node_id] - 1, px, max(py,cur_y));
                    }
                    else if(py==cur_y && cur_x != px)
                    {
                        graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, min(px,cur_x), max(px,cur_x), py, py, stamp);
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(layer_output[node_id] - 1, min(px,cur_x), py);
                        net_routes[idd1+1] = IDX(layer_output[node_id] - 1, max(px,cur_x), py);
                    }
                    break;
                }else{
                    layer_output[node_id] = -1;
                }
            }
        }
        if(par_layer==-1)
        {
            layer_output[node_id] = -1;
            return;
        }
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
        int *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
        int path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
        int child_idx = path / MAX_LAYER;
        if( child_idx != node_id)
        {
            layer_output[node_id] = -1;
            return;
        }
    }
    int num_child = child_num_sum[node_id+1] - child_num_sum[node_id];
    int minl = l+1;
    int maxl = (l+1)==MAX_LAYER?1:minl;
    minl = min(minl,layer_output[node_id]);
    maxl = max(maxl,layer_output[node_id]);
    assert(num_child>=0);
    if(num_child>0)
    {
        for(int child_id=0; child_id<num_child; child_id++)
        {
            int layer_of_child = cur_best_path[child_id * MAX_LAYER + layer_output[node_id]] % MAX_LAYER;
            assert(layer_of_child!=0);
            minl = min(layer_of_child, minl);
            maxl = max(layer_of_child, maxl);
        }
    }
    if(minl<maxl)
    {
        int idd1 = atomicAdd(net_routes,2);
        net_routes[idd1] = IDX(minl - 1, cur_x, cur_y);
        net_routes[idd1+1] = IDX(maxl - 1, cur_x, cur_y);
    }
}

void process_net(int thread_idx, vector<int> &nets2route, int thread_num, std::atomic<int>& currentNetId) {
    while (true) {
        int netId = currentNetId.fetch_add(1);
        if (netId >= nets2route.size()) {
            break;
        }
        nets[nets2route[netId]].generate_detours(move(congestionView_cpu), move(congestionView_xsum_cpu), move(congestionView_ysum_cpu), false);
    }
}

void multithreaded_processing(vector<int> &nets2route) {
    std::vector<std::thread> threads;
    int max_threads = 8;
    threads.reserve(max_threads);
    std::atomic<int> currentNetId(0);
    for (int i = 0; i < max_threads; ++i) {
        threads.emplace_back(process_net, i, std::ref(nets2route), max_threads, std::ref(currentNetId));
    }
    for (auto& t : threads) {
        t.join();
    }
}

void Lshape_route_detour_wrap(vector<int> &nets2route)
{
    double DAG_start_time = elapsed_time();
    if (nets2route.size() == 0)
    {
        return;
    }
    if(LOG) printf("[%5.1f] Stage 2 rerouting: net ordering starts\n", elapsed_time());
    sort(nets2route.begin(), nets2route.end(), [](int l, int r)
         { return nets[l].hpwl > nets[r].hpwl; });
    if(LOG) printf("[%5.1f] Stage 2 rerouting: net ordering ends\n", elapsed_time());
    congestionView_cpu = new bool[X * Y * sizeof(bool)];
    congestionView_xsum_cpu = new float[X * Y * sizeof(float)];
    congestionView_ysum_cpu = new float[X * Y * sizeof(float)];
    cudaMemcpy(congestionView_cpu, congestion, X * Y * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(congestionView_xsum_cpu, congestion_xsum, X * Y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(congestionView_ysum_cpu, congestion_ysum, X * Y * sizeof(float), cudaMemcpyDeviceToHost);
    if(LOG) printf("[%5.1f] Stage 2 rerouting: detour generation starts\n", elapsed_time());
    const double detour_gen_start_time = elapsed_time();
    const bool use_gpu_flute = gpu_flute_enabled();
    vector<int> gpu_flute_nets;
    for (int net_id : nets2route) {
        if (nets[net_id].rsmt.size() >= 1) continue;
        if (!use_gpu_flute || !gpu_flute_takes_degree(nets[net_id].pins.size()))
            nets[net_id].construct_rsmt();
        else
            gpu_flute_nets.emplace_back(net_id);
    }
    if (use_gpu_flute) construct_rsmt_gpu(gpu_flute_nets);
    cudb::reset_cpu_tree_center_profile();
    multithreaded_processing(nets2route);
    cudb::print_cpu_tree_center_profile("Stage 2");
    record_runtime_stage("  S2: detour generation", detour_gen_start_time);
    if(LOG) printf("[%5.1f] Stage 2 rerouting: detour generation ends\n", elapsed_time());
    if(LOG) printf("[%5.1f] Stage 2 rerouting: batch generation starts\n", elapsed_time());
    const double batch_gen_start_time = elapsed_time();
    auto batches = generate_batches_rsmt(nets2route, 300000);
    record_runtime_stage("  S2: batch generation", batch_gen_start_time);
    if(LOG) printf("[%5.1f] Stage 2 rerouting: batch generation ends\n", elapsed_time());
    report_batch_coverage(batches);
    int net_cnt_estimate = 0;
    int node_num_max = 0;
    int par_num_max = 0;
    for (int ii = 0; ii < batches.size(); ii++)
    {
        int tmp = 0;
        int tmp2 = 0;
        if (batches[ii].size() > net_cnt_estimate)
        {
            net_cnt_estimate = batches[ii].size();
        }
        for (int j = 0; j < batches[ii].size(); j++)
        {
            auto &graph_x = nets[batches[ii][j]].rsmt;
            tmp += nets[batches[ii][j]].node_index_cnt;
            tmp2 += nets[batches[ii][j]].par_num_sum_cpu[nets[batches[ii][j]].node_index_cnt];
        }
        if (tmp > node_num_max)
        {
            node_num_max = tmp;
        }
        if (tmp2 > par_num_max)
        {
            par_num_max = tmp2;
        }
    }

    net_cnt_estimate += 5;
    node_cnt_estimate = node_num_max + 10;
    parent_cnt_estimate = par_num_max + 10;
    if(LOG) printf("[%5.1f] Stage 2 rerouting: augmented-DAG route starts\n", elapsed_time());
    cudaMalloc(&node_cnt_sum, net_cnt_estimate * sizeof(int));
    cudaMalloc(&nodes, node_cnt_estimate * sizeof(int));
    cudaMalloc(&net_depth, net_cnt_estimate * sizeof(int));
    cudaMalloc(&batch_depth, (batches.size() + 1) * sizeof(int));
    cudaMalloc(&child_num_sum, node_cnt_estimate * sizeof(int));
    cudaMalloc(&par_num_sum, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_net_idx, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_net_idx2, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_depth, node_cnt_estimate * sizeof(int));
    cudaMalloc(&depth_node, node_cnt_estimate * sizeof(int));
    cudaMalloc(&layer_range, node_cnt_estimate * sizeof(int));
    cudaMalloc(&costs, node_cnt_estimate * MAX_LAYER * sizeof(double));
    cudaMalloc(&locks, parent_cnt_estimate * MAX_LAYER * sizeof(int));
    cudaMemset(locks, 0, sizeof(int) * parent_cnt_estimate * MAX_LAYER);
    cudaMalloc(&layer_output, node_cnt_estimate * sizeof(int));
    cudaMalloc(&par_nodes, parent_cnt_estimate * sizeof(int));
    cudaMalloc(&cost_edges, parent_cnt_estimate * 81 * sizeof(double));
    cudaMalloc(&best_change, parent_cnt_estimate * 81 * sizeof(int));
    cudaMalloc(&best_path, parent_cnt_estimate * MAX_LAYER * sizeof(int));
    cudaMalloc(&childCosts, parent_cnt_estimate * MAX_LAYER * sizeof(double));
    cudaMalloc(&childCosts_road, parent_cnt_estimate * MAX_LAYER * sizeof(int));
    cudaMalloc(&currentChildIDX, parent_cnt_estimate * sizeof(int));
    ///////////////////////////////////  cpu arrays init starts  ////////////////////////////////////////
    node_cnt_sum_cpu = new int[net_cnt_estimate]();
    int reserve_node_num = node_cnt_estimate;
    int biggest_depth = MAX_DEPTH;
    node_depth_cpu = new int[reserve_node_num]();
    net_depth_cpu = new int[net_cnt_estimate]();
    batch_depth_cnt_cpu = new int[batches.size() + 1]();
    depth_node_cnt_cpu = new int[biggest_depth * (batches.size() + 1)]();
    depth_node_cpu = new int[reserve_node_num]();
    nodes_cpu = new int[reserve_node_num]();
    node_net_idx_cpu = new int[reserve_node_num]();
    node_net_idx2_cpu = new int[reserve_node_num]();
    child_num_cpu = new int[reserve_node_num]();
    child_num_sum_cpu = new int[reserve_node_num]();
    par_num_cpu = new int[reserve_node_num]();
    par_num_sum_cpu = new int[reserve_node_num]();
    par_nodes_cpu = new int[parent_cnt_estimate]();
    currentChildIDX_cpu = new int[parent_cnt_estimate]();
    depthID2nodeSequence = new int[batches.size() * MAX_DEPTH];
    ///////////////////////////////////  cpu arrays init ends  ////////////////////////////////////////

    reset_stage2_depth_schedule();
    stage2_host_prep_seconds = 0;
    stage2_gpu_route_seconds = 0;
    augmented_dag_gpu_profile = {};
    augmented_dag_gpu_profile.enabled = augmented_dag_profile_enabled();
    if(augmented_dag_gpu_profile.enabled) {
        cudaEventCreate(&augmented_dag_gpu_profile.start);
        cudaEventCreate(&augmented_dag_gpu_profile.end);
        for(int i = 0; i < 4; i++) cudaEventCreate(&remove_and_cost_marks[i]);
    }
    for (int ii = batches.size() - 1; ii >= 0; ii--)
    {
        Lshape_route_detour(batches[ii]);
    }
    print_stage2_depth_schedule();
    print_augmented_dag_gpu_profile();
    runtime_stages.push_back({"  S2: host DAG prep + upload", stage2_host_prep_seconds});
    runtime_stages.push_back({"  S2: GPU route batches", stage2_gpu_route_seconds});
    if(augmented_dag_gpu_profile.enabled) {
        cudaEventDestroy(augmented_dag_gpu_profile.start);
        cudaEventDestroy(augmented_dag_gpu_profile.end);
        for(int i = 0; i < 4; i++) cudaEventDestroy(remove_and_cost_marks[i]);
    }

    cudaFree(node_cnt_sum);
    cudaFree(par_nodes);
    cudaFree(nodes);
    cudaFree(from);
    cudaFree(layer_range);
    cudaFree(in_degree);
    cudaFree(currentChildIDX);
    cudaFree(par_num_sum);
    cudaFree(locks);
    cudaFree(layer_output);
    cudaFree(fixed_layers);
    cudaFree(node_net_idx);
    cudaFree(node_net_idx2);
    cudaFree(lock_gpu);
    cudaFree(node_depth);
    cudaFree(net_depth);
    cudaFree(batch_depth);
    cudaFree(depth_node);
    cudaFree(depth_node_cnt);
    cudaFree(childCosts);
    cudaFree(childCosts_road);
    cudaFree(best_path);
    delete[] congestionView_cpu;
    delete[] node_cnt_sum_cpu;
    delete[] node_depth_cpu;
    delete[] net_depth_cpu;
    delete[] batch_depth_cnt_cpu;
    delete[] depth_node_cnt_cpu;
    delete[] depth_node_cpu;
    delete[] nodes_cpu;
    delete[] node_net_idx_cpu;
    delete[] node_net_idx2_cpu;
    delete[] child_num_cpu;
    delete[] child_num_sum_cpu;
    delete[] par_num_cpu;
    delete[] par_num_sum_cpu;
    delete[] par_nodes_cpu;
    delete[] currentChildIDX_cpu;
    if(LOG) printf("[%5.1f] Stage 2 rerouting: augmented-DAG route ends\n", elapsed_time());
    DAG_time = elapsed_time() - DAG_start_time;
    if(LOG) printf("[%5.1f] Stage 2 rip-up and rerouting ends\n", elapsed_time());
}

void Lshape_route_detour(vector<int> &nets2route) {
    const double fn_start_time = elapsed_time();
    vector<vector<int>> batches;
    batches.push_back(nets2route);
    vector<int> batch_cnt_sum(batches.size() + 1, 0);
    for(int i = 0; i < batches.size(); i++) {
        batch_cnt_sum[i + 1] = batch_cnt_sum[i] + batches[i].size();
        for(int j = 0; j < batches[i].size(); j++)
        {
            int net_id = batch_cnt_sum[i] + j;
            nets2route[net_id] = batches[i][j];
        }
    }
    int net_cnt = nets2route.size();
    int node_cnt = 0;
    int par_cnt = 0;
    for(auto net_id : nets2route)
    {
        node_cnt += nets[net_id].node_index_cnt;
        par_cnt += nets[net_id].par_num_sum_cpu[nets[net_id].node_index_cnt];
    }
    ////////////////////////////////////////////// cpu array memset starts //////////////////////////////////////////////////
    memset(node_cnt_sum_cpu, 0, (net_cnt + 1) * sizeof(int));
    int batch_reserve_node = node_cnt + 10;
    int reserve_node_num = min(node_cnt_estimate, batch_reserve_node);//to be optimized
    int reserve_par_num = min(parent_cnt_estimate, par_cnt+10);
    memset(net_depth_cpu, 0, net_cnt * sizeof(int));
    memset(batch_depth_cnt_cpu, 0, (batches.size()+1) * sizeof(int));
    memset(depth_node_cnt_cpu, 0, MAX_DEPTH*(batches.size()+1) * sizeof(int));
    memset(child_num_sum_cpu, 0, reserve_node_num * sizeof(int));
    memset(par_num_cpu, 0, reserve_node_num * sizeof(int));
    memset(par_num_sum_cpu, 0, reserve_node_num * sizeof(int));
    memset(depthID2nodeSequence, 0, batches.size()*MAX_DEPTH * sizeof(int));
    ////////////////////////////////////////////// cpu array memset ends //////////////////////////////////////////////////
    for(int b_id=0; b_id<batches.size(); b_id++)
    for(int j=0; j< batches[b_id].size(); j++){
        int net_idx = batches[b_id][j];
        auto &graph_x = nets[net_idx].rsmt;
        int select_root = nets[net_idx].select_root;
        int id = batch_cnt_sum[b_id] + j;
        int net_num_nodes = nets[net_idx].node_index_cnt;
        node_cnt_sum_cpu[id+1] = net_num_nodes;
        for(int n_i= 0; n_i < net_num_nodes; n_i++)
        { 
            int node_id = node_cnt_sum_cpu[id] + n_i;
            nodes_cpu[node_id] = nets[net_idx].nodes_cpu[n_i];
            child_num_cpu[node_id] = nets[net_idx].child_num_cpu[n_i];
            child_num_sum_cpu[node_id+1] = child_num_sum_cpu[node_id] + child_num_cpu[node_id];
            node_depth_cpu[node_id] = nets[net_idx].node_depth_cpu[n_i];
            int depth = node_depth_cpu[node_id];
            batch_depth_cnt_cpu[b_id+1] = max(batch_depth_cnt_cpu[b_id+1], node_depth_cpu[node_id]+1);
            net_depth_cpu[id] = max(net_depth_cpu[id], depth);
            par_num_cpu[node_id] = nets[net_idx].par_num_cpu[n_i];
            par_num_sum_cpu[node_id+1] = par_num_sum_cpu[node_id] + par_num_cpu[node_id];
            node_net_idx_cpu[node_id] = net_idx;
            node_net_idx2_cpu[node_id] = id;
        }
        int par_num_total = nets[net_idx].par_num_sum_cpu[net_num_nodes];
        int node_start = node_cnt_sum_cpu[id];
        int pid_total = 0;
        for(int n_i= 0; n_i < net_num_nodes; n_i++)
        {
            int node_id = node_cnt_sum_cpu[id] + n_i;
            for(int n_pid = 0; n_pid <  par_num_cpu[node_id]; n_pid++)
            {
                currentChildIDX_cpu[par_num_sum_cpu[node_start] + pid_total] 
                    = nets[net_idx].currentChildIDX_cpu[pid_total] + node_id * 10;
                par_nodes_cpu[par_num_sum_cpu[node_start] + pid_total] = nets[net_idx].par_nodes_cpu[pid_total] + node_cnt_sum_cpu[id];
                pid_total++;
            }
        }
        for(int i=node_cnt_sum_cpu[id];i<node_cnt_sum_cpu[id]+node_cnt_sum_cpu[id + 1];i++)
        {
            int depth_node_i = node_depth_cpu[i];
            depth_node_cnt_cpu[b_id*MAX_DEPTH+depth_node_i+1]++;
        }     
        node_cnt_sum_cpu[id + 1] += node_cnt_sum_cpu[id];
    }
    for(int bid=1; bid <= batches.size(); bid++)
    {
        batch_depth_cnt_cpu[bid]+=batch_depth_cnt_cpu[bid-1];
    }
    for(int bid=0; bid < batches.size(); bid++)
    {
        assert(batch_depth_cnt_cpu[bid+1]-batch_depth_cnt_cpu[bid]<MAX_DEPTH);
        for(int d = 0; d< batch_depth_cnt_cpu[bid+1]-batch_depth_cnt_cpu[bid]; d++)
        {
            depth_node_cnt_cpu[bid*MAX_DEPTH+d+1]+=depth_node_cnt_cpu[bid*MAX_DEPTH+d];
        }
        for(int d = 0; d<= batch_depth_cnt_cpu[bid+1]-batch_depth_cnt_cpu[bid]; d++)
        {
            depthID2nodeSequence[batch_depth_cnt_cpu[bid]+d] = node_cnt_sum_cpu[batch_cnt_sum[bid]] + depth_node_cnt_cpu[bid*MAX_DEPTH+d];
            depth_node_cnt_cpu[bid*MAX_DEPTH+d] += node_cnt_sum_cpu[batch_cnt_sum[bid]];
        }
        
        for(int node_id = node_cnt_sum_cpu[batch_cnt_sum[bid]]; node_id < node_cnt_sum_cpu[batch_cnt_sum[bid+1]]; node_id++)
        {
            int depth = node_depth_cpu[node_id];
            depth_node_cpu[depth_node_cnt_cpu[bid*MAX_DEPTH+depth]++] = node_id;
        }
    }
    for(int bid = 0; bid < batches.size(); ++bid)
        stage2_depth_levels.emplace_back(batch_depth_cnt_cpu[bid + 1] - batch_depth_cnt_cpu[bid]);
    int node_total = node_cnt_sum_cpu[net_cnt];
    for(int node_id = 1; node_id <= node_total; node_id++)
    {
        child_num_sum_cpu[node_id] = child_num_cpu[node_id-1];
        child_num_sum_cpu[node_id] += child_num_sum_cpu[node_id - 1];
    }
    cudaMemcpy(net_ids, nets2route.data(), net_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(layer_output, 0xFF, reserve_node_num * sizeof(int));
    cudaMemcpy(node_cnt_sum, node_cnt_sum_cpu, (net_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nodes, nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(net_depth, net_depth_cpu, net_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(batch_depth, batch_depth_cnt_cpu, (batches.size()+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(child_num_sum, child_num_sum_cpu, (node_cnt_sum_cpu[net_cnt]+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(par_num_sum, par_num_sum_cpu, (node_total+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node_net_idx, node_net_idx_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node_depth, node_depth_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_node, depth_node_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(par_nodes, par_nodes_cpu, par_num_sum_cpu[node_total] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(currentChildIDX, currentChildIDX_cpu, par_num_sum_cpu[node_total] * sizeof(int), cudaMemcpyHostToDevice);
    init_costs<<<node_total, 10>>>(node_total*MAX_LAYER);
    init_min_child_costs<<<(child_num_sum_cpu[node_total]+1) * MAX_LAYER, 1>>>((child_num_sum_cpu[node_total]+1) * MAX_LAYER);
    init_road<<<(child_num_sum_cpu[node_total]+1) * MAX_LAYER, 1>>>((child_num_sum_cpu[node_total]+1) * MAX_LAYER);
    
    const double gpu_loop_start_time = elapsed_time();
    stage2_host_prep_seconds += gpu_loop_start_time - fn_start_time;
    for(int i = 0; i < batches.size(); i++) {
        int total_node_num = batch_cnt_sum[i+1] - batch_cnt_sum[i];
        int net_offset = batch_cnt_sum[i];
        int next_net_offset = batch_cnt_sum[i+1];

        NVTX_PUSH("S2/batch", STAGE);
        NVTX_PUSH("S2/ripup", RIPUP);
        record_remove_and_cost_mark(0);
        graph::commit_wire_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, ++global_timestamp, -1);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, global_timestamp, -1);     
        global_timestamp++;
        record_remove_and_cost_mark(1);
        NVTX_POP();
        NVTX_PUSH("S2/update_cost", UPDATE_COST);
        graph::update_cost();
        record_remove_and_cost_mark(2);
        NVTX_POP();
        NVTX_PUSH("S2/compute_presum", PRESUM);
        graph::compute_presum<<<all_track_cnt, THREAD_NUM, sizeof(double) * XY>>> ();
        graph::finish_cost_refresh();
        record_remove_and_cost_mark(3);
        NVTX_POP();
        if(augmented_dag_gpu_profile.enabled)
            accumulate_remove_and_cost_marks();
        else
            cudaDeviceSynchronize();
        int cur_batch_depth = batch_depth_cnt_cpu[i+1] - batch_depth_cnt_cpu[i];
        NVTX_PUSH("S2/bottom_up_DP", BOTTOM_UP);
        start_augmented_dag_gpu_timer();
        for(int d = cur_batch_depth - 1; d >= 0; d--)
        {
            int shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d];
            int end_shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d+1];
            augmented_dag_gpu_profile.level_node_visits += end_shift - shift;
            Lshape_route_node_cuda<<<BLOCK_NUM(end_shift-shift+1), 512>>> (shift, end_shift);
        }
        stop_augmented_dag_gpu_timer(augmented_dag_gpu_profile.bottom_up_seconds);
        NVTX_POP();
        NVTX_PUSH("S2/traceback", TRACEBACK);
        start_augmented_dag_gpu_timer();
        for(int d = 0; d < cur_batch_depth; d++)
        {
            int shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d];
            int end_shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d+1];
            get_routing_tree_cuda<<<BLOCK_NUM(end_shift-shift+1), 512>>> (shift, end_shift, d, global_timestamp);
            cudaDeviceSynchronize();
        }
        stop_augmented_dag_gpu_timer(augmented_dag_gpu_profile.traceback_seconds);
        NVTX_POP();
        NVTX_PUSH("S2/commit", COMMIT);
        start_augmented_dag_gpu_timer();
        graph::batch_wire_update(global_timestamp);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, global_timestamp);
        stop_augmented_dag_gpu_timer(augmented_dag_gpu_profile.commit_seconds);
        NVTX_POP();
        NVTX_POP();// S2/batch
    }
    stage2_gpu_route_seconds += elapsed_time() - gpu_loop_start_time;
}

}
