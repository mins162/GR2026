#pragma once
#include "database.hpp"
#include <queue>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include "gpu_flute.hpp"
#include "gpu_batch_gen.hpp"

namespace cudb {

__managed__ int *rip_up_list;
__managed__ bool *congestion;
__managed__ float *congestion_xsum;
__managed__ float *congestion_ysum;
std::mutex mtx;

__managed__ unsigned long long ROUTE_PER_PIN = 27;

__managed__ int DIR, L, X, Y, XY, NET_NUM, PIN_NUM, *x_edge_len, *y_edge_len, *pin_acc_num, *pins;//DIR: direction of layer 0 in cuda database
int *flute_x_coords, *flute_y_coords;
__managed__ double unit_length_wire_cost, unit_via_cost, *unit_length_short_costs, *layer_min_len;
__managed__ float *vcost, *wcost, *capacity, *demand;
__managed__ double total_wirelength = 0, total_overflow_cost, layer_overflow_cost[10];
__managed__ int total_via_count = 0;

__managed__ bool *is_of_net;
__managed__ int *of_edge_sum;
__managed__ int *routes, *timestamp, *pre_demand;
__managed__ int *last, all_track_cnt, *idx2track, *track_net, *track_pos, *track_xy;
__managed__ double *presum;
// Cells whose demand changed since the last update_cost().  vcost depends only
// on demand at the cell and one neighbor, so recomputing just these cells (plus
// the dependent neighbor) is equivalent to the full-grid rebuild.  Overflowing
// the list is safe: it forces one full rebuild.
__managed__ int *dirty_cells;
__managed__ int dirty_cell_cnt = 0, dirty_cell_capacity = 0, dirty_cell_limit = 0;
__managed__ bool dirty_overflow = true;
__managed__ bool incremental_vcost_on = false;
// Per-track dirty flags for compute_presum, keyed l * XY + x (vertical layers)
// or l * XY + y (horizontal layers) — the same key idx2track uses.  A presum
// track only changes when demand changes inside it, so clean tracks keep their
// stored prefix sums.
__managed__ bool *track_dirty;
__managed__ bool incremental_presum_on = false;
// Same track key, but for the wire-demand commit: a track is flagged when the
// traceback drops a pre_demand marker on it, and only flagged tracks need the
// prefix-sum / commit / clear passes of batch_wire_update.  Cleared per track
// by the commit kernel itself, so unflagged tracks are provably all-zero.
__managed__ bool *pre_demand_track_dirty;
__managed__ int *net_ids;
int *net_ids_cpu;
int global_timestamp = 0;
vector<vector<int>> net_x_cpu, net_y_cpu;
vector<int> pin_cnt_sum_cpu;

#define IDX(l, x, y) ((l) * X * Y + (x) * Y + (y))
#define THREAD_NUM 512
#define BLOCK_NUM(n) ((n) / THREAD_NUM + 1)
#define BLOCK_CNT(tot, thread_cnt) ((tot) / (thread_cnt) + ((tot) % (thread_cnt) > 0))
#define INF 1e22


void build_cuda_database();

struct net {
    void construct_rsmt();
    void finalize_rsmt(const unordered_map<int, int> &layer);
    void generate_detours(bool *congestionView_cpu, float *congestionView_xsum_cpu, float *congestionView_ysum_cpu,
                                  bool construct_segments = true, bool display = false);
    void calc_hpwl();

    int minx, maxx, miny, maxy, hpwl, original_net_id;
    vector<int> pins;
    vector<vector<int>> rsmt;
    vector<pair<int, int>> rsmt_h_segments, rsmt_v_segments;
    vector<int> par_num_cpu;
    vector<int> par_num_sum_cpu;
    vector<int> currentChildIDX_cpu;
    vector<int> par_nodes_cpu;
    vector<int> child_num_cpu;
    vector<int> node_depth_cpu;
    vector<int> nodes_cpu;
    vector<int> points;
    int node_index_cnt = 0;
    int MAX_LAYER=10;
    int select_root = 0;
    // 2-D grid position returned by the optional GPU tree-center pass; -1
    // preserves the original select_root_net() behavior.
    int gpu_tree_center_pos = -1;
    bool gpu_tree_center_fallback = false;

    flute::Tree tree;
    unordered_map<int, int> layer;
};
vector<net> nets;

// Set INSTANTGR_GPU_FLUTE=0 to retain the original all-CPU FLUTE behavior.
// This is an A/B validation switch; the default remains GPU-FLUTE for
// high-degree root nets.
bool gpu_flute_enabled() {
    const char *value = getenv("INSTANTGR_GPU_FLUTE");
    return value == nullptr || string(value) != "0";
}

// A/B knob for the CPU/GPU overlap itself, separate from GPU-FLUTE: with 0 the
// high-degree GPU batch runs to completion before the low-degree CPU loop
// starts, so the two stage timers add up instead of hiding one another.
// INSTANTGR_GPU_FLUTE=0 cannot measure this, since it removes the GPU path
// along with the overlap.
bool flute_overlap_enabled() {
    const char *value = getenv("INSTANTGR_FLUTE_OVERLAP");
    return value == nullptr || string(value) != "0";
}

// GPU receives nets whose pin degree is at least this value: the original
// split, CPU degree <= DEGREE (9), GPU degree >= 10.
constexpr int GPU_FLUTE_MIN_DEGREE = DEGREE + 1;

// Experiment knob: INSTANTGR_GPU_FLUTE_MIN_DEGREE lowers the split so that
// LUT-degree roots also go through GPU-FLUTE (2 sends every net to the GPU;
// the device LUT kernel solves them directly).  Unset keeps the original 10.
int gpu_flute_min_degree() {
    static const int min_degree = [] {
        const char *value = getenv("INSTANTGR_GPU_FLUTE_MIN_DEGREE");
        if(value == nullptr || *value == '\0') return GPU_FLUTE_MIN_DEGREE;
        char *end = nullptr;
        const long parsed = strtol(value, &end, 10);
        if(*end != '\0' || parsed < 2 || parsed > INT_MAX)
            throw invalid_argument("INSTANTGR_GPU_FLUTE_MIN_DEGREE must be an integer >= 2");
        return static_cast<int>(parsed);
    }();
    return min_degree;
}

// Escape hatch for pathological roots: nets above this degree skip GPU-FLUTE
// and take the original CPU FLUTE path (its MAXD is 10000).  Unset or 0
// disables the cap.  Kept as an experiment knob; the O(1) precomputed break
// scores are the intended fix for huge nets.
int gpu_flute_max_degree() {
    static const int max_degree = [] {
        const char *value = getenv("INSTANTGR_GPU_FLUTE_MAX_DEGREE");
        if(value == nullptr || *value == '\0' || string(value) == "0") return INT_MAX;
        char *end = nullptr;
        const long parsed = strtol(value, &end, 10);
        if(*end != '\0' || parsed < gpu_flute_min_degree() || parsed > INT_MAX)
            throw invalid_argument("INSTANTGR_GPU_FLUTE_MAX_DEGREE must be 0 (off) or an integer >= "
                                   "the GPU-FLUTE minimum degree");
        return static_cast<int>(parsed);
    }();
    return max_degree;
}

// One predicate for both stages' CPU/GPU net partition.
bool gpu_flute_takes_degree(int degree) {
    return degree >= gpu_flute_min_degree() && degree <= gpu_flute_max_degree();
}

bool gpu_flute_profile_enabled() {
    const char *value = getenv("INSTANTGR_GPU_FLUTE_PROFILE");
    return value != nullptr && string(value) == "1";
}

bool gpu_flute_validation_enabled() {
    const char *value = getenv("INSTANTGR_GPU_FLUTE_VALIDATE");
    return value != nullptr && string(value) == "1";
}

// Opt-in only: preserve the original BFS-last-node root selection unless the
// experiment is explicitly enabled with INSTANTGR_GPU_TREE_CENTER=1.
bool gpu_tree_center_enabled() {
    const char *value = getenv("INSTANTGR_GPU_TREE_CENTER");
    return value != nullptr && string(value) == "1";
}

// Exact host-side tree-center root selection.  Unlike INSTANTGR_GPU_TREE_CENTER,
// this runs on the already reconstructed RSMT graph used by Stage 2, so its
// root is guaranteed to attain the graph radius.  On by default since leaf
// peeling made it a net win (2026-08-24 A/B); INSTANTGR_TREE_CENTER=0 disables.
bool cpu_tree_center_enabled() {
    static const bool enabled = [] {
        const char *value = getenv("INSTANTGR_TREE_CENTER");
        if(value == nullptr || *value == '\0' || string(value) == "cpu") return true;
        if(string(value) != "0")
            throw invalid_argument("INSTANTGR_TREE_CENTER must be 'cpu' or '0'");
        return false;
    }();
    return enabled;
}

int cpu_tree_center_min_degree() {
    static const int min_degree = [] {
        const char *value = getenv("INSTANTGR_TREE_CENTER_MIN_DEGREE");
        if(value == nullptr || *value == '\0') return GPU_FLUTE_MIN_DEGREE;
        char *end = nullptr;
        const long parsed = strtol(value, &end, 10);
        if(*end != '\0' || parsed < 2 || parsed > INT_MAX)
            throw invalid_argument("INSTANTGR_TREE_CENTER_MIN_DEGREE must be an integer >= 2");
        return static_cast<int>(parsed);
    }();
    return min_degree;
}

// The legacy-vs-center depth comparison and the BFS oracle for the peeled
// center cost extra full traversals per net, so they only run in profile runs.
bool tree_center_stats_enabled() {
    static const bool enabled = [] {
        const char *value = getenv("INSTANTGR_AUGMENTED_DAG_PROFILE");
        return value != nullptr && string(value) == "1";
    }();
    return enabled;
}

struct CpuTreeCenterProfile {
    atomic<long long> eligible_nets{0};
    atomic<long long> improved_nets{0};
    atomic<long long> legacy_depth_sum{0};
    atomic<long long> center_depth_sum{0};
    atomic<long long> bfs_nanoseconds{0};
};

CpuTreeCenterProfile cpu_tree_center_profile;

void reset_cpu_tree_center_profile() {
    cpu_tree_center_profile.eligible_nets.store(0);
    cpu_tree_center_profile.improved_nets.store(0);
    cpu_tree_center_profile.legacy_depth_sum.store(0);
    cpu_tree_center_profile.center_depth_sum.store(0);
    cpu_tree_center_profile.bfs_nanoseconds.store(0);
}

void print_cpu_tree_center_profile(const char *stage) {
    if(!cpu_tree_center_enabled()) return;
    const long long eligible = cpu_tree_center_profile.eligible_nets.load();
    if(eligible == 0) {
        printf("%s CPU tree-center: eligible=0 (degree >= %d)\n",
               stage, cpu_tree_center_min_degree());
        return;
    }
    const double find_seconds = cpu_tree_center_profile.bfs_nanoseconds.load() / 1e9;
    if(!tree_center_stats_enabled()) {
        printf("%s CPU tree-center profile: eligible=%lld, aggregate center-find=%.3fs\n",
               stage, eligible, find_seconds);
        return;
    }
    const long long improved = cpu_tree_center_profile.improved_nets.load();
    const long long legacy_sum = cpu_tree_center_profile.legacy_depth_sum.load();
    const long long center_sum = cpu_tree_center_profile.center_depth_sum.load();
    const double reduction = legacy_sum == 0 ? 0.0 :
        100.0 * static_cast<double>(legacy_sum - center_sum) / legacy_sum;
    printf("%s CPU tree-center profile: eligible=%lld, improved=%lld (%.1f%%), "
           "depth sum=%lld -> %lld (%.1f%% reduction), aggregate center-find=%.3fs\n",
           stage, eligible, improved, 100.0 * improved / eligible,
           legacy_sum, center_sum, reduction, find_seconds);
}

// Set INSTANTGR_RSMT_DEPTH_PROFILE=1 to write rsmt_depth_profile.csv, or set
// it to an explicit CSV path.  Keep the default filter at degree 10 because
// profiling every small net in a large benchmark can produce a huge file.
const char *rsmt_depth_profile_path() {
    const char *value = getenv("INSTANTGR_RSMT_DEPTH_PROFILE");
    if(value == nullptr || *value == '\0' || string(value) == "0") return nullptr;
    return string(value) == "1" ? "rsmt_depth_profile.csv" : value;
}

int rsmt_depth_profile_min_pins() {
    static const int min_pins = [] {
        const char *value = getenv("INSTANTGR_RSMT_DEPTH_PROFILE_MIN_PINS");
        if(value == nullptr || *value == '\0') return 10;
        char *end = nullptr;
        const long parsed = strtol(value, &end, 10);
        if(*end != '\0' || parsed < 1 || parsed > INT_MAX)
            throw invalid_argument("INSTANTGR_RSMT_DEPTH_PROFILE_MIN_PINS must be an integer >= 1");
        return static_cast<int>(parsed);
    }();
    return min_pins;
}


void net::calc_hpwl() {
    minx = X, maxx = 0, miny = Y, maxy = 0;
    for(auto p : pins) {
        int x = p / Y % X, y = p % Y;
        minx = min(minx, x);
        maxx = max(maxx, x);
        miny = min(miny, y);
        maxy = max(maxy, y);
    }
    hpwl = maxx - minx + maxy - miny;
}


// Set INSTANTGR_GPU_BATCH_GEN=0 to keep the sequential first-fit batch
// generation below instead of the journal's GPU algorithm.
bool gpu_batch_gen_enabled() {
    const char *value = getenv("INSTANTGR_GPU_BATCH_GEN");
    return value == nullptr || string(value) != "0";
}

// INSTANTGR_GPU_BATCH_GEN_VALIDATE=1 replays the batches to confirm that no
// net collides with a higher-priority member of its own batch, and reports how
// far the GPU partition drifted from the CPU one.  Slow: it runs both paths.
bool gpu_batch_gen_validate() {
    const char *value = getenv("INSTANTGR_GPU_BATCH_GEN_VALIDATE");
    return value != nullptr && string(value) == "1";
}

vector<vector<int>> generate_batches_rsmt_cpu(vector<int> &nets2route, int MAX_BATCH_SIZE) {
    vector<vector<int>> batches;
    vector<vector<bool>> batch_vis;

    auto has_conflict = [&] (int net_id, int batch_id) {
        for(auto p : nets[net_id].points) if(batch_vis[batch_id][p]) return true;
        return false;
    };

    auto mark_3x3 = [&] (int pos, int batch_id) {
        int _x = pos / Y, _y = pos % Y;
        for(int x = _x - 1; x <= _x + 1; x++) if(0 <= x && x < X)
            for(int y = _y - 1; y < _y + 1; y++) if(0 <= y && y < Y)
                if(x == _x || y == _y) batch_vis[batch_id][x * Y + y] = 1;
    };


    long long segment_len = 0, segment_cnt = 0, failed = 0;
    for(auto net_id : nets2route) {
        int batch_id = -1;
        for(int i = 0; i < batches.size(); i++) if(batches[i].size() < MAX_BATCH_SIZE)
        {
            if(!has_conflict(net_id, i)) { batch_id = i; break; }
            else failed++;
        }
        if(batch_id == -1) {
            batch_id = batches.size();
            batches.emplace_back(vector<int> ());
            batch_vis.emplace_back(vector<bool> (X * Y, 0));
        }
        batches[batch_id].emplace_back(net_id);
        for(auto seg : nets[net_id].rsmt_h_segments) {
            segment_len += seg.second / Y - seg.first / Y;
            segment_cnt++;
            for(auto p = seg.first; p <= seg.second; p += Y) batch_vis[batch_id][p] = 1;
        }
        for(auto seg : nets[net_id].rsmt_v_segments) {
            for(auto p = seg.first; p <= seg.second; p += 1) batch_vis[batch_id][p] = 1;
        }
        for(auto p : nets[net_id].points) mark_3x3(p, batch_id);
    }
    return move(batches);
}

// Flatten every net's marks into the CSR layout the GPU kernels read.  A net's
// priority is its position in nets2route, the order the CPU path walks, so
// both paths hand a contested cell to the same net.
gpu_batch::HostMarks collect_batch_marks(const vector<int> &nets2route) {
    const int net_cnt = nets2route.size();
    gpu_batch::HostMarks marks;
    marks.h_off.resize(net_cnt + 1, 0);
    marks.v_off.resize(net_cnt + 1, 0);
    marks.p_off.resize(net_cnt + 1, 0);
    for(int i = 0; i < net_cnt; i++) {
        const auto &target = nets[nets2route[i]];
        marks.h_off[i + 1] = marks.h_off[i] + target.rsmt_h_segments.size();
        marks.v_off[i + 1] = marks.v_off[i] + target.rsmt_v_segments.size();
        marks.p_off[i + 1] = marks.p_off[i] + target.points.size();
    }
    marks.h_lo.resize(marks.h_off[net_cnt]);
    marks.h_hi.resize(marks.h_off[net_cnt]);
    marks.v_lo.resize(marks.v_off[net_cnt]);
    marks.v_hi.resize(marks.v_off[net_cnt]);
    marks.p_pos.resize(marks.p_off[net_cnt]);
    const int worker_cnt = 8;
    vector<thread> workers;
    for(int w = 0; w < worker_cnt; w++)
        workers.emplace_back([&marks, &nets2route, net_cnt, w, worker_cnt] {
            for(int i = w; i < net_cnt; i += worker_cnt) {
                const auto &target = nets[nets2route[i]];
                int at = marks.h_off[i];
                for(auto seg : target.rsmt_h_segments) {
                    marks.h_lo[at] = seg.first;
                    marks.h_hi[at] = seg.second;
                    at++;
                }
                at = marks.v_off[i];
                for(auto seg : target.rsmt_v_segments) {
                    marks.v_lo[at] = seg.first;
                    marks.v_hi[at] = seg.second;
                    at++;
                }
                at = marks.p_off[i];
                for(auto pos : target.points) marks.p_pos[at++] = pos;
            }
        });
    for(auto &worker : workers) worker.join();
    return marks;
}

// The invariant the router relies on, checked by replaying a batch in priority
// order: no net's point may be covered by a member placed before it.
bool batches_are_conflict_free(const vector<vector<int>> &batches) {
    vector<bool> vis(X * Y, false);
    vector<int> touched;
    auto mark = [&] (int pos) {
        if(!vis[pos]) {
            vis[pos] = true;
            touched.emplace_back(pos);
        }
    };
    bool ok = true;
    for(int batch_id = 0; batch_id < batches.size() && ok; batch_id++) {
        touched.clear();
        for(auto net_id : batches[batch_id]) {
            for(auto pos : nets[net_id].points) if(vis[pos]) {
                printf("[gpu-batch-gen] batch %d: net %d collides at cell %d\n", batch_id, net_id, pos);
                ok = false;
                break;
            }
            if(!ok) break;
            for(auto seg : nets[net_id].rsmt_h_segments)
                for(auto pos = seg.first; pos <= seg.second; pos += Y) mark(pos);
            for(auto seg : nets[net_id].rsmt_v_segments)
                for(auto pos = seg.first; pos <= seg.second; pos += 1) mark(pos);
            for(auto pos : nets[net_id].points) {
                const int x = pos / Y, y = pos % Y;
                if(y > 0) mark(pos - 1);
                mark(pos);
                if(x > 0) mark(pos - Y);
                if(x + 1 < X) mark(pos + Y);
            }
        }
        for(auto pos : touched) vis[pos] = false;
    }
    return ok;
}

vector<vector<int>> generate_batches_rsmt(vector<int> &nets2route, int MAX_BATCH_SIZE = 1000000) {
    auto _time = elapsed_time();
    vector<vector<int>> batches;
    bool on_gpu = false;
    gpu_batch::Result gpu_result;
    double collect_seconds = 0;

    if(gpu_batch_gen_enabled()) try {
        const double collect_start = elapsed_time();
        auto marks = collect_batch_marks(nets2route);
        collect_seconds = elapsed_time() - collect_start;
        gpu_batch::Result &result = gpu_result;
        if(gpu_batch::generate(marks, nets2route.size(), X, Y, MAX_BATCH_SIZE, result)) {
            vector<int> batch_size(result.batch_count, 0);
            for(auto batch_id : result.batch_of_net)
                if(batch_id >= 0 && batch_id < result.batch_count) batch_size[batch_id]++;
            int placed = 0;
            for(auto size : batch_size) placed += size;
            if(placed != (int)nets2route.size()) {
                printf("[gpu-batch-gen] %d of %zu nets were left unplaced; falling back to the CPU path\n",
                       (int)nets2route.size() - placed, nets2route.size());
            } else {
                // A batch can come out empty when the round that opened it lost
                // every candidate to a net in another batch; the router should
                // not be handed an empty batch to launch kernels for.
                vector<int> batch_index(result.batch_count, -1);
                for(int i = 0; i < result.batch_count; i++) if(batch_size[i] > 0) {
                    batch_index[i] = batches.size();
                    batches.emplace_back(vector<int> ());
                    batches.back().reserve(batch_size[i]);
                }
                for(int i = 0; i < nets2route.size(); i++)
                    batches[batch_index[result.batch_of_net[i]]].emplace_back(nets2route[i]);
                on_gpu = true;
            }
        }
    } catch(const std::exception &error) {
        // The flattened marks of a large design are a sizable host allocation;
        // running out of memory here is a reason to fall back, not to die.
        printf("[gpu-batch-gen] %s; falling back to the CPU path\n", error.what());
        batches.clear();
        on_gpu = false;
    }
    if(!on_gpu) batches = generate_batches_rsmt_cpu(nets2route, MAX_BATCH_SIZE);

    _time = elapsed_time() - _time;
    if(LOG) cout << setw(40) << "Batch" << setw(20) << "#Nets" << setw(20) << "#Batches" << setw(20) << "Time" << endl;
    if(LOG) cout << setw(40) << (on_gpu ? "Generation (GPU)" : "Generation (CPU)") << setw(20) << nets2route.size() << setw(20) << batches.size() << setw(20) << setprecision(2) << _time << endl;
    if(LOG && on_gpu)
        printf("        rounds %d, %.1f commits per net, wavefront %d, %d batches open at once (%d retired early); mark collection %.3fs\n",
               gpu_result.rounds, (double) gpu_result.commits / max<size_t>(1, nets2route.size()),
               gpu_result.wavefront, gpu_result.ring_slots, gpu_result.retired, collect_seconds);

    if(gpu_batch_gen_validate()) {
        const bool ok = batches_are_conflict_free(batches);
        printf("[gpu-batch-gen] conflict-free: %s\n", ok ? "yes" : "NO");
        if(on_gpu) {
            auto reference = generate_batches_rsmt_cpu(nets2route, MAX_BATCH_SIZE);
            vector<int> cpu_batch_of_net(nets.size(), -1);
            for(int i = 0; i < reference.size(); i++)
                for(auto net_id : reference[i]) cpu_batch_of_net[net_id] = i;
            long long moved = 0;
            for(int i = 0; i < batches.size(); i++)
                for(auto net_id : batches[i]) if(cpu_batch_of_net[net_id] != i) moved++;
            printf("[gpu-batch-gen] batches GPU %zu vs CPU %zu, %lld of %zu nets in a different batch\n",
                   batches.size(), reference.size(), moved, nets2route.size());
        }
    }
    return move(batches);
}

double TOT_RSMT_LENGTH = 0;
vector<vector<int>> my_flute(unordered_set<int> &pos) {
    const int MAX_DEGREE = 10000;
    vector<int> x;
    vector<int> y;
    int cnt = 0;
    vector<int> nodes, parent;
    vector<tuple<int, int, int>> edges;
    for(auto e : pos) {
        x.push_back(db::dr_x[e / Y]);
        y.push_back(db::dr_y[e % Y]);
        cnt++;
    }
    auto tree = flute::flute(cnt, x.data(), y.data(), 3);
    for(int i = 0; i < cnt * 2 - 2; i++) {
        flute::Branch &branch = tree.branch[i];
        nodes.emplace_back(db::dr2gr_x[branch.x] * Y + db::dr2gr_y[branch.y]);
    }
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    parent.resize(nodes.size());
    for(int i = 0; i < nodes.size(); i++) parent[i] = i;
    edges.reserve(cnt * 2);
    for(int i = 0; i < cnt * 2 - 2; i++) if(tree.branch[i].n < cnt * 2 - 2) {
        Branch &branch1 = tree.branch[i], &branch2 = tree.branch[branch1.n];
        int u, v;
        u = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch1.x] * Y + db::dr2gr_y[branch1.y]) - nodes.begin();
        v = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch2.x] * Y + db::dr2gr_y[branch2.y]) - nodes.begin();
        if(u == v) continue;
        edges.emplace_back(make_tuple(abs(branch1.x - branch2.x) + abs(branch1.y - branch2.y), u, v));
            
    }
    sort(edges.begin(), edges.end());
    function<int(int)> find_parent = [&] (int x) { return x == parent[x] ? x : parent[x] = find_parent(parent[x]); };
    vector<vector<int>> graph(nodes.size());
    for(auto edge : edges) {
        int u = get<1> (edge), v = get<2> (edge), par_u = find_parent(u), par_v = find_parent(v);
        if(par_u != par_v) {
            graph[u].emplace_back(v);
            graph[v].emplace_back(u);
            TOT_RSMT_LENGTH += get<0> (edge);
            parent[par_u] = par_v;
        }
    }
    int tot_degree = 0;
    for(int i = 0; i < nodes.size(); i++) tot_degree += graph[i].size();
    assert(tot_degree == 2 * (nodes.size() - 1));
    graph.emplace_back(move(nodes));
    return move(graph);
}

vector<vector<int>> gpu_flute_tree_to_graph(const gpu_flute::Branch *branches, int degree) {
    vector<int> nodes, parent;
    vector<tuple<int, int, int>> edges;
    nodes.reserve(2 * degree - 2);
    edges.reserve(2 * degree - 2);
    for(int i = 0; i < 2 * degree - 2; ++i) {
        const auto &branch = branches[i];
        nodes.emplace_back(db::dr2gr_x[branch.x] * Y + db::dr2gr_y[branch.y]);
    }
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    parent.resize(nodes.size());
    for(int i = 0; i < nodes.size(); ++i) parent[i] = i;
    for(int i = 0; i < 2 * degree - 2; ++i) {
        const auto &a = branches[i];
        if(a.n >= 2 * degree - 2) continue;
        const auto &b = branches[a.n];
        int u = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[a.x] * Y + db::dr2gr_y[a.y]) - nodes.begin();
        int v = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[b.x] * Y + db::dr2gr_y[b.y]) - nodes.begin();
        if(u != v) edges.emplace_back(abs(a.x - b.x) + abs(a.y - b.y), u, v);
    }
    sort(edges.begin(), edges.end());
    function<int(int)> find_parent = [&] (int x) { return x == parent[x] ? x : parent[x] = find_parent(parent[x]); };
    vector<vector<int>> graph(nodes.size());
    for(auto edge : edges) {
        int u = get<1>(edge), v = get<2>(edge);
        int pu = find_parent(u), pv = find_parent(v);
        if(pu == pv) continue;
        graph[u].emplace_back(v);
        graph[v].emplace_back(u);
        TOT_RSMT_LENGTH += get<0>(edge);
        parent[pu] = pv;
    }
    int total_degree = 0;
    for(const auto &adj : graph) total_degree += adj.size();
    assert(total_degree == 2 * (int(nodes.size()) - 1));
    graph.emplace_back(move(nodes));
    return graph;
}

long long rsmt_wirelength(const vector<vector<int>> &tree) {
    const vector<int> &nodes = tree.back();
    long long length = 0;
    for(int i = 0; i + 1 < tree.size(); ++i)
        for(int j : tree[i]) if(j < i) {
            int xi = nodes[i] / Y, yi = nodes[i] % Y;
            int xj = nodes[j] / Y, yj = nodes[j] % Y;
            length += abs(db::dr_x[xi] - db::dr_x[xj]) + abs(db::dr_y[yi] - db::dr_y[yj]);
        }
    return length;
}

void make_cpu_hanan(const net &target, vector<int> &xs, vector<int> &ys, vector<int> &s) {
    struct point { int x, y, rank; };
    vector<point> points;
    points.reserve(target.pins.size());
    for(int pin : target.pins)
        points.push_back({db::dr_x[(pin / Y) % X], db::dr_y[pin % Y], -1});

    // Match flute::flute() exactly: x-selection sort uses y descending only
    // for equal x; the subsequent y-selection sort has no tie-break.
    for(int i = 0; i + 1 < points.size(); ++i) {
        int best = i;
        for(int j = i + 1; j < points.size(); ++j)
            if(points[j].x < points[best].x ||
               (points[j].x == points[best].x && points[j].y > points[best].y))
                best = j;
        swap(points[i], points[best]);
    }
    xs.resize(points.size());
    for(int i = 0; i < points.size(); ++i) {
        xs[i] = points[i].x;
        points[i].rank = i;
    }
    ys.resize(points.size());
    s.resize(points.size());
    for(int i = 0; i + 1 < points.size(); ++i) {
        int best = i;
        for(int j = i + 1; j < points.size(); ++j)
            if(points[j].y < points[best].y) best = j;
        ys[i] = points[best].y;
        s[i] = points[best].rank;
        swap(points[i], points[best]);
    }
    ys.back() = points.back().y;
    s.back() = points.back().rank;
}

void net::finalize_rsmt(const unordered_map<int, int> &layer) {
    rsmt_h_segments.clear();
    rsmt_v_segments.clear();
    rsmt_h_segments.reserve(rsmt.size());
    rsmt_v_segments.reserve(rsmt.size());
    points = rsmt.back();
    for(int i = 0; i < rsmt.back().size(); i++) {
        int xi = rsmt.back()[i] / Y, yi = rsmt.back()[i] % Y;
        for(auto j : rsmt[i]) if(j < i) {
            int xj = rsmt.back()[j] / Y, yj = rsmt.back()[j] % Y;
            int minx = min(xi, xj), maxx = max(xi, xj), miny = min(yi, yj), maxy = max(yi, yj);
            if(xi != xj && yi != yj) {
                rsmt_h_segments.emplace_back(minx * Y + miny, maxx * Y + miny);
                rsmt_h_segments.emplace_back(minx * Y + maxy, maxx * Y + maxy);
                rsmt_v_segments.emplace_back(minx * Y + miny, minx * Y + maxy);
                rsmt_v_segments.emplace_back(maxx * Y + miny, maxx * Y + maxy);
                points.emplace_back(xi * Y + yj);
                points.emplace_back(xj * Y + yi);
            } else if(xi != xj) {
                rsmt_h_segments.emplace_back(minx * Y + miny, maxx * Y + miny);
            } else if(yi != yj) {
                rsmt_v_segments.emplace_back(minx * Y + miny, minx * Y + maxy);
            } else {
                cerr << "error" << endl;
            }
        }
    }
    for(auto &e : rsmt.back()) e += (layer.count(e) ? layer.at(e) : L) * X * Y;
}

void net::construct_rsmt() {
    unordered_map<int, int> layer;
    unordered_set<int> pos, nodes;
    for(int i = 0; i < pins.size(); i++) {
        int pos2D = pins[i] % (X * Y);
        pos.insert(pos2D);
        layer[pos2D] = pins[i] / X / Y;
    }
    assert(pos.size() == pins.size());

    rsmt = my_flute(pos);
    finalize_rsmt(layer);
}

// Root nets above the FLUTE LUT degree are processed as one GPU-FLUTE batch.
// This intentionally leaves net::construct_rsmt() untouched for low-degree
// roots, so callers can tune the CPU/GPU threshold without losing the original
// CPU implementation.
void construct_rsmt_gpu(vector<int> &net_ids) {
    if(net_ids.empty()) return;
    const bool validate = gpu_flute_validation_enabled();
    const bool profile = gpu_flute_profile_enabled();
    const bool use_gpu_tree_center = gpu_tree_center_enabled();
    const double solve_start_time = elapsed_time();
    auto result = gpu_flute::solve_high_degree(net_ids, pin_cnt_sum_cpu,
                                                pin_acc_num, pins,
                                                flute_x_coords, flute_y_coords,
                                                X, Y, 3, validate, profile,
                                                use_gpu_tree_center);
    const double solve_end_time = elapsed_time();
    assert(result.net_ids.size() == result.tree_offsets.size() - 1);
    atomic<int> mismatch_count(0);
    atomic<int> hanan_mismatch_count(0);
    atomic<int> gpu_tree_center_selected(0);
    atomic<int> gpu_tree_center_fallback_count(0);
    // The device solve is a fraction of a second even with every root net on
    // the GPU; this host rebuild (tree -> graph -> finalize_rsmt) is the bulk
    // of construct_rsmt_gpu, so it is spread over 8 threads like the CPU
    // FLUTE loop.  Nets are disjoint; the counters above are atomic.
    auto rebuild = [&](int i) {
        net &target = nets[result.net_ids[i]];
        unordered_map<int, int> layer;
        for(int pin : target.pins) {
            int pos2D = pin % (X * Y);
            layer[pos2D] = pin / X / Y;
        }
        const int degree = target.pins.size();
        const int branch_count = result.tree_offsets[i + 1] - result.tree_offsets[i];
        assert(branch_count == 2 * degree - 2);
        target.rsmt = gpu_flute_tree_to_graph(result.trees.data() + result.tree_offsets[i], degree);
        target.gpu_tree_center_pos = -1;
        target.gpu_tree_center_fallback = false;
        if(use_gpu_tree_center) {
            assert(result.tree_centers.size() == result.net_ids.size());
            const auto center = result.tree_centers[i];
            if(center.x < 0) {
                target.gpu_tree_center_fallback = true;
                ++gpu_tree_center_fallback_count;
            } else {
                target.gpu_tree_center_pos = db::dr2gr_x[center.x] * Y + db::dr2gr_y[center.y];
                bool found_center = false;
                for(int point : target.rsmt.back())
                    found_center |= point == target.gpu_tree_center_pos;
                assert(found_center);
                ++gpu_tree_center_selected;
            }
        }
        if(validate) {
            vector<int> cpu_xs, cpu_ys, cpu_s;
            make_cpu_hanan(target, cpu_xs, cpu_ys, cpu_s);
            const int pin_offset = pin_cnt_sum_cpu[result.net_ids[i]];
            bool hanan_match = true;
            for(int j = 0; j < degree; ++j)
                if(cpu_xs[j] != result.root_xs[pin_offset + j] ||
                   cpu_ys[j] != result.root_ys[pin_offset + j] ||
                   cpu_s[j] != result.root_s[pin_offset + j]) {
                    hanan_match = false;
                    break;
                }
            if(!hanan_match) {
                if(++hanan_mismatch_count <= 16) {
                    printf("GPU-FLUTE Hanan mismatch: net=%d degree=%d\n",
                           result.net_ids[i], degree);
                    for(int j = 0; j < degree; ++j)
                        if(cpu_xs[j] != result.root_xs[pin_offset + j] ||
                           cpu_ys[j] != result.root_ys[pin_offset + j] ||
                           cpu_s[j] != result.root_s[pin_offset + j]) {
                            printf("  idx=%d cpu=(%d,%d,s=%d) gpu=(%d,%d,s=%d)\n",
                                   j, cpu_xs[j], cpu_ys[j], cpu_s[j],
                                   result.root_xs[pin_offset + j],
                                   result.root_ys[pin_offset + j],
                                   result.root_s[pin_offset + j]);
                            break;
                        }
                }
            }
            unordered_set<int> pos;
            for(int pin : target.pins) pos.insert(pin % (X * Y));
            const auto cpu_tree = my_flute(pos);
            const long long gpu_length = rsmt_wirelength(target.rsmt);
            const long long cpu_length = rsmt_wirelength(cpu_tree);
            if(gpu_length != cpu_length) {
                if(++mismatch_count <= 16)
                    printf("GPU-FLUTE mismatch: net=%d degree=%d gpu=%lld cpu=%lld\n",
                           result.net_ids[i], degree, gpu_length, cpu_length);
            }
        }
        target.finalize_rsmt(layer);
    };
    {
        const int net_count = result.net_ids.size();
        atomic<int> next_chunk(0);
        vector<thread> workers;
        for(int t = 0; t < 8; ++t)
            workers.emplace_back([&] {
                constexpr int CHUNK = 1024;
                for(int begin = next_chunk.fetch_add(CHUNK); begin < net_count;
                    begin = next_chunk.fetch_add(CHUNK))
                    for(int i = begin; i < min(begin + CHUNK, net_count); ++i) rebuild(i);
            });
        for(auto &worker : workers) worker.join();
    }
    if(validate)
        printf("GPU-FLUTE validation: Hanan=%d WL=%d / %zu high-degree nets\n",
               hanan_mismatch_count.load(), mismatch_count.load(), result.net_ids.size());
    if(use_gpu_tree_center)
        printf("GPU tree-center: selected=%d, legacy fallback=%d / %zu high-degree nets\n",
               gpu_tree_center_selected.load(), gpu_tree_center_fallback_count.load(), result.net_ids.size());
    if(profile) {
        const double rebuild_time = elapsed_time() - solve_end_time;
        printf("GPU-FLUTE profile: nets=%zu, degree >= %d\n",
               result.net_ids.size(), gpu_flute_min_degree());
        printf("  solve wall time: %.3fs, host tree rebuild: %.3fs\n",
               result.profile.host_wall_seconds, rebuild_time);
        const double solve_wall = result.profile.host_wall_seconds;
        printf("  explicit H2D API time: %.3fs (%zu bytes, %.1f%% of solve wall)\n",
               result.profile.h2d_api_seconds, result.profile.h2d_bytes,
               solve_wall > 0 ? result.profile.h2d_api_seconds / solve_wall * 100 : 0.0);
        printf("  explicit D2H API time: %.3fs (%zu bytes, %.1f%% of solve wall)\n",
               result.profile.d2h_api_seconds, result.profile.d2h_bytes,
               solve_wall > 0 ? result.profile.d2h_api_seconds / solve_wall * 100 : 0.0);
        printf("  GPU device pipeline event time: %.3fs (%.1f%% of solve wall)\n",
               result.profile.device_pipeline_seconds,
               solve_wall > 0 ? result.profile.device_pipeline_seconds / solve_wall * 100 : 0.0);
        if(use_gpu_tree_center)
            printf("  GPU tree-center pass: %.3fs (%.1f%% of solve wall)\n",
                   result.profile.tree_center_device_seconds,
                   solve_wall > 0 ? result.profile.tree_center_device_seconds / solve_wall * 100 : 0.0);
        printf("  construct_rsmt_gpu total: %.3fs\n", elapsed_time() - solve_start_time);
    }
}


typedef unsigned int BITSET_TYPE;
const int BITSET_LEN = 32;

int select_root_net(const vector<vector<int>> &rsmt)
{
    queue<pair<int,int>> list;
    int visited[rsmt.size()-1];
    int select = -1;
    for(int i=0; i < rsmt.size()-1; i++)
    {
        visited[i]=0;
        if(rsmt[i].size()==1)
        {
            list.push(make_pair(i, -1));
        }
    }
    while(!list.empty())
    {
        pair<int,int> front_element = list.front();
        list.pop();
        select = front_element.first;
        if(visited[select]) continue;
        visited[select]=1;
        int fa = front_element.second;
        for(int j = 0; j< rsmt[select].size(); j++)
        {
            if(rsmt[select][j]!=fa)
            {
                list.push(make_pair(rsmt[select][j], select));
            }
        }
    }
    return select;
}

int rsmt_node_for_2d_position(const vector<vector<int>> &rsmt, int position_2d) {
    for(int i = 0; i + 1 < rsmt.size(); ++i)
        if(rsmt.back()[i] % (X * Y) == position_2d) return i;
    return -1;
}

struct RsmtDepthStats {
    int current_root;
    int current_max_depth;
    int min_max_depth;
    int diameter;
    int center0;
    int center1;
};

pair<int, vector<int>> rsmt_bfs_dist(const vector<vector<int>> &rsmt, int start) {
    const int node_count = rsmt.size() - 1;
    vector<int> dist(node_count, -1);
    queue<int> work;
    dist[start] = 0;
    work.push(start);
    while(!work.empty()) {
        const int u = work.front();
        work.pop();
        for(int v : rsmt[u]) if(dist[v] == -1) {
            dist[v] = dist[u] + 1;
            work.push(v);
        }
    }
    int farthest = start;
    for(int i = 1; i < node_count; ++i)
        if(dist[i] > dist[farthest]) farthest = i;
    return {farthest, move(dist)};
}

RsmtDepthStats rsmt_depth_stats(const vector<vector<int>> &rsmt, int root_override = -1) {
    const int current_root = root_override >= 0 ? root_override : select_root_net(rsmt);
    auto [ignored, current_dist] = rsmt_bfs_dist(rsmt, current_root);
    const int current_max_depth = *max_element(current_dist.begin(), current_dist.end());

    // RSMT is a tree.  Its radius (the smallest possible maximum depth) is
    // ceil(diameter / 2); the middle node(s) of a diameter are the roots that
    // attain it.
    const int endpoint0 = rsmt_bfs_dist(rsmt, 0).first;
    auto [endpoint1, endpoint0_dist] = rsmt_bfs_dist(rsmt, endpoint0);
    const int diameter = endpoint0_dist[endpoint1];

    vector<int> parent(rsmt.size() - 1, -1);
    queue<int> work;
    parent[endpoint0] = endpoint0;
    work.push(endpoint0);
    while(!work.empty()) {
        const int u = work.front();
        work.pop();
        for(int v : rsmt[u]) if(parent[v] == -1) {
            parent[v] = u;
            work.push(v);
        }
    }
    vector<int> path;
    for(int u = endpoint1;; u = parent[u]) {
        path.emplace_back(u);
        if(u == endpoint0) break;
    }
    const int center0 = path[diameter / 2];
    const int center1 = (diameter & 1) ? path[diameter / 2 + 1] : center0;
    return {current_root, current_max_depth, (diameter + 1) / 2,
            diameter, center0, center1};
}

// Exact tree center by leaf peeling: strip the current leaf layer until the
// tree is exhausted; the last node enqueued is a center (either one when the
// tree has two).  Peeling touches each edge once, so unlike the double BFS in
// rsmt_depth_stats() this is a single traversal, and the thread-local scratch
// avoids the per-call heap allocations that dominated the old cost.
int rsmt_tree_center_peel(const vector<vector<int>> &rsmt) {
    const int node_count = (int) rsmt.size() - 1;
    if(node_count <= 2) return 0;
    thread_local vector<int> degree, order;
    if((int) degree.size() < node_count) {
        degree.resize(node_count);
        order.resize(node_count);
    }
    int tail = 0;
    for(int i = 0; i < node_count; ++i) {
        degree[i] = (int) rsmt[i].size();
        if(degree[i] == 1) order[tail++] = i;
    }
    for(int head = 0; head < tail; ++head) {
        const int u = order[head];
        for(int v : rsmt[u]) if(--degree[v] == 1) order[tail++] = v;
    }
    return order[tail - 1];
}

// Return the exact tree center of the host RSMT via leaf peeling.  In profile
// runs, also account for its depth benefit relative to the root used by the
// caller (node 0 in Stage 1; legacy_root < 0 lets the stats pass derive the
// legacy select_root_net() root itself) and check the peeled center against
// the double-BFS radius oracle.
int cpu_tree_center_root(const net &target, int legacy_root) {
    static once_flag cpu_tree_center_notice;
    call_once(cpu_tree_center_notice, [] {
        printf("CPU tree-center: enabled for nets with degree >= %d (exact host RSMT, leaf peeling)\n",
               cpu_tree_center_min_degree());
    });
    const auto find_start = chrono::steady_clock::now();
    const int center = rsmt_tree_center_peel(target.rsmt);
    const auto find_end = chrono::steady_clock::now();
    cpu_tree_center_profile.eligible_nets.fetch_add(1, memory_order_relaxed);
    cpu_tree_center_profile.bfs_nanoseconds.fetch_add(
        chrono::duration_cast<chrono::nanoseconds>(find_end - find_start).count(),
        memory_order_relaxed);
    if(tree_center_stats_enabled()) {
        const auto legacy_stats = rsmt_depth_stats(target.rsmt, legacy_root);
        const auto center_bfs = rsmt_bfs_dist(target.rsmt, center);
        const int center_depth = center_bfs.second[center_bfs.first];
        if(center_depth != legacy_stats.min_max_depth)
            printf("tree-center peel mismatch: original_net=%d peeled depth=%d radius=%d\n",
                   target.original_net_id, center_depth, legacy_stats.min_max_depth);
        cpu_tree_center_profile.improved_nets.fetch_add(
            legacy_stats.current_max_depth > center_depth, memory_order_relaxed);
        cpu_tree_center_profile.legacy_depth_sum.fetch_add(
            legacy_stats.current_max_depth, memory_order_relaxed);
        cpu_tree_center_profile.center_depth_sum.fetch_add(
            center_depth, memory_order_relaxed);
    }
    return center;
}

void write_rsmt_depth_profile(const vector<int> &nets2route) {
    const char *path = rsmt_depth_profile_path();
    if(path == nullptr) return;
    const int min_pins = rsmt_depth_profile_min_pins();
    ofstream out(path);
    if(!out) throw runtime_error(string("cannot open RSMT depth profile: ") + path);
    out << "subnet_id,original_net_id,net_name,root_policy,pin_degree,rsmt_nodes,"
           "current_root_x,current_root_y,current_max_depth,diameter,"
           "min_max_depth,center0_x,center0_y,center1_x,center1_y,depth_gap\n";

    int written = 0, improved = 0, max_gap = 0;
    for(int subnet_id : nets2route) {
        const auto &target = nets[subnet_id];
        if(target.pins.size() < min_pins) continue;
        const int override_root = target.gpu_tree_center_pos < 0 ? -1 :
                                  rsmt_node_for_2d_position(target.rsmt, target.gpu_tree_center_pos);
        assert(target.gpu_tree_center_pos < 0 || override_root >= 0);
        const auto stats = rsmt_depth_stats(target.rsmt, override_root);
        const auto &nodes = target.rsmt.back();
        const auto xy = [] (int point) { return pair<int, int>{point / Y % X, point % Y}; };
        const auto [root_x, root_y] = xy(nodes[stats.current_root]);
        const auto [center0_x, center0_y] = xy(nodes[stats.center0]);
        const auto [center1_x, center1_y] = xy(nodes[stats.center1]);
        const int gap = stats.current_max_depth - stats.min_max_depth;
        out << subnet_id << ',' << target.original_net_id << ','
            << '"' << db::nets[target.original_net_id].name << '"' << ','
            << (override_root >= 0 ? "gpu_tree_center" :
                (target.gpu_tree_center_fallback ? "gpu_tree_center_fallback" : "legacy_bfs")) << ','
            << target.pins.size() << ',' << nodes.size() << ','
            << root_x << ',' << root_y << ',' << stats.current_max_depth << ','
            << stats.diameter << ',' << stats.min_max_depth << ','
            << center0_x << ',' << center0_y << ','
            << center1_x << ',' << center1_y << ',' << gap << '\n';
        ++written;
        improved += gap > 0;
        max_gap = max(max_gap, gap);
    }
    printf("RSMT depth profile: %d nets (degree >= %d), %d improvable, max gap=%d, file=%s\n",
           written, min_pins, improved, max_gap, path);
}

void net::generate_detours(bool *congestionView_cpu, float *congestionView_xsum_cpu, float *congestionView_ysum_cpu,
                             bool construct_segments, bool display) {
    
    auto graph_x = rsmt;
    int node_estimate = (graph_x.size()-1)*10;

    par_num_cpu.clear();
    par_num_sum_cpu.clear();
    currentChildIDX_cpu.clear();
    par_nodes_cpu.clear();
    child_num_cpu.clear();
    node_depth_cpu.clear();
    nodes_cpu.clear();
    node_index_cnt = 0;

    par_num_cpu.reserve(node_estimate);
    par_num_sum_cpu.reserve(node_estimate);
    currentChildIDX_cpu.reserve(node_estimate);
    par_nodes_cpu.reserve(node_estimate);
    child_num_cpu.reserve(node_estimate);
    node_depth_cpu.reserve(node_estimate);
    nodes_cpu.reserve(node_estimate);

    par_num_cpu.emplace_back(0);
    par_num_sum_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    par_nodes_cpu.emplace_back(0);
    child_num_cpu.emplace_back(0);
    node_depth_cpu.emplace_back(0);
    nodes_cpu.emplace_back(0);
    
    int depth_max = 0;
    // CPU center takes precedence deliberately: it is the correctness oracle
    // for this experiment and lets us measure root-depth impact independently
    // of the currently unverified GPU graph reconstruction.
    if(cpu_tree_center_enabled() && pins.size() >= cpu_tree_center_min_degree()) {
        select_root = cpu_tree_center_root(*this, -1);
    } else {
        select_root = select_root_net(graph_x);
        if(gpu_tree_center_pos >= 0) {
            select_root = rsmt_node_for_2d_position(graph_x, gpu_tree_center_pos);
            assert(select_root >= 0);
            // Debug-only oracle: the GPU-selected coordinate must attain the CPU
            // tree radius on the reconstructed, coordinate-contracted RSMT.
            if(gpu_flute_validation_enabled()) {
                const auto gpu_stats = rsmt_depth_stats(graph_x, select_root);
                if(gpu_stats.current_max_depth != gpu_stats.min_max_depth) {
                    const auto &positions = graph_x.back();
                    const auto xy = [] (int point) { return pair<int, int>{point / Y % X, point % Y}; };
                    const auto [gpu_x, gpu_y] = xy(positions[select_root]);
                    const auto [center0_x, center0_y] = xy(positions[gpu_stats.center0]);
                    const auto [center1_x, center1_y] = xy(positions[gpu_stats.center1]);
                    printf("GPU tree-center mismatch: original_net=%d gpu=(%d,%d) depth=%d min=%d cpu_center=(%d,%d),(%d,%d); legacy fallback\n",
                           original_net_id, gpu_x, gpu_y, gpu_stats.current_max_depth,
                           gpu_stats.min_max_depth, center0_x, center0_y, center1_x, center1_y);
                    gpu_tree_center_pos = -1;
                    gpu_tree_center_fallback = true;
                    select_root = select_root_net(graph_x);
                }
            }
        }
    }
    vector<int> congestionRegionID[2];
    vector<vector<pair<int, int> >> congestionRanges;
    congestionRanges.resize(2);
    vector<vector<vector<int>>> stems;
    stems.resize(2);
    congestionRegionID[0].resize(graph_x.size());
    congestionRegionID[1].resize(graph_x.size());
    for(int g_i = 0; g_i < graph_x.size(); g_i++)
    {
        congestionRegionID[0][g_i] = -1;
        congestionRegionID[1][g_i] = -1;
    }
    congestionRanges[0].resize(graph_x.size());
    congestionRanges[1].resize(graph_x.size());
    stems[0].resize(graph_x.size());
    stems[1].resize(graph_x.size());
    function<int(int, int)> getRegionID = [&] (int x, int direction) {
        if(x==-1) return -1;
        if(congestionRegionID[direction][x] == -1) return -1;
        if(congestionRegionID[direction][x] != x)
        {
            if(congestionRegionID[direction][x] == x)
            {
                assert(0);
            }
            int ans = getRegionID(congestionRegionID[direction][x], direction);
            congestionRegionID[direction][x] = ans;
            return ans;
        }else{
            return congestionRegionID[direction][x];
        }
    };
    for(int x = 0; x < graph_x.back().size(); x++)
    {
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur / Y % X, cury = position_cur % Y;
        for(int dir=0; dir<2; dir++)
        {
            if(curl<MAX_LAYER-1)
            {
                int stem_pos = dir?curx:cury;
                stems[dir][x].push_back(stem_pos);
            }
            int trunk_pos = dir?cury:curx;
            congestionRanges[dir][x] = make_pair(trunk_pos, trunk_pos);
        }
     
    }
    function<void(int, int)> markCongestion = [&] (int x, int par) {
        int position_cur = graph_x.back()[x];
        int curx = position_cur / Y % X, cury = position_cur % Y;
        for(auto e : graph_x[x]) if(e != par)
        {
            int ex = graph_x.back()[e] / Y % X;
            int ey = graph_x.back()[e] % Y;
            int dir=-1;
            int congestion = -1;
            if(ex == curx)
            {
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 1;
                }
            }
            else if(ey == cury)
            {
                bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0;
                if(is_congestion_x)
                {
                    dir = 0;
                    congestion = 1;
                }
            }
            else{
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0 || 
                                        (congestionView_ysum_cpu[ex*Y+max(ey, cury)]-congestionView_ysum_cpu[ex*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 2;
                }
                if(congestion==-1)
                {

                    bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0 || 
                                        (congestionView_xsum_cpu[max(ex, curx)*Y+ey]-congestionView_xsum_cpu[min(ex, curx)*Y+ey])>0;
                    if(is_congestion_x)
                    {
                        dir = 0;
                        congestion = 2;
                    }
                }
            }
            if(congestion == 1)
            {
                int target_region = -1;
                if(x!=select_root)
                {
                    if(congestionRegionID[dir][x]==-1)
                    {
                        congestionRegionID[dir][x] = x;
                    }
                    int region_x = getRegionID(x, dir);
                    target_region = region_x;
                }else{
                    if(congestionRegionID[dir][e]==-1)
                    {
                        congestionRegionID[dir][e] = e;
                    }
                    int region_e = getRegionID(e, dir);
                    target_region = region_e;
                }
                
                congestionRegionID[dir][e] = target_region;
                if(x!=select_root)
                for(auto pos: stems[dir][e])
                {
                    stems[dir][target_region].push_back(pos);
                }
                congestionRanges[dir][target_region].first = min(congestionRanges[dir][target_region].first, congestionRanges[dir][e].first);
                congestionRanges[dir][target_region].second = max(congestionRanges[dir][target_region].second, congestionRanges[dir][e].second);
            }
            else if(congestion == 2)
            {
                if(x!=select_root&&congestionRegionID[dir][x]==-1)
                {
                    congestionRegionID[dir][x] = x;
                    
                }
                if(congestionRegionID[dir][e]==-1)
                {
                    congestionRegionID[dir][e] = e;
                }
            }
            markCongestion(e, x);
        }
        for(int dir=0; dir<2; dir++)
        {
            int x_region = getRegionID(x, dir);
            for(auto e : graph_x[x]) if(e != par)
            {
                int ex = graph_x.back()[e] / Y % X;
                int ey = graph_x.back()[e] % Y;
                int e_region = getRegionID(e, dir);
                if(x_region!=e_region)
                {
                    if(x_region>=0)
                    {
                        stems[dir][x_region].push_back(dir?ex:ey);
                        congestionRanges[dir][x_region].first = min(congestionRanges[dir][x_region].first, dir?ey:ex);
                        congestionRanges[dir][x_region].second = max(congestionRanges[dir][x_region].second, dir?ey:ex);
                    }
                    if(e_region>=0)
                    {
                        stems[dir][e_region].push_back(dir?curx:cury);
                        congestionRanges[dir][e_region].first = min(congestionRanges[dir][e_region].first, dir?cury:curx);
                        congestionRanges[dir][e_region].second = max(congestionRanges[dir][e_region].second, dir?cury:curx);
                    }
                }
            }
        }
    };
    markCongestion(select_root, -1);

    function<int(int, int, int, int)> create_node = [&] (int l, int x, int y, int num_child) {
        int node_idx_insert = node_index_cnt++;
        par_num_cpu.emplace_back(0);
        par_num_sum_cpu.emplace_back(0);
        par_nodes_cpu.emplace_back(0);
        child_num_cpu.emplace_back(0);
        node_depth_cpu.emplace_back(0);
        nodes_cpu.emplace_back(0);
        
        par_num_sum_cpu[node_idx_insert+1] = 0;
        node_depth_cpu[node_idx_insert] = 0;
        child_num_cpu[node_idx_insert] = num_child;
        nodes_cpu[node_idx_insert] = l * X * Y + x * Y + y;
        return node_idx_insert;
    };

    function<int(int, int, int)> connect_node = [&] (int par_node_index, int cur_index, int cur_child_id) {
        int node_idx_insert = cur_index;
        int position_cur = nodes_cpu[cur_index];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int position_par = nodes_cpu[par_node_index];
        int parx = position_par/ Y % X, pary = position_par % Y;
        if(construct_segments)
        {
            if(curx==parx)
            {
                rsmt_v_segments.emplace_back(make_pair(curx*Y+min(cury, pary), curx*Y+max(cury, pary)));
            }
            if(cury==pary)
            {
                rsmt_h_segments.emplace_back(make_pair(min(curx, parx)+cury*X, max(curx, parx)+cury*X));
            }
        }        
        points.emplace_back(curx*Y+cury);
        points.emplace_back(parx*Y+pary);
        assert(curx==parx||cury==pary);
        node_depth_cpu[node_idx_insert] = max(node_depth_cpu[node_idx_insert], node_depth_cpu[par_node_index] + 1);
        int depth = node_depth_cpu[node_idx_insert];

        int position = par_num_sum_cpu[node_idx_insert]+par_num_cpu[node_idx_insert]++;
        par_nodes_cpu.emplace_back(0);
        currentChildIDX_cpu.emplace_back(0);
        par_nodes_cpu[position] = par_node_index;
        depth_max = max(depth_max, depth+1);
        assert(cur_child_id<child_num_cpu[par_node_index]);
        currentChildIDX_cpu[position]=cur_child_id;
        return node_idx_insert;
    };
    float ratio = 0.15;
    int num_tracks = 9;
    function<int(int, int, int)> calc_displace = [&] (int query_pos, int dir, int region_id) {
        int ans = 0;
        for(auto pos: stems[dir][region_id])
        {
            ans+=abs(pos-query_pos);
        }
        return ans;
    };
    function<vector<int>(int, int)> get_mirror_places = [&] (int graph_node_id, int dir) {
        assert(graph_node_id<graph_x.back().size());
        int position_cur = graph_x.back()[graph_node_id];
        assert(congestionRegionID[dir][graph_node_id]>=0);
        int congestion_region = congestionRegionID[dir][graph_node_id];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int trunk_len = congestionRanges[dir][congestion_region].second - congestionRanges[dir][congestion_region].first;
        int max_displace = ratio*float(trunk_len);
        int origional_pos = dir?curx:cury;
        int origional_displacement = calc_displace(origional_pos, dir, congestion_region);
        int init_low = origional_pos;
        int init_high = origional_pos;
        int bound = dir?X:Y;
        while (init_low - 1 >= 0 && calc_displace(init_low - 1, dir, congestion_region) - origional_displacement <= max_displace) init_low--;
        while (init_high + 1 < bound && calc_displace(init_high - 1, dir, congestion_region) - origional_displacement <= max_displace) init_high++;
        int step = 1;
        while ((origional_pos - init_low) / (step + 1) + (init_high - origional_pos) / (step + 1) >= num_tracks) step++;
        init_low = origional_pos - (origional_pos - init_low) / step * step;
        init_high = origional_pos + (init_high - origional_pos) / step * step;
        vector<int> shifts;
        for (double pos = init_low; pos <= init_high; pos += step) {
            int shiftAmount = (pos - origional_pos); 
            if(shiftAmount==0) continue;
            shifts.push_back(pos);
            int min_trunk = congestionRanges[dir][congestion_region].first;
            int max_trunk = congestionRanges[dir][congestion_region].second;
        }
        std::vector<int> indices(shifts.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::vector<int> new_shifts;
        for (int index : indices) {
            new_shifts.push_back(shifts[index]);
        }
    
        shifts = new_shifts;
        return shifts;
    };

    function<void(int, int, int, int, int, vector<vector<int>>, vector<vector<int>>)> dfs_detours = [&] (int x, int par, int par_node_idx, int child_idx, int depth, vector<vector<int>> mirrors, vector<vector<int>> old_mirror_places) {
        if(mirrors.size()==0)
        {
            mirrors.resize(2);
        }
        int size = graph_x.back().size() - 1;
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur/ Y % X, cury = position_cur % Y;
        int node_idx = -1;
        if(x==select_root)
        {
            node_idx = create_node(curl,curx,cury, graph_x[x].size());
            par_num_sum_cpu[node_idx+1] += par_num_cpu[node_idx];
            par_num_sum_cpu[node_idx+1] += par_num_sum_cpu[node_idx];
        }
        vector<vector<int>> new_mirrors;
        vector<vector<int>> mirror_places;
        new_mirrors.resize(2);
        mirror_places.resize(2);
        if(old_mirror_places.size()==2)
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(congestionRegionID[dir][x], dir);
            int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
            if(region_id >= 0)
            {
                assert(region_id<graph_x.back().size());
                if(getRegionID(congestionRegionID[dir][par], dir)==par)
                {
                    depth+=2;
                }
                if(x==region_id)
                {
                    mirror_places[dir] = get_mirror_places(x, dir);
                }else{
                    mirror_places[dir] = old_mirror_places[dir];
                }
                if(region_id!=x&&mirror_places[dir].size()!=mirrors[dir].size())
                {
                    assert(0);
                }
                assert(region_id==x||mirror_places[dir].size()==mirrors[dir].size());
                for(int m_i = 0; m_i < mirror_places[dir].size(); m_i++)
                {
                    int new_x = dir?mirror_places[dir][m_i]:curx;
                    int new_y = dir?cury:mirror_places[dir][m_i];
                    int new_mirror=-1;
                    if(region_id!=x)
                    {
                        int pre_idx = mirrors[dir][m_i];
                        new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                        assert( mirrors[dir].size()==mirror_places[dir].size());
                        connect_node(pre_idx, new_mirror, child_idx);
                    }
                    else{
                        if(x!=select_root)
                        {
                            int position_par = nodes_cpu[par_node_idx];
                            int parx = position_par/ Y % X, pary = position_par % Y;
                            if(new_x!=parx&&new_y!=pary)
                            {
                                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                    int midx = pathIndex ? new_x : parx;
                                    int midy = pathIndex ? pary : new_y;
                                    int node_insert = create_node(MAX_LAYER-1, midx, midy, 1);
                                    connect_node(par_node_idx, node_insert, child_idx);
                                    par_num_sum_cpu[node_insert + 1] += par_num_cpu[node_insert];
                                    par_num_sum_cpu[node_insert + 1] += par_num_sum_cpu[node_insert];
                                }
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                                connect_node(new_mirror-1, new_mirror, 0);
                                connect_node(new_mirror-2, new_mirror, 0);
                            }
                            else{
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                                connect_node(par_node_idx, new_mirror, child_idx);
                            }
                        }else{
                            new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                        } 
                    }
                    assert(new_mirror>0);
                    new_mirrors[dir].push_back(new_mirror);
                    par_num_sum_cpu[new_mirror + 1] += par_num_cpu[new_mirror];
                    par_num_sum_cpu[new_mirror + 1] += par_num_sum_cpu[new_mirror];
                }
            }
        }
        if(par_node_idx == -1){}
        else {
            int px = nodes_cpu[par_node_idx] / Y % X, py = nodes_cpu[par_node_idx] % Y;
            vector<int> pre_node_idxs;
            vector<int> pre_node_idxs_direct;
            if(px != curx && py != cury)
            {
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int midx = pathIndex ? curx : px;
                    int midy = pathIndex ? py : cury;
                    int pre_node = create_node(MAX_LAYER-1, midx, midy, 1);
                    connect_node(par_node_idx, pre_node, child_idx);
                    par_num_sum_cpu[pre_node + 1] += par_num_cpu[pre_node];
                    par_num_sum_cpu[pre_node + 1] += par_num_sum_cpu[pre_node];                    
                    pre_node_idxs.push_back(pre_node);
                }
                
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int length_edge = pathIndex ? (max(py, cury) - min(py, cury)) : (max(px, curx) - min(px, curx));
                    int max_z_shape = min(10, length_edge);
                    for(int dispace_id = 1; dispace_id < max_z_shape; dispace_id++)
                    {
                        int midx1 = pathIndex ? px : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy1 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : py;
                        int pre_node1 = create_node(MAX_LAYER-1, midx1, midy1, 1);
                        connect_node(par_node_idx, pre_node1, child_idx);
                        par_num_sum_cpu[pre_node1 + 1] += par_num_cpu[pre_node1];
                        par_num_sum_cpu[pre_node1 + 1] += par_num_sum_cpu[pre_node1];
    
                        int midx2 = pathIndex ? curx : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy2 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : cury;
                        int pre_node2 = create_node(MAX_LAYER-1, midx2, midy2, 1);
                        connect_node(pre_node1, pre_node2, 0);
                        par_num_sum_cpu[pre_node2 + 1] += par_num_cpu[pre_node2];
                        par_num_sum_cpu[pre_node2 + 1] += par_num_sum_cpu[pre_node2];                   
                        pre_node_idxs.push_back(pre_node2);
                    }
                }
            }
            for(int dir = 0; dir<2; dir++)
            {
                int region_id = getRegionID(congestionRegionID[dir][x], dir);
                int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
                if(par_region_id>=0&&region_id!=par_region_id)
                {
                    for(auto node_par_mirror: mirrors[dir])
                    {
                        int position_par = nodes_cpu[node_par_mirror];
                        int parx = position_par/ Y % X, pary = position_par % Y;
                        if(parx!=curx&&pary!=cury)
                        {
                            for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                int midx = pathIndex ? curx : parx;
                                int midy = pathIndex ? pary : cury;
                                int node_insert = create_node(MAX_LAYER-1, midx, midy, 1);
                                connect_node(node_par_mirror, node_insert, child_idx);
                                par_num_sum_cpu[node_insert + 1] += par_num_cpu[node_insert] + par_num_sum_cpu[node_insert];
                                pre_node_idxs.push_back(node_insert);
                            }
                        }else{
                            pre_node_idxs_direct.push_back(node_par_mirror);
                        }
                    }
                }
            }
            int connect_parent = par_node_idx;
            int connect_child_idx = child_idx;

            node_idx = create_node(curl, curx, cury, graph_x[x].size() - 1);
            
            if(px == curx || py == cury){
                connect_node(connect_parent, node_idx, connect_child_idx);
            }
            if(x!=select_root)
            {
                for(auto pre_node_idx: pre_node_idxs)
                {
                    connect_node(pre_node_idx, node_idx, 0);
                }
                for(auto pre_node_idx: pre_node_idxs_direct)
                {
                    connect_node(pre_node_idx, node_idx, child_idx);
                }
            }
        }
        depth_max = max(depth_max, node_depth_cpu[node_idx]+1);
        if(x!=select_root)
        {
            par_num_sum_cpu[node_idx+1] += (par_num_sum_cpu[node_idx] + par_num_cpu[node_idx]);
        }
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(x, dir);
            int par_region_id = getRegionID(par, dir);
            if(region_id<0) continue;
            if(true)
            {
                int pos2 = dir?cury:curx;
                assert(region_id>=0||region_id<graph_x.back().size());
                int is_tail = congestionRanges[dir][region_id].second==pos2 || congestionRanges[dir][region_id].first==pos2;
                int is_pin = curl < MAX_LAYER - 1;
                assert(x!=select_root);
                if(x!=select_root&&is_pin)
                {
                    for(auto mirror_id: new_mirrors[dir])
                    {
                        int node_duplicate = create_node(curl, curx, cury, 0);
                        child_num_cpu[mirror_id] = graph_x[x].size();// connect in advance
                        connect_node(mirror_id, node_duplicate, graph_x[x].size()-1);
                        par_num_sum_cpu[node_duplicate + 1] += par_num_cpu[node_duplicate];
                        par_num_sum_cpu[node_duplicate + 1] += par_num_sum_cpu[node_duplicate];
                    }
                }
            }
        }
        int idx = 0;
        for(auto e : graph_x[x]) if(e != par)
        {
            depth_max = max(depth_max, depth+1);
            dfs_detours(e, x, node_idx, idx++, depth+1, new_mirrors, mirror_places);
        }
    };
    if(construct_segments)
    {
        rsmt_h_segments.clear();
        rsmt_v_segments.clear();
        rsmt_h_segments.reserve(rsmt.size()*(num_tracks*2));
        rsmt_v_segments.reserve(rsmt.size()*(num_tracks*2));
    }
    points.clear();
    points.reserve(rsmt.size()*(num_tracks*4));
    dfs_detours(select_root, -1, -1, -1, 0, {}, {});
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end());
    points.erase(last, points.end());
    if(construct_segments)
    {
        for(int ii=0; ii<rsmt_h_segments.size();ii++)
        {
            int xpos = rsmt_h_segments[ii].first%X;
            int ypos = rsmt_h_segments[ii].first/X;
            rsmt_h_segments[ii].first = xpos*Y+ypos;
            xpos = rsmt_h_segments[ii].second%X;
            ypos = rsmt_h_segments[ii].second/X;
            rsmt_h_segments[ii].second = xpos*Y+ypos;
        }
    }
}



// The per-net subnet split below reads one finished db::nets entry at a time
// and never looks at the grid, so it runs as a consumer thread next to the
// single-threaded net-file parse instead of waiting for it.  db::read()
// publishes its progress through db_parsed_net_count.  net_break_count and
// max_pin_cnt are reported by build_cuda_database() after the consumer joins.
int net_break_count = 0, max_pin_cnt = 1;

void build_nets_from_parse() {
    while(!db_cap_parsed.load(std::memory_order_acquire)) std::this_thread::yield();
    L = db::L - 1;
    X = db::X;
    Y = db::Y;
    XY = max(X, Y);

    const int MAX_PIN_SIZE = 2000;
    size_t consumed = 0, hpwl_done = 0;
    while(true) {
        // Read the finished flag first: if it is set, the count read after it
        // is already the parser's final one.
        const bool parse_done = db_parse_finished.load(std::memory_order_acquire);
        const size_t available = db_parsed_net_count.load(std::memory_order_acquire);
        if(consumed == available) {
            if(parse_done) break;
            // Sleep rather than yield: the consumer keeps up with the parser
            // and would otherwise spend most of the parse in a sched_yield
            // loop, slowing down the very thread it is waiting on.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }
        for(int db_net_id = consumed; db_net_id < available; db_net_id++) {
            auto &db_net = db::nets[db_net_id];
            if(db_net.pins.size() == 1) continue;
            max_pin_cnt = max(max_pin_cnt, (int) db_net.pins.size());
            if(db_net.pins.size() <= MAX_PIN_SIZE) {

                net new_net;
                new_net.pins = db_net.pins;
                new_net.original_net_id = db_net_id;
                for(auto &p : new_net.pins) 
                    if(p >= X * Y) p -= X * Y;
                db_net.subnets.emplace_back(nets.size());
                nets.emplace_back(move(new_net));
                db_net.unfinished_subnet_count = db_net.subnets.size();
                continue;
            }
            net_break_count++;
            vector<int> pins = db_net.pins, sz(db_net.pins.size(), 1), par(db_net.pins.size());
            vector<tuple<int, int, int>> edges;
            edges.reserve(db_net.pins.size() * db_net.pins.size() / 2);

            function<int(int)> find_par = [&] (int x) { return x == par[x] ? x : par[x] = find_par(par[x]); };

            for(int i = 0; i < db_net.pins.size(); i++) {
                par[i] = i;
                if(pins[i] >= X * Y) pins[i] -= X * Y;
                for(int j = 0; j < i; j++) {
                    int x0 = db_net.pins[i] / Y % X, y0 = db_net.pins[i] % Y;
                    int x1 = db_net.pins[j] / Y % X, y1 = db_net.pins[j] % Y;
                    edges.emplace_back(make_tuple(j, i, abs(x0 - x1) + abs(y0 - y1)));
                }
            }
            sort(edges.begin(), edges.end(), [&] (tuple<int, int, int> l, tuple<int, int, int> r) {
                return get<2> (l) < get<2> (r);
            });
            for(auto e : edges) {
                int u = find_par(get<0> (e)), v = find_par(get<1> (e));
                if(u == v || sz[u] + sz[v] > MAX_PIN_SIZE) continue;
                if(sz[u] > sz[v]) swap(u, v);
                par[u] = v;
                sz[v] += sz[u];
            }
            vector<vector<int>> new_pins(pins.size());
            for(int i = 0; i < pins.size(); i++)
                new_pins[find_par(i)].emplace_back(pins[i]);
            for(auto e : edges) {
                int u = get<0> (e), v = get<1> (e);
                int par_u = find_par(u), par_v = find_par(v);
                if(par_u == par_v) continue;
                if(sz[par_u] > sz[par_v]) {
                    swap(u, v);
                    swap(par_u, par_v);
                }
                sz[par_u]++;
                new_pins[par_u].emplace_back(pins[v]);
                par[par_v] = par_u;
                db_net.subnets.emplace_back(nets.size());
                nets.emplace_back(net());
                nets.back().pins = move(new_pins[par_v]);
                nets.back().original_net_id = db_net_id;
            }
            for(int i = 0; i < pins.size(); i++) if(find_par(i) == i) {
                db_net.subnets.emplace_back(nets.size());
                nets.emplace_back(net());
                nets.back().pins = move(new_pins[i]);
                nets.back().original_net_id = db_net_id;
            }
            db_net.unfinished_subnet_count = db_net.subnets.size();
        }
        // Inside the loop, so the hpwl pass overlaps the parse too instead of
        // becoming a serial tail once the parser is done.
        for(; hpwl_done < nets.size(); hpwl_done++) nets[hpwl_done].calc_hpwl();
        consumed = available;
    }
}

void build_cuda_database() {


    const int MAX_LEN_INT = 1700000000, MAX_LEN_DOUBLE = 1700000000;
    static int *temp_int = new int[MAX_LEN_INT];
    static double *temp_double = new double[MAX_LEN_DOUBLE];
    static float *temp_float = new float[MAX_LEN_DOUBLE];

    
    pin_cnt_sum_cpu.resize(1 + nets.size(), 0);
    for(int i = 0; i < nets.size(); i++)
        pin_cnt_sum_cpu[i + 1] = pin_cnt_sum_cpu[i] + nets[i].pins.size();
    printf("    MAX PINS: %d\n", max_pin_cnt);
    printf("    Broken Nets: %d\n", net_break_count);

    
    NET_NUM = nets.size();


    DIR = db::layers[1].dir;
    unit_length_wire_cost = db::unit_length_wire_cost;
    unit_via_cost = db::unit_via_cost;

    assert(X - 1 <= MAX_LEN_INT);
    for(int i = 0; i < X - 1; i++) temp_int[i] = db::x_edge_len[i];
    cudaMalloc(&x_edge_len, (X - 1) * sizeof(int));
    cudaMemcpy(x_edge_len, temp_int, (X - 1) * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0; i < X; ++i) temp_int[i] = db::dr_x[i];
    cudaMalloc(&flute_x_coords, X * sizeof(int));
    cudaMemcpy(flute_x_coords, temp_int, X * sizeof(int), cudaMemcpyHostToDevice);


    assert(Y - 1 <= MAX_LEN_INT);
    for(int i = 0; i < Y - 1; i++) temp_int[i] = db::y_edge_len[i];
    cudaMalloc(&y_edge_len, (Y - 1) * sizeof(int));
    cudaMemcpy(y_edge_len, temp_int, (Y - 1) * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0; i < Y; ++i) temp_int[i] = db::dr_y[i];
    cudaMalloc(&flute_y_coords, Y * sizeof(int));
    cudaMemcpy(flute_y_coords, temp_int, Y * sizeof(int), cudaMemcpyHostToDevice);

    assert(L <= MAX_LEN_DOUBLE);
    for(int i = 0; i < L; i++) temp_double[i] = db::unit_length_short_costs[i + 1];
    cudaMalloc(&unit_length_short_costs, L * sizeof(double));
    cudaMemcpy(unit_length_short_costs, temp_double, L * sizeof(double), cudaMemcpyHostToDevice);

    assert(L <= MAX_LEN_DOUBLE);
    for(int i = 0; i < L; i++) temp_double[i] = db::layers[i + 1].min_len;
    cudaMalloc(&layer_min_len, L * sizeof(double));
    cudaMemcpy(layer_min_len, temp_double, L * sizeof(double), cudaMemcpyHostToDevice);


    assert(L * X * Y <= MAX_LEN_DOUBLE);
    for(int l = 0; l < L; l++)
        for(int x = 0; x < X; x++)
            for(int y = 0; y < Y; y++)
                temp_float[IDX(l, x, y)] = db::capacity[l + 1][x][y];
    cudaMalloc(&capacity, L * X * Y * sizeof(float));
    cudaMemcpy(capacity, temp_float, L * X * Y * sizeof(float), cudaMemcpyHostToDevice);
    

    cudaMalloc(&pin_acc_num, (1 + NET_NUM) * sizeof(int));
    cudaMemcpy(pin_acc_num, pin_cnt_sum_cpu.data(), (1 + NET_NUM) * sizeof(int), cudaMemcpyHostToDevice);
    PIN_NUM = pin_cnt_sum_cpu.back();

    if(LOG) cerr << "PIN_NUM " << PIN_NUM << endl;

    
    assert(PIN_NUM <= MAX_LEN_INT);

    for(auto &dbnet : db::nets)
        for(auto pin : dbnet.pins) 
            if(pin < X * Y) total_via_count++;

    for(int i = 0, pin_id = 0; i < NET_NUM; i++)
        for(auto pin : nets[i].pins) temp_int[pin_id++] = pin;
    cudaMalloc(&pins, PIN_NUM * sizeof(int));
    cudaMemcpy(pins, temp_int, PIN_NUM * sizeof(int), cudaMemcpyHostToDevice);


    net_x_cpu.resize(NET_NUM);
    net_y_cpu.resize(NET_NUM);
    

    all_track_cnt = 0;
    for(int l = 0; l < L; l++) all_track_cnt += (l & 1 ^ DIR) ? X : Y;
    assert(all_track_cnt <= MAX_LEN_INT);
    for(int l = 0, cnt = 0; l < L; l++) {
        if((l & 1 ^ DIR) == 0) for(int y = 0; y < Y; y++) temp_int[cnt++] = l * XY + y;
        if((l & 1 ^ DIR) == 1) for(int x = 0; x < X; x++) temp_int[cnt++] = l * XY + x;
        if(l + 1 == L) assert(cnt == all_track_cnt);
    }
    cudaMalloc(&idx2track, all_track_cnt * sizeof(int));
    cudaMemcpy(idx2track, temp_int, sizeof(int) * all_track_cnt, cudaMemcpyHostToDevice);


    cudaMalloc(&congestion, X * Y * sizeof(bool));
    cudaMalloc(&congestion_xsum, X * Y * sizeof(float));
    cudaMalloc(&congestion_ysum, X * Y * sizeof(float));


    cudaMallocManaged(&routes, (ROUTE_PER_PIN * PIN_NUM) * sizeof(int));

#ifdef INSTANTGR_LEGACY_PRESUM
    cudaMalloc(&wcost, L * X * Y * sizeof(float));
#endif
    cudaMalloc(&vcost, L * X * Y * sizeof(float));
    // Set before any kernel runs: the commit kernels read this flag, so it must
    // not be written from the host once the per-batch pipeline is in flight.
    // The list is sized from the requested fraction so the env knob is never
    // silently clamped by a fixed capacity (30% of a 207M-cell grid needs 62M
    // entries; the old 32M cap turned that request into 15.5%).
    {
        const char *value = getenv("INSTANTGR_INCREMENTAL_VCOST");
        incremental_vcost_on = value == nullptr || string(value) != "0";
        const char *presum_value = getenv("INSTANTGR_INCREMENTAL_PRESUM");
        incremental_presum_on = presum_value == nullptr || string(presum_value) != "0";
        const char *max_dirty = getenv("INSTANTGR_INCREMENTAL_VCOST_MAX_DIRTY");
        double fraction = 0.20;
        if(max_dirty != nullptr && *max_dirty != '\0') {
            fraction = atof(max_dirty);
            if(fraction <= 0 || fraction > 1)
                throw invalid_argument("INSTANTGR_INCREMENTAL_VCOST_MAX_DIRTY must be in (0, 1]");
        }
        dirty_cell_limit = (int) min((double) ((long long) L * X * Y), ceil(fraction * L * X * Y));
        dirty_cell_capacity = dirty_cell_limit;
        dirty_overflow = true;
        if(LOG && incremental_vcost_on)
            printf("Incremental vcost: dirty-cell limit %d of %lld grid cells (%.1f%%), list memory %.0f MB\n",
                   dirty_cell_limit, (long long) L * X * Y,
                   100.0 * dirty_cell_limit / ((double) L * X * Y),
                   dirty_cell_limit * sizeof(int) / 1048576.0);
    }
    cudaMalloc(&dirty_cells, (size_t) dirty_cell_capacity * sizeof(int));
    cudaMalloc(&track_dirty, (size_t) L * XY * sizeof(bool));
    cudaMemset(track_dirty, 1, (size_t) L * XY * sizeof(bool));
    cudaMalloc(&pre_demand_track_dirty, (size_t) L * XY * sizeof(bool));
    cudaMemset(pre_demand_track_dirty, 0, (size_t) L * XY * sizeof(bool));
    cudaMalloc(&presum, L * X * Y * sizeof(double));
    cudaMalloc(&demand, L * X * Y * sizeof(float));
    cudaMalloc(&pre_demand, L * X * Y * sizeof(int));



    cudaMalloc(&net_ids, NET_NUM * sizeof(int));
    cudaMalloc(&is_of_net, NET_NUM * sizeof(bool));
    cudaMalloc(&of_edge_sum, L * X * Y * sizeof(int));
    cudaMalloc(&last, L * X * Y * sizeof(int));
    cudaMalloc(&timestamp, L * X * Y * sizeof(int));


    

    cudaMemset(demand, 0, sizeof(float) * L * X * Y);
    cudaMemset(timestamp, 0, sizeof(int) * L * X * Y);
    cudaMemset(pre_demand, 0, sizeof(int) * L * X * Y);

    net_ids_cpu = new int[NET_NUM];
}

}

using namespace cudb;
