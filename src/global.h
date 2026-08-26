#pragma once
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <thread>
#include <atomic>
#include <chrono>
#include <bitset>
#include <vector>
#include <string>
#include <queue>
#include <tuple>
#include <cmath>
#include <set>
#include <map>

using namespace std;

const bool LOG = true;
int mode = 0;
std::chrono::high_resolution_clock::time_point program_start;


queue<int> nets2output;
char output_buffer[1000000000];
FILE *out_file;
double input_time, output_time, Lshape_time, DAG_time;

// Handshake from the net-file parser to cudb::build_nets_from_parse(), which
// consumes finished db::nets entries while the parse is still running.
// db::nets is reserved up front, so entries below db_parsed_net_count never
// move and the consumer can read them without locking.
std::atomic<bool> db_cap_parsed{false};
std::atomic<size_t> db_parsed_net_count{0};
std::atomic<bool> db_parse_finished{false};

struct RuntimeStage {
    string name;
    double seconds;
};

vector<RuntimeStage> runtime_stages;
__managed__ double of_cost_scale = 1;

inline double elapsed_time() {
    std::chrono::duration<double> time_now = std::chrono::high_resolution_clock::now() - program_start;
    return time_now.count(); 
}

inline void record_runtime_stage(const char *name, double start_time) {
    runtime_stages.push_back({name, elapsed_time() - start_time});
}

void print_GPU_memory_usage() {// in Gigabytes
    size_t free_bytes, total_bytes;
    auto cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    assert(cuda_status == cudaSuccess);
    printf("        GPU memory consumption: %.2f GB", (total_bytes - free_bytes) / 1024.0 / 1024.0 / 1024.0);
    cout << endl;
}
