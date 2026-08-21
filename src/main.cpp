#include "Lshape_route.hpp"
#include "Lshape_route_detour.hpp"
#include "graph.hpp"
#include "database.hpp"
#include "database_cuda.hpp"

void route() {
    double stage_start = elapsed_time();
    build_cuda_database();
    record_runtime_stage("build CUDA database", stage_start);

    stage_start = elapsed_time();
    for(int i = 0; i < db::nets.size(); i++)
        if(db::nets[i].pins.size() == 1) nets2output.push(i);
    thread output_thread1(graph::output_nets);
    record_runtime_stage("enqueue 1-pin nets + writer 1 launch", stage_start);

    stage_start = elapsed_time();
    cudaFuncSetAttribute(graph::compute_presum, cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024);
    cudaFuncSetAttribute(graph::compute_presum_general, cudaFuncAttributeMaxDynamicSharedMemorySize, 60 * 1024);
    vector<int> nets2route_all;
    nets2route_all.resize(nets.size());
    for(int i = 0; i < nets.size(); i++) nets2route_all[i] = i;
    record_runtime_stage("route setup", stage_start);

    stage_start = elapsed_time();
    Lshape_route::Lshape_route(nets2route_all);
    record_runtime_stage("Lshape route", stage_start);

    stage_start = elapsed_time();
    if(LOG) graph::report_score();
    record_runtime_stage("score after Lshape", stage_start);

    stage_start = elapsed_time();
    output_thread1.join();
    record_runtime_stage("wait for writer 1", stage_start);

    int of_threshold = 0;

    if(nets.size() > 20000000) mode = 1;

    if(LOG) printf("[%5.1f] Stage 2 rip-up and rerouting starts\n", elapsed_time());
    stage_start = elapsed_time();
    if(LOG) printf("[%5.1f] Stage 2: congestion extraction + rip-up starts\n", elapsed_time());
    graph::extract_congestionView<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
    graph::extract_congestionView_xsum<<<Y, THREAD_NUM, sizeof(float) * X>>> ();
    graph::extract_congestionView_ysum<<<X, THREAD_NUM, sizeof(float) * Y>>> ();
    auto of_nets = graph::ripup(of_threshold);
    graph::finish_nets(of_nets.second);
    if(LOG) printf("[%5.1f] Stage 2: congestion extraction + rip-up ends\n", elapsed_time());
    record_runtime_stage("extract congestion + ripup", stage_start);

    stage_start = elapsed_time();
    thread output_thread2(graph::output_nets);
    record_runtime_stage("writer 2 launch", stage_start);

    stage_start = elapsed_time();
    Lshape_route_detour::Lshape_route_detour_wrap(of_nets.first);
    record_runtime_stage("DAG/detour route", stage_start);

    stage_start = elapsed_time();
    if(LOG) graph::report_score();
    record_runtime_stage("score after DAG/detour", stage_start);

    stage_start = elapsed_time();
    output_thread2.join();
    record_runtime_stage("wait for writer 2", stage_start);

    stage_start = elapsed_time();
    graph::finish_nets(of_nets.first);
    graph::output_nets();
    record_runtime_stage("finish nets + final output", stage_start);
}

void runtime_breakdown() {
    const double total_time = elapsed_time();

    // One line naming the configuration this number was measured under, so a
    // pasted table is self-describing in A/B comparisons.
#ifdef INSTANTGR_LEGACY_PRESUM
    const char *fusion = "off";
#else
    const char *fusion = "on";
#endif
    char gpu_flute_desc[64];
    if(!cudb::gpu_flute_enabled())
        snprintf(gpu_flute_desc, sizeof(gpu_flute_desc), "off");
    else if(cudb::gpu_flute_max_degree() == INT_MAX)
        snprintf(gpu_flute_desc, sizeof(gpu_flute_desc), "on");
    else
        snprintf(gpu_flute_desc, sizeof(gpu_flute_desc), "on (max-degree=%d)",
                 cudb::gpu_flute_max_degree());
    printf("\nconfig: gpu-flute=%s | wcost-presum-fusion=%s | incremental-vcost=%s (dirty limit %.0f%%) | "
           "incremental-presum=%s | tree-center=%s\n",
           gpu_flute_desc, fusion,
           cudb::incremental_vcost_on ? "on" : "off",
           100.0 * cudb::dirty_cell_limit / ((double) cudb::L * cudb::X * cudb::Y),
           cudb::incremental_presum_on ? "on" : "off",
           cudb::cpu_tree_center_enabled() ? "cpu" : (cudb::gpu_tree_center_enabled() ? "gpu" : "off"));

    // Two-level table.  Stage names starting with two spaces are sub-stages:
    // their time is already inside a parent row, so they are shown (indented,
    // when >= 1% of wall clock) but excluded from the accounted/other math.
    // Parent rows below 2% fold into one "other" line.
    const double parent_threshold = total_time * 0.02;
    const double child_threshold = total_time * 0.01;
    const auto row = [&] (const char *name, double seconds) {
        printf("%-30s %8.2f s %6.1f %%\n", name, seconds, seconds / total_time * 100);
    };
    const auto print_children = [&] (const char *marker) {
        for(const auto &stage : runtime_stages)
            if(stage.name.rfind(marker, 0) == 0 && stage.seconds >= child_threshold)
                row(stage.name.c_str(), stage.seconds);
    };
    double accounted = input_time, other = 0;
    int other_cnt = 0;
    printf("------------------------------------------------\n");
    row("input", input_time);
    for(const auto &stage : runtime_stages) {
        if(stage.name[0] == ' ') continue;
        accounted += stage.seconds;
        if(stage.seconds >= parent_threshold) {
            row(stage.name.c_str(), stage.seconds);
            if(stage.name == "Lshape route") print_children("  S1");
            if(stage.name == "DAG/detour route") print_children("  S2");
        } else {
            other += stage.seconds;
            other_cnt++;
        }
    }
    other += total_time - accounted;// unaccounted wall clock joins the misc row
    if(other > 0.005) {
        char label[64];
        snprintf(label, sizeof(label), "other (%d minor stages)", other_cnt);
        row(label, other);
    }
    printf("------------------------------------------------\n");
    row("total", total_time);
    printf("\n");
}



int main(int argc, char *argv[]) {
    program_start = std::chrono::high_resolution_clock::now();

    const int cap_file_idx = 2, net_file_idx = 4, out_file_idx = 6;
    db::read(argv[cap_file_idx], argv[net_file_idx]);

    double stage_start = elapsed_time();
    out_file = fopen(argv[out_file_idx], "w");
    record_runtime_stage("open output", stage_start);

    stage_start = elapsed_time();
    readLUT("POWV9.dat", "POST9.dat");
    record_runtime_stage("read FLUTE LUT", stage_start);

    route();

    stage_start = elapsed_time();
    fclose(out_file);
    record_runtime_stage("close output", stage_start);
    if(LOG) runtime_breakdown();
    // quick_exit() skips stream flushing; flush explicitly so the breakdown
    // survives redirection into a file or pipe.
    fflush(stdout);
    cout.flush();
    quick_exit(0);
}
