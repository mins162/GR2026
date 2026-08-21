#include "Lshape_route.hpp"
#include "Lshape_route_detour.hpp"
#include "graph.hpp"
#include "database.hpp"
#include "database_cuda.hpp"

void route() {
    build_cuda_database();
    for(int i = 0; i < db::nets.size(); i++)
        if(db::nets[i].pins.size() == 1) nets2output.push(i);
    thread output_thread1(graph::output_nets);
    
    cudaFuncSetAttribute(graph::compute_presum, cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024);
    cudaFuncSetAttribute(graph::compute_presum_general, cudaFuncAttributeMaxDynamicSharedMemorySize, 60 * 1024);
    vector<int> nets2route_all;
    nets2route_all.resize(nets.size());
    for(int i = 0; i < nets.size(); i++) nets2route_all[i] = i;
    
    
    Lshape_route::Lshape_route(nets2route_all);
    if(LOG) graph::report_score();
    output_thread1.join();

    int of_threshold = 0;

    if(nets.size() > 20000000) mode = 1;

    graph::extract_congestionView<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
    graph::extract_congestionView_xsum<<<Y, THREAD_NUM, sizeof(float) * X>>> ();
    graph::extract_congestionView_ysum<<<X, THREAD_NUM, sizeof(float) * Y>>> ();
    auto of_nets = graph::ripup(of_threshold);
    graph::finish_nets(of_nets.second);
    thread output_thread2(graph::output_nets);
    Lshape_route_detour::Lshape_route_detour_wrap(of_nets.first);
    if(LOG) graph::report_score();
    output_thread2.join();
    graph::finish_nets(of_nets.first);
    graph::output_nets();
}

// Same table format as the optimized tree's runtime_breakdown() so the two
// logs can be compared line by line.  Upstream has no feature switches, so the
// config line states the fixed configuration.
void runtime_breakdown() {
    const double total_time = elapsed_time();
    printf("\nconfig: upstream cuhk-eda/InstantGR | cpu-flute only | full-grid cost rebuild every batch\n");
    const auto row = [&] (const char *name, double seconds) {
        printf("%-30s %8.2f s %6.1f %%\n", name, seconds, seconds / total_time * 100);
    };
    printf("------------------------------------------------\n");
    row("input", input_time);
    row("Lshape route", Lshape_time);
    row("DAG/detour route", DAG_time);
    const double other = total_time - input_time - Lshape_time - DAG_time;
    if(other > 0.005) row("other", other);
    printf("------------------------------------------------\n");
    row("total", total_time);
    printf("\n");
}



int main(int argc, char *argv[]) {
    program_start = std::chrono::high_resolution_clock::now();

    const int cap_file_idx = 2, net_file_idx = 4, out_file_idx = 6;
    db::read(argv[cap_file_idx], argv[net_file_idx]);
    out_file = fopen(argv[out_file_idx], "w");
    readLUT("POWV9.dat", "POST9.dat");
    route();
    fclose(out_file);
    if(LOG) runtime_breakdown();
    // quick_exit() skips stream flushing, which silently discards the tail of
    // buffered stdout (final logs, score, this breakdown) when output is
    // redirected.  Flush explicitly before leaving.
    fflush(stdout);
    cout.flush();
    quick_exit(0);
}