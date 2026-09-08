"""Render every figure into visualization/figures/ as SVG + PNG."""
import importlib

import style

SCRIPTS = ["plot_speedup", "plot_base_vs_opt", "plot_ladder",
           "plot_ladder_stacked", "plot_contributions", "plot_breakdown", "plot_breakdown_pie", "plot_overlap_timeline", "plot_memcpy_timeline", "plot_cpu_gpu_timeline", "plot_strategy_blocks", "plot_gpu_flute_overlap", "plot_s2_batch", "plot_routing", "plot_routing_layers", "plot_die_overview"]

if __name__ == "__main__":
    print("style: " + style.use_style())
    for name in SCRIPTS:
        mod = importlib.import_module(name)
        print(mod.NAME)
        style.save(mod.build(), mod.NAME)
