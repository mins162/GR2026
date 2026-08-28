"""Render every figure into visualization/figures/ as SVG + PNG."""
import importlib

import style

SCRIPTS = ["plot_speedup", "plot_base_vs_opt", "plot_ladder",
           "plot_contributions", "plot_breakdown", "plot_overlap_timeline"]

if __name__ == "__main__":
    print("style: " + style.use_style())
    for name in SCRIPTS:
        mod = importlib.import_module(name)
        print(mod.NAME)
        style.save(mod.build(), mod.NAME)
