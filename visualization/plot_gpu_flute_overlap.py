"""GPU FLUTE hidden under the CPU FLUTE loop, as CPU / transfer / GPU blocks.

Zoom of the S1 RSMT window from the 2026-08-26 nsys profile.  The main thread
issues no CUDA call between 6.3 s and 11.4 s (checked in the sqlite), so its
lane is the CPU FLUTE loop end to end.  The helper thread's lane is what the
profile recorded for it: kernels from first to last drawn as ONE block (the
uploads and the reverse-merge gaps inside are not broken out), the two big
transfers on top, then the host-side tree build that construct_rsmt_gpu does
after the GPU returns.  The join at 11.4 s finds it long finished.
"""
import csv
import gzip
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import style
from plot_strategy_blocks import CPU, XFER, GPU, BOX

NAME = "gpu_flute_overlap"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample",
                   "mempool_group-incr.memcpy.csv.gz")
T0, T1 = 6.0, 11.7
CPU_LOOP = (6.3, 11.4)        # log: "Stage 1 RSMT: CPU degree < 10 ... wall=5.150s"
GPU_THREAD_WALL = 2.229       # log: "Stage 1 RSMT: GPU wall=2.229s"
HELPER_LO, HELPER_HI = 6.3, 6.75   # the helper thread's own CUDA calls sit here


def load():
    k, h = [], []
    with gzip.open(CSV, "rt") as f:
        for r in csv.DictReader(f):
            s, e = float(r["start_s"]), float(r["end_s"])
            if not (HELPER_LO <= s <= HELPER_HI):
                continue
            if r["kind"] == "K":
                k.append((s, e))
            elif r["kind"] in "HD" and int(r["bytes"]) > 1e6:
                h.append((s, e))
    return np.array(k), np.array(h)


def block(ax, x0, x1, y, c, h=0.72, z=3):
    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=c, lw=0, zorder=z))


def build():
    kern, xfer = load()
    gpu0, gpu1 = kern[:, 0].min(), kern[:, 1].max()
    helper_start = HELPER_LO
    helper_end = helper_start + GPU_THREAD_WALL

    fig, ax = plt.subplots(figsize=(4.32, 1.75))
    y_cpu, y_gpu = 1.0, 0.0
    block(ax, *CPU_LOOP, y_cpu, CPU)
    block(ax, gpu0, gpu1, y_gpu, GPU)
    for s, e in xfer:
        block(ax, s, e, y_gpu, XFER, z=4)
    block(ax, gpu1, helper_end, y_gpu, CPU)

    ax.add_patch(Rectangle((helper_start - 0.05, y_gpu - 0.46), helper_end - helper_start + 0.1,
                           1.92, fill=False, ec=BOX, lw=0.8, zorder=5))
    ax.text(helper_start - 0.05, y_cpu + 0.5, "GPU FLUTE thread, %.2f s, fully under the CPU loop"
            % GPU_THREAD_WALL, fontsize=4.6, color=BOX, va="bottom")
    ax.annotate("join: tail beyond CPU loop = 0.000 s", xy=(CPU_LOOP[1], y_cpu), xytext=(9.6, y_cpu - 0.75),
                fontsize=4.4, color=style.MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.5, color=style.MUTED))
    ax.text((gpu0 + gpu1) / 2, y_gpu - 0.5, "GPU %.2f s" % (gpu1 - gpu0), fontsize=4.4,
            color=style.MUTED, ha="center", va="top")
    ax.text((gpu1 + helper_end) / 2, y_gpu - 0.5, "host tree build", fontsize=4.4,
            color=style.MUTED, ha="center", va="top")
    ax.text(sum(CPU_LOOP) / 2, y_cpu + 0.1, "CPU FLUTE, 2.37 M nets, 8 threads, %.2f s"
            % (CPU_LOOP[1] - CPU_LOOP[0]), fontsize=4.6, color="white", ha="center", va="center")

    ax.set_xlim(T0, T1)
    ax.set_ylim(-0.95, 1.95)
    ax.set_yticks([y_cpu, y_gpu])
    ax.set_yticklabels(["main thread", "GPU FLUTE\nthread"], fontsize=5.2)
    ax.set_xlabel("wall clock (s), S1 RSMT, mempool_group")
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.legend(handles=[Patch(color=CPU, label="CPU"), Patch(color=XFER, label="Data transfer"),
                       Patch(color=GPU, label="GPU")],
              fontsize=5.0, loc="upper right", handlelength=1.1, frameon=True,
              edgecolor=style.INK, fancybox=False, borderpad=0.4, ncol=3)
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
