"""One Stage 2 batch, from rip-up through vcost to commit, as blocks.

Zoom of ~16 ms around 22.0 s in the 2026-08-26 nsys profile, chosen from the
middle of the S2 route loop.  Kernel names come from the profile; the phase
they belong to follows the loop body in src/Lshape_route_detour.hpp (rip-up,
update_cost, compute_presum, bottom-up DP, traceback, commit).  The host lane
is the main thread: purple where it computes (the next batch's DAG prep,
which runs while the commit kernels are still in flight), yellow inside
cudaMemcpy, blank while it waits in cudaDeviceSynchronize.
"""
import csv
import gzip
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import style
from plot_strategy_blocks import CPU, XFER, GPU, BOX

NAME = "s2_batch"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample",
                   "mempool_group-incr.memcpy.csv.gz")
# Window: from the host prep that precedes this batch's upload to the end of
# the next batch's upload.  Absolute nsys seconds.
T0, T1 = 22.0016, 22.0183
PHASE = {                     # kernel -> phase, in loop order
    "init_costs": "init", "init_min_child_costs": "init", "init_road": "init",
    "commit_wire_demand": "rip-up", "update_vcost_selective": "vcost",
    "compute_presum": "presum", "reset_dirty_state": "presum",
    "Lshape_route_node_cuda": "DP route", "get_routing_tree_cuda": "traceback",
    "compute_presum_general": "commit", "commit_dirty_tracks": "commit",
}


def load():
    kern, api, xfer = [], [], []
    with gzip.open(CSV, "rt") as f:
        for r in csv.DictReader(f):
            s, e = float(r["start_s"]), float(r["end_s"])
            if e < T0 or s > T1:
                continue
            if r["kind"] == "N":
                kern.append((s, e, r["name"]))
            elif r["kind"] == "A":
                api.append((s, e))
            elif r["kind"] == "R":
                xfer.append((s, e))
    kern.sort()
    return kern, api, xfer


def phases(kern):
    """Group consecutive kernels by phase; commit_via_demand joins whatever
    phase it follows (rip-up removes, commit adds)."""
    out = []
    for s, e, name in kern:
        ph = PHASE.get(name, out[-1][0] if out else "?")
        if out and out[-1][0] == ph:
            out[-1][2] = e
        else:
            out.append([ph, s, e])
    return out


def ms(t):
    return (t - T0) * 1e3


def block(ax, x0, x1, y, c, h=0.72, z=3):
    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=c, lw=0, zorder=z))


def build():
    kern, api, xfer = load()
    fig, ax = plt.subplots(figsize=(4.32, 1.9))
    y_cpu, y_gpu = 1.0, 0.0

    # host: computing = outside every CUDA call; memcpy on top.
    t = T0
    compute = []
    for s, e in sorted(api):
        if s > t:
            compute.append((t, s))
        t = max(t, e)
    if t < T1:
        compute.append((t, T1))
    for s, e in compute:
        block(ax, ms(s), ms(e), y_cpu, CPU)
    # The two long host segments are the DAG prep of this batch and the next.
    for (s, e), label in zip(sorted(compute, key=lambda c: c[0] - c[1])[:2], ("DAG prep", "next DAG\nprep")):
        ax.text(ms((s + e) / 2), y_cpu + 0.1, label, fontsize=4.3, color="white", ha="center", va="center")
    for s, e in xfer:
        block(ax, ms(s), ms(e), y_cpu, XFER, z=4)

    ph = phases(kern)
    for i, (name, s, e) in enumerate(ph):
        block(ax, ms(s), ms(e), y_gpu, GPU)
        # narrow phases get their label lifted onto a second line
        lift = 0.0 if ms(e) - ms(s) > 1.2 else 0.28
        ax.text(ms((s + e) / 2), y_gpu - 0.56 - lift, name, fontsize=4.3, color=style.MUTED,
                ha="center", va="top")

    v0 = min(s for n, s, e in ph if n == "vcost")
    c1 = max(e for n, s, e in ph if n == "commit")
    ax.add_patch(Rectangle((ms(v0) - 0.08, y_gpu - 0.46), ms(c1) - ms(v0) + 0.16, 1.92,
                           fill=False, ec=BOX, lw=0.8, zorder=5))
    ax.text(ms(v0) - 0.08, y_cpu + 0.5, "vcost → commit, %.1f ms" % (ms(c1) - ms(v0)),
            fontsize=4.6, color=BOX, va="bottom")

    ax.set_xlim(0, ms(T1))
    ax.set_ylim(-1.25, 1.95)
    ax.set_yticks([y_cpu, y_gpu])
    ax.set_yticklabels(["host", "device"], fontsize=5.2)
    ax.set_xlabel("time within one S2 batch (ms), mempool_group, t = %.3f s" % T0)
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
