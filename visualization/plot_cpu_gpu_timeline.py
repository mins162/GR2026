"""Where the CPU and the GPU each spend the run, side by side on one clock.

Same nsys profile and CSV as plot_memcpy_timeline.py.  The profile carried no
CPU sampling (tools/nsys_profile.sh defaults to --sample=none), so "CPU
computing" is defined as the main thread being *outside* every CUDA runtime
call: parsing, FLUTE, batch generation, DAG build.  Time inside cudaMemcpy /
cudaDeviceSynchronize / cudaLaunchKernel is drawn as "CPU in CUDA API",
which is mostly waiting for the GPU.  GPU is the union of all kernels.
"""
import csv
import gzip
import os

import numpy as np
import matplotlib.pyplot as plt

import style
from plot_memcpy_timeline import STAGES, T_END, BIN, busy_per_bin

NAME = "cpu_gpu_timeline"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample",
                   "mempool_group-incr.memcpy.csv.gz")


def load():
    rows = {"K": [], "A": []}
    with gzip.open(CSV, "rt") as f:
        for r in csv.DictReader(f):
            if r["kind"] in rows:
                rows[r["kind"]].append((float(r["start_s"]), float(r["end_s"])))
    return {k: np.array(v).reshape(-1, 2) for k, v in rows.items()}


def build():
    d = load()
    edges = np.arange(0, T_END + BIN, BIN)
    t = edges[:-1]
    api = busy_per_bin(d["A"], edges) / BIN * 100
    cpu = 100 - api
    gpu = busy_per_bin(d["K"], edges) / BIN * 100
    cpu_s = cpu.sum() * BIN / 100
    api_s = (d["A"][:, 1] - d["A"][:, 0]).sum()
    gpu_s = (d["K"][:, 1] - d["K"][:, 0]).sum()

    fig, (top, bot) = plt.subplots(2, 1, figsize=(4.32, 2.6), sharex=True,
                                   gridspec_kw={"hspace": 0.18})
    for ax in (top, bot):
        for i, (name, s, e) in enumerate(STAGES):
            if i % 2:
                ax.axvspan(s, e, color="#f1f0eb", zorder=0, lw=0)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xlim(0, T_END)
        ax.xaxis.grid(False)
        style.strip(ax)
    for name, s, e in STAGES:
        top.text((s + e) / 2, 103, name, ha="center", va="bottom", fontsize=4.3,
                 color=style.MUTED)

    # CPU row: its own work stacked under the time it sat inside the CUDA API.
    top.fill_between(t, 0, cpu, step="post", color=style.OURS, lw=0, zorder=3,
                     label="CPU computing  (%.1f s)" % cpu_s)
    top.fill_between(t, cpu, 100, step="post", color=style.BASE, lw=0, zorder=3,
                     label="CPU inside CUDA API, mostly waiting  (%.1f s)" % api_s)
    top.set_ylabel("CPU (%)", fontsize=5.7)
    top.legend(fontsize=4.6, loc="lower left", bbox_to_anchor=(0.0, 1.14), ncol=2,
               handlelength=1.2, columnspacing=1.0)

    bot.fill_between(t, 0, gpu, step="post", color=style.PAPER, lw=0, zorder=3,
                     label="GPU kernels running  (%.1f s)" % gpu_s)
    bot.set_ylabel("GPU (%)", fontsize=5.7)
    bot.set_xlabel("wall clock (s), mempool_group, incremental config")
    bot.legend(fontsize=4.6, loc="upper left", handlelength=1.2)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
