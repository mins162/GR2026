"""The run as CPU / transfer / GPU blocks, after InstantGR's Fig. 11.

One clock, Stage 1 above the dashed line and Stage 2 below it.  Each stage has
a host lane and a device lane, so overlap shows as two colours at one x.  Every
block comes from the 2026-08-26 nsys profile (same CSV as the other timeline
scripts): the run is cut into 20 ms cells and each cell takes the colour of
what dominated it -- host lane: computing / inside cudaMemcpy / otherwise
waiting (blank); device lane: kernels covering >= 30 % of the cell.

The orange boxes mark where the report's strategies act, with times from the
router log (nsys_results_0826_012934/mempool_group-incr.log).
"""
import csv
import gzip
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import style
from plot_memcpy_timeline import T_END

NAME = "strategy_blocks"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample",
                   "mempool_group-incr.memcpy.csv.gz")
CELL = 0.02
STAGE2_AT = 17.0                  # log: "Stage 2 rip-up and rerouting starts"
CPU, XFER, GPU, BOX = "#6e2a6b", "#d9a63a", "#2f6fb8", "#e8873a"

# (label, t0, t1, stage, label above or below the box) -- log stamps.
BOXES = [
    ("CPU/GPU overlap",           6.2, 11.5, 1, "above"),  # GPU FLUTE under the CPU FLUTE loop
    ("GPU batch gen",            11.8, 12.8, 1, "above"),  # S1 batch generation
    ("GPU batch gen",            18.9, 20.2, 2, "above"),  # S2 batch generation
    ("incremental vcost/commit", 20.2, 25.8, 2, "below"),  # S2 route batches
]


def load():
    rows = {"K": [], "A": [], "R": []}
    with gzip.open(CSV, "rt") as f:
        for r in csv.DictReader(f):
            if r["kind"] in rows:
                rows[r["kind"]].append((float(r["start_s"]), float(r["end_s"])))
    return {k: np.array(v).reshape(-1, 2) for k, v in rows.items()}


def runs(colour_per_cell, edges):
    """Merge consecutive same-colour cells into (x0, width, colour) blocks."""
    out = []
    for i, c in enumerate(colour_per_cell):
        if c and out and out[-1][2] == c and abs(out[-1][0] + out[-1][1] - edges[i]) < 1e-9:
            out[-1][1] += CELL
        elif c:
            out.append([edges[i], CELL, c])
    return out


def lane(ax, blocks, y, lo, hi, h=0.72):
    for x, w, c in blocks:
        if x + w <= lo or x >= hi:
            continue
        x0, x1 = max(x, lo), min(x + w, hi)
        ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=c, lw=0, zorder=3))


def build():
    d = load()
    edges = np.arange(0, T_END + CELL, CELL)
    api = _busy(d["A"], edges)
    xfer = _busy(d["R"], edges)
    gpu = _busy(d["K"], edges)
    comp = CELL - api
    host = [CPU if c >= CELL / 2 else XFER if x >= CELL / 2 else None
            for c, x in zip(comp, xfer)]
    dev = [GPU if g >= 0.3 * CELL else None for g in gpu]
    host_blocks, dev_blocks = runs(host, edges), runs(dev, edges)

    fig, ax = plt.subplots(figsize=(4.32, 2.3))
    # Stage 1 lanes on top, Stage 2 below, one shared clock.
    lane(ax, host_blocks, 3.0, 0, STAGE2_AT)
    lane(ax, dev_blocks, 2.2, 0, STAGE2_AT)
    lane(ax, host_blocks, 0.9, STAGE2_AT, T_END)
    lane(ax, dev_blocks, 0.1, STAGE2_AT, T_END)
    ax.axhline(1.55, color=style.INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.text(0.1, 3.55, "Stage 1: initial routing", fontsize=5.2, color=style.MUTED, va="bottom")
    ax.text(0.1, 1.35, "Stage 2: rip-up and reroute", fontsize=5.2, color=style.MUTED, va="top")

    for label, t0, t1, stage, side in BOXES:
        ybot = 2.2 - 0.46 if stage == 1 else 0.1 - 0.46
        ax.add_patch(Rectangle((t0, ybot), t1 - t0, 1.72, fill=False, ec=BOX, lw=0.8, zorder=4))
        if side == "above":
            ax.text(t0, ybot + 1.76, label, fontsize=4.4, color=BOX, va="bottom")
        else:
            ax.text(t0, ybot - 0.04, label, fontsize=4.4, color=BOX, va="top")

    ax.set_xlim(0, T_END)
    ax.set_ylim(-0.75, 3.95)
    ax.set_yticks([3.0, 2.2, 0.9, 0.1])
    ax.set_yticklabels(["host", "device", "host", "device"], fontsize=5.2)
    ax.set_xlabel("wall clock (s), mempool_group, incremental config")
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.legend(handles=[Patch(color=CPU, label="CPU"), Patch(color=XFER, label="Data transfer"),
                       Patch(color=GPU, label="GPU")],
              fontsize=5.2, loc="upper right", handlelength=1.1, frameon=True,
              edgecolor=style.INK, fancybox=False, borderpad=0.5)
    style.strip(ax)
    return fig


def _busy(iv, edges):
    out = np.zeros(len(edges) - 1)
    for s, e in iv:
        lo = int(s / CELL)
        hi = min(int(e / CELL), len(out) - 1)
        for b in range(lo, hi + 1):
            out[b] += max(0.0, min(e, edges[b + 1]) - max(s, edges[b]))
    return out


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
