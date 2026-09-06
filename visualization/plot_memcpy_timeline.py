"""CPU<->GPU traffic over the whole run, on top of the router's own stages.

Read from the 2026-08-26 nsys profile of mempool_group (incremental config),
dumped by tools/nsys_memcpy_csv.py.  The nsys clock and the router's log clock
agree to within 0.05 s (the first L-shape kernel lands at 14.106 s on nsys, the
log says the stage starts at 14.1), so the stage bands are taken straight from
the log without shifting.

Two things the summary table cannot show: the copies come in thousands of tiny
bursts synchronised with the routing batches, and the host spends ten times
longer *inside* cudaMemcpy than the GPU spends actually copying.
"""
import csv
import gzip
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import style

NAME = "memcpy_timeline"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample",
                   "mempool_group-incr.memcpy.csv.gz")

# nsys_results_0826_012934/mempool_group-incr.log, "[ t]" stamps.
STAGES = [
    ("input",        0.0,  4.4),
    ("build DB",     4.4,  6.3),
    ("S1 RSMT",      6.3, 11.4),
    ("S1 DAG",       11.4, 14.1),
    ("S1 route",     14.1, 17.0),
    ("S2 prep",      17.0, 20.2),
    ("S2 route",     20.2, 28.3),
    ("finish",       28.3, 29.9),
]
T_END = 29.9
BIN = 0.1
H2D, D2H, KERN = style.OURS, style.OURS_LIGHT, style.BASE


def load():
    rows = {"K": [], "H": [], "D": [], "R": []}
    with gzip.open(CSV, "rt") as f:
        for r in csv.DictReader(f):
            if r["kind"] in rows:
                rows[r["kind"]].append((float(r["start_s"]), float(r["end_s"]), int(r["bytes"])))
    return {k: np.array(v).reshape(-1, 3) for k, v in rows.items()}


def busy_per_bin(iv, edges):
    """Seconds of `iv` (start, end) intervals falling in each bin."""
    out = np.zeros(len(edges) - 1)
    for s, e in iv:
        lo = int(s / BIN)
        hi = min(int(e / BIN), len(out) - 1)
        for b in range(lo, hi + 1):
            out[b] += max(0.0, min(e, edges[b + 1]) - max(s, edges[b]))
    return out


def lane(ax, iv, y, color, half=0.32):
    segs = [[(s, y - half), (s, y + half)] for s, e, _ in iv]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=0.25, zorder=3))


def build():
    d = load()
    fig, (top, bot) = plt.subplots(2, 1, figsize=(4.32, 3.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.12})

    for ax in (top, bot):
        for i, (name, s, e) in enumerate(STAGES):
            if i % 2:
                ax.axvspan(s, e, color="#f1f0eb", zorder=0, lw=0)
    for name, s, e in STAGES:
        top.text((s + e) / 2, 2.62, name, ha="center", va="bottom", fontsize=4.3,
                 color=style.MUTED)

    lane(top, d["K"], 2, KERN)
    lane(top, d["H"], 1, H2D)
    lane(top, d["D"], 0, D2H)

    # The handful of big transfers, named; everything else is the dense burst.
    big = d["H"][d["H"][:, 2] > 30e6]
    clusters = []
    for s, e, b in big:
        if clusters and s - clusters[-1][0] < 1.5:
            clusters[-1][1].append(b)
        else:
            clusters.append([s, [b]])
    for s, bs in clusters:
        top.text(s - 0.15, 1.42, " / ".join("%.0f" % (b / 1e6) for b in bs) + " MB",
                 fontsize=4.3, color=style.OURS, ha="left", va="bottom")
    top.set_yticks([2, 1, 0])
    top.set_yticklabels(["GPU kernels",
                         "H→D copy\n{:,} copies, {:.2f} GB".format(len(d["H"]), d["H"][:, 2].sum() / 1e9),
                         "D→H copy\n{:,} copies, {:.0f} MB".format(len(d["D"]), d["D"][:, 2].sum() / 1e6)],
                        fontsize=5.2)
    top.set_ylim(-0.55, 2.6)
    top.yaxis.grid(False)
    top.xaxis.grid(False)
    style.strip(top, keep=("left",))

    # Per 100 ms: wall time the host sat inside cudaMemcpy vs. time the GPU
    # engine actually spent moving bytes.  Same unit, so one axis carries both.
    edges = np.arange(0, T_END + BIN, BIN)
    host = busy_per_bin(d["R"][:, :2], edges) / BIN * 100
    gpu = busy_per_bin(np.vstack([d["H"][:, :2], d["D"][:, :2]]), edges) / BIN * 100
    mid = edges[:-1]
    bot.fill_between(mid, host, step="post", color=style.MUTED, alpha=0.55, lw=0,
                     label="host blocked in cudaMemcpy  (%.2f s total)" % (d["R"][:, 1] - d["R"][:, 0]).sum())
    bot.fill_between(mid, gpu, step="post", color=style.OURS, lw=0,
                     label="GPU actually copying  (%.2f s total)" % (
                         (d["H"][:, 1] - d["H"][:, 0]).sum() + (d["D"][:, 1] - d["D"][:, 0]).sum()))
    bot.set_ylim(0, 100)
    bot.set_yticks([0, 50, 100])
    bot.set_ylabel("share of each 0.1 s (%)", fontsize=5.7)
    bot.set_xlabel("wall clock (s), mempool_group, incremental config")
    bot.set_xlim(0, T_END)
    bot.xaxis.grid(False)
    bot.legend(fontsize=4.8, loc="upper left", handlelength=1.2)
    style.strip(bot)

    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
