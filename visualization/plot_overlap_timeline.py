"""The overlap, drawn: the GPU lane disappears underneath the CPU loop.

This is the one contribution a table cannot carry -- serial spends GPU time and
CPU time back to back, overlapped spends only the CPU loop and the GPU tail
beyond it measures 0.000 s.
"""
import matplotlib.pyplot as plt

import data
import style

NAME = "overlap_timeline"
DESIGN = "mempool_group"


def build():
    d = data.OVERLAP[DESIGN]
    fig, ax = plt.subplots(figsize=(3.6, 1.9))

    # Serial: GPU FLUTE first, then the CPU loop.
    ax.broken_barh([(0, d["gpu"])], (2.6, 0.55), facecolors=style.PAPER,
                   hatch=style.PAPER_HATCH, edgecolor="white", linewidth=0.5, zorder=3)
    ax.broken_barh([(d["gpu"], d["cpu"])], (2.6, 0.55), facecolors=style.BASE,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(d["gpu"] + d["cpu"] + 0.15, 2.87, "%.2f s" % (d["gpu"] + d["cpu"]),
            va="center", fontsize=7, weight="bold")

    # Overlapped: two lanes running at once, GPU wall fully inside the CPU loop.
    ax.broken_barh([(0, d["overlapped_cpu"])], (1.35, 0.55), facecolors=style.OURS,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.broken_barh([(0, d["gpu_wall"])], (0.6, 0.55), facecolors=style.PAPER,
                   hatch=style.PAPER_HATCH, edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(d["overlapped_cpu"] + 0.15, 1.62, "%.2f s" % d["overlapped_cpu"],
            va="center", fontsize=7, weight="bold", color=style.OURS)
    ax.annotate("tail beyond CPU loop = %.3f s" % d["tail"],
                xy=(d["gpu_wall"], 0.87), xytext=(d["gpu_wall"] + 0.9, 0.28),
                fontsize=6, color=style.MUTED,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=style.MUTED))

    ax.axvline(d["overlapped_cpu"], color="#c3c2b7", lw=0.7, ls="--", zorder=2)
    ax.set_yticks([2.87, 1.62, 0.87])
    ax.set_yticklabels(["serial: GPU then CPU", "overlapped: CPU lane",
                        "overlapped: GPU lane"], fontsize=6.5)
    ax.set_xlabel("S1 RSMT wall clock (s), %s" % DESIGN)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0.2, 3.5)
    ax.yaxis.grid(False)
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
