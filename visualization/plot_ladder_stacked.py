"""The same ladder as one bar: base height, cut into what each knob removed.

Bottom block is what is left after everything is on; every block above it is one
contribution's share of the 64.19 s that went away.  The hatched block is the
one algorithm taken from the ICCAD'22 paper.
"""
import matplotlib.pyplot as plt

import data
import style

NAME = "ladder_stacked"

# Bottom-up: the remainder first, then the ladder steps in reverse order, so the
# bar reads in the same order the doc switches the knobs on when read downward.
BLOCKS = [
    ("opt (what is left)",          28.28, "#b4b2a9", None),
    ("tree-center",                  0.55, "#b6dbf8", None),
    ("GPU batch generation",         4.00, "#8ec4f2", None),
    ("CPU/GPU overlap",              2.05, "#6aaeea", None),
    ("GPU-FLUTE (ICCAD'22)",         5.95, "#8a8a8a", style.PAPER_HATCH),
    ("incremental wire commit",      5.46, "#2a78d6", None),
    ("incremental vcost/presum",    46.18, "#185fa5", None),
]
BASE = 92.47


def build():
    fig, ax = plt.subplots(figsize=(3.7, 3.2))

    bottom = 0.0
    anchors = []
    for label, sec, color, hatch in BLOCKS:
        ax.bar(0, sec, 0.52, bottom=bottom, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.7, zorder=3)
        # The remainder block is named by the bold caption under the bar, so it
        # does not get a leader label of its own.
        if sec != BLOCKS[0][1]:
            anchors.append([label, sec, bottom + sec / 2, bottom + sec / 2, color])
        bottom += sec

    # Thin blocks would print their labels on top of each other; push the text
    # anchors apart and let the leader lines do the pointing.
    anchors.sort(key=lambda a: -a[3])
    for i in range(1, len(anchors)):
        if anchors[i - 1][3] - anchors[i][3] < 9.0:
            anchors[i][3] = anchors[i - 1][3] - 9.0

    for label, sec, y, ty, color in anchors:
        ax.annotate("%s\n%.2f s  (%.1f%%)" % (label, sec, 100 * sec / BASE),
                    xy=(0.27, y), xytext=(0.62, ty), va="center", ha="left",
                    fontsize=5.2, color=style.MUTED,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#c3c2b7"))

    ax.text(0, BASE + 2.5, "base  92.47 s", ha="center", fontsize=7,
            weight="bold", color=style.INK)
    ax.text(0, -5.0, "opt (what is left)  28.28 s   3.27x", ha="center", fontsize=7,
            weight="bold", color=style.INK)

    ax.set_ylim(-8, 104)
    ax.set_xlim(-0.45, 2.6)
    ax.set_ylabel("total runtime (s), mempool_group")
    ax.set_xticks([])
    ax.xaxis.grid(False)
    style.strip(ax, keep=("left",))
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
