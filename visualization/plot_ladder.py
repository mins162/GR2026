"""Cumulative ladder: every column is a full runtime, the coloured cap is what
that one knob removed from the column before it.

Columns are drawn from zero rather than floating, so each step visibly overlaps
its predecessor and the cap reads as a slice taken off the top.  Hatched cap =
the one algorithm from the ICCAD'22 paper.
"""
import matplotlib.pyplot as plt

import data
import style

NAME = "ladder_waterfall"


def build():
    rows = data.LADDER
    labels = []
    fig, ax = plt.subplots(figsize=(4.32, 2.91))

    # Every column runs from zero: the grey body is what that configuration
    # still costs, the cap on top is what the knob just removed.
    for i, (label, total, delta, ours) in enumerate(rows):
        if delta is None:
            ax.bar(i, total, 0.62, color=style.BASE, zorder=3)
            ax.text(i, total + 7.0, "%.2f s" % total, ha="center", fontsize=5.2,
                    weight="bold", color=style.INK)
        else:
            prev = rows[i - 1][1]
            ax.bar(i, total, 0.62, color=style.BASE, zorder=3)
            ax.bar(i, prev - total, 0.62, bottom=total,
                   color=style.OURS if ours else style.PAPER,
                   hatch=None if ours else style.PAPER_HATCH,
                   edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(i, prev + 2.2, "%+.2f" % delta, ha="center", fontsize=5.2,
                    color=style.OURS if ours else style.PAPER)
            ax.text(i, total - 5.5, "%.2f" % total, ha="center", fontsize=4.6,
                    color="#6b6b6b", zorder=5)
        labels.append(label)

    ax.bar(len(rows), rows[-1][1], 0.62, color=style.BASE, zorder=3)
    ax.text(len(rows), rows[-1][1] + 7.0, "%.2f s" % rows[-1][1], ha="center",
            fontsize=5.2, weight="bold", color=style.INK)
    labels.append("opt")

    ax.set_ylim(0, 118)
    ax.set_ylabel("total runtime (s)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=4.5, rotation=38, ha="right")
    ax.xaxis.grid(False)
    ax.set_title("mempool_group: 92.47 s -> 28.28 s", fontsize=6.7, color=style.MUTED)

    ours = plt.Rectangle((0, 0), 1, 1, color=style.OURS)
    paper = plt.Rectangle((0, 0), 1, 1, color=style.PAPER, hatch=style.PAPER_HATCH)
    ax.legend([ours, paper], ["our contribution", "ICCAD'22 algorithm"],
              fontsize=5.2, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
