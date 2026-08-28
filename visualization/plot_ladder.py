"""Cumulative ladder as a waterfall: where the 64.19 s actually went.

Hatched bar = the one step whose algorithm comes from the ICCAD'22 paper.  The
distinction survives greyscale printing, which the colour alone would not.
"""
import matplotlib.pyplot as plt

import data
import style

NAME = "ladder_waterfall"


def build():
    rows = data.LADDER
    labels = []
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    # Anchor bars run from zero; step bars float between the two totals.
    for i, (label, total, delta, ours) in enumerate(rows):
        if delta is None:
            ax.bar(i, total, 0.62, color=style.BASE, zorder=3)
            ax.text(i, total + 7.0, "%.2f s" % total, ha="center", fontsize=6.5,
                    weight="bold", color=style.INK)
        else:
            prev = rows[i - 1][1]
            ax.bar(i, prev - total, 0.62, bottom=total,
                   color=style.OURS if ours else style.PAPER,
                   hatch=None if ours else style.PAPER_HATCH,
                   edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(i, prev + 2.2, "%+.2f" % delta, ha="center", fontsize=6.5,
                    color=style.OURS if ours else style.PAPER)
        labels.append(label)

    ax.bar(len(rows), rows[-1][1], 0.62, color=style.BASE, zorder=3)
    ax.text(len(rows), rows[-1][1] + 7.0, "%.2f s" % rows[-1][1], ha="center",
            fontsize=6.5, weight="bold", color=style.INK)
    labels.append("opt")

    ax.set_ylim(0, 118)
    ax.set_ylabel("total runtime (s)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=5.8, rotation=38, ha="right")
    ax.xaxis.grid(False)
    ax.set_title("mempool_group: 92.47 s -> 28.28 s", fontsize=8, color=style.MUTED)

    ours = plt.Rectangle((0, 0), 1, 1, color=style.OURS)
    paper = plt.Rectangle((0, 0), 1, 1, color=style.PAPER, hatch=style.PAPER_HATCH)
    ax.legend([ours, paper], ["our contribution", "ICCAD'22 algorithm"],
              fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
