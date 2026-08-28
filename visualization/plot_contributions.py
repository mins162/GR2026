"""Leave-one-out margin per contribution, on both designs that were run twice.

bsg_chip is on the same axis on purpose: GPU batch generation lands at -0.02 s
there, and that non-result is part of what the figure has to say.
"""
import numpy as np
import matplotlib.pyplot as plt

import data
import style

NAME = "contribution_margins"


def build():
    rows = data.LEAVE_ONE_OUT
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    h = 0.34

    fig, ax = plt.subplots(figsize=(4.32, 2.8))
    for i, (label, grp, bsg, ours) in enumerate(rows):
        color = style.OURS if ours else style.PAPER
        hatch = None if ours else style.PAPER_HATCH
        ax.barh(y[i] + h / 2, grp, h, color=color, hatch=hatch,
                edgecolor="white", linewidth=0.4, zorder=3)
        ax.barh(y[i] - h / 2, bsg, h, color=color, alpha=0.42, hatch=hatch,
                edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(grp + 0.9, y[i] + h / 2, "%+.2f" % grp, va="center",
                fontsize=4.7, color=style.MUTED)
        ax.text(max(bsg, 0) + 0.9, y[i] - h / 2, "%+.2f" % bsg, va="center",
                fontsize=4.7, color=style.MUTED)

    ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.set_xlabel("seconds lost when this one knob is turned off")
    ax.set_xlim(-3, 56)
    ax.yaxis.grid(False)

    solid = plt.Rectangle((0, 0), 1, 1, color=style.OURS)
    faint = plt.Rectangle((0, 0), 1, 1, color=style.OURS, alpha=0.42)
    ax.legend([solid, faint], ["mempool_group", "bsg_chip"], fontsize=5.2,
              loc="lower right")
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
