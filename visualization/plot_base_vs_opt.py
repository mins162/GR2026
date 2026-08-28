"""base and opt side by side, in seconds.

Deliberately linear: the two small designs stay short because they really are
short, and flattening that would hide where the win actually is.
"""
import numpy as np
import matplotlib.pyplot as plt

import data
import style

NAME = "base_vs_opt"


def build():
    rows = sorted(data.ANCHORS, key=lambda r: -r[1])
    names = [r[0] for r in rows]
    idx = np.arange(len(rows))
    w = 0.34

    fig, ax = plt.subplots(figsize=(4.2, 2.69))
    ax.bar(idx - w / 2, [r[1] for r in rows], w, label="base", color=style.BASE, zorder=3)
    ax.bar(idx + w / 2, [r[2] for r in rows], w, label="opt", color=style.OURS, zorder=3)

    for i, (name, base, opt, mult) in enumerate(rows):
        ax.text(i - w / 2, base + 1.5, "%.2f" % base, ha="center", fontsize=4.7, color=style.MUTED)
        ax.text(i + w / 2, opt + 1.5, "%.2f" % opt, ha="center", fontsize=4.7, color=style.OURS)
        ax.text(i, max(base, opt) + 6.5, "%.2fx" % mult, ha="center",
                fontsize=6.7, weight="bold", color=style.INK)

    ax.set_ylim(0, 108)
    ax.set_ylabel("total runtime (s)")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=4.7, rotation=20, ha="right")
    ax.xaxis.grid(False)
    ax.legend(fontsize=5.7, loc="upper right")
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
