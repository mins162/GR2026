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

    ax.set_ylim(0, 108)
    ax.set_ylabel("total runtime (s)")
    # Designs are numbered, not named: the names belong in the caption.  The
    # speedup sits under its own pair rather than floating over the bars.
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx], fontsize=6.5)
    for i, (name, base, opt, mult) in enumerate(rows):
        ax.text(i, -0.115, "%.2fx" % mult, ha="center", va="top", fontsize=6.5,
                weight="bold", color=style.INK, transform=ax.get_xaxis_transform())
    ax.xaxis.grid(False)
    ax.legend(fontsize=5.7, loc="upper right")
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
