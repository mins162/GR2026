"""Where the runtime sits, base vs opt, as one stacked bar each.

Same axis for both, so the bar lengths carry 92.47 vs 28.28 while the segments
carry the mix.  Two pie charts would have thrown that away.
"""
import matplotlib.pyplot as plt

import data
import style

NAME = "stage_breakdown"

SHADES = ["#b4b2a9", style.OURS, style.OURS_LIGHT, "#cfd8de", "#e4e2da"]


def build():
    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    left_base = left_opt = 0.0
    for (label, base, opt), color in zip(data.STAGES, SHADES):
        ax.barh(1, base, 0.5, left=left_base, color=color, edgecolor="white",
                linewidth=0.6, zorder=3, label=label)
        ax.barh(0, opt, 0.5, left=left_opt, color=color, edgecolor="white",
                linewidth=0.6, zorder=3)
        # Only wide enough segments get an inline number; the rest read from the legend.
        if base >= 6:
            ax.text(left_base + base / 2, 1, "%.1f" % base, ha="center", va="center",
                    fontsize=6, color="white")
        if opt >= 5:
            ax.text(left_opt + opt / 2, 0, "%.1f" % opt, ha="center", va="center",
                    fontsize=6, color="white")
        left_base += base
        left_opt += opt

    ax.text(left_base + 1.5, 1, "92.47 s", va="center", fontsize=7, weight="bold")
    ax.text(left_opt + 1.5, 0, "28.28 s", va="center", fontsize=7, weight="bold",
            color=style.OURS)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["base", "opt"], fontsize=7.5)
    ax.set_xlabel("runtime (s), mempool_group")
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.5, 1.5)
    ax.yaxis.grid(False)
    ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.50))
    style.strip(ax)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
