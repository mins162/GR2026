"""Runtime breakdown as a donut: what the remaining 28.28 s is spent on.

Percentages only -- the 92.47 -> 28.28 magnitude is the other figures' job.
Labels say what the stage does, not what the log calls it.
"""
import matplotlib.pyplot as plt

import data
import style

NAME = "breakdown_donut"

# (label, seconds, colour).  Blue = real routing work, grey = everything else,
# so the 3/4-of-the-time split reads without touching the legend.
SLICES = [
    ("Routing pass 1\nL-shape search",      10.90, style.OURS),
    ("Routing pass 2\ndetour search",       10.47, style.OURS_LIGHT),
    ("Reading input files",                  4.67, "#a8a79f"),
    ("Writing result",                       1.43, "#c9c8c0"),
    ("Other",                                0.82, "#e4e2da"),
]
TOTAL = 28.28


def build():
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    sizes = [s[1] for s in SLICES]
    wedges, _ = ax.pie(
        sizes, colors=[s[2] for s in SLICES], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.0))

    # Leader lines instead of inline labels: two slices are too thin to hold
    # text.  Label anchors are then pushed apart per side, or the three small
    # slices at the top would print on top of each other.
    import math
    placed = {1: [], -1: []}
    for wedge, (label, sec, _c) in zip(wedges, SLICES):
        ang = (wedge.theta1 + wedge.theta2) / 2
        x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        placed[1 if x >= 0 else -1].append([label, sec, x, y, y])

    for side, entries in placed.items():
        entries.sort(key=lambda e: -e[4])
        for i in range(1, len(entries)):
            gap = entries[i - 1][4] - entries[i][4]
            if gap < 0.62:
                entries[i][4] = entries[i - 1][4] - 0.62
        for label, sec, x, y, ty in entries:
            ax.annotate("%s\n%.2f s  (%.1f%%)" % (label, sec, 100 * sec / TOTAL),
                        xy=(0.82 * x, 0.82 * y), xytext=(1.30 * side, ty),
                        ha="left" if side > 0 else "right", va="center",
                        fontsize=5.2, color=style.MUTED,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#c3c2b7",
                                        connectionstyle="arc3,rad=0.08"))

    ax.text(0, 0.10, "28.28 s", ha="center", va="center", fontsize=9.5,
            weight="bold", color=style.INK)
    ax.text(0, -0.16, "after optimisation", ha="center", va="center",
            fontsize=5.5, color=style.MUTED)
    ax.text(0, -1.42, "routing work 75.5%  |  everything else 24.5%",
            ha="center", fontsize=5.5, color=style.MUTED)
    ax.set(aspect="equal")
    ax.set_xlim(-2.3, 2.3)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
