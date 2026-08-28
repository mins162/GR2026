"""Shared look for every figure: IEEE-ish, greyscale-safe, English labels only.

SciencePlots is used when it is installed and skipped when it is not, so the
figures come out either way.  Labels stay English on purpose -- see
docs/visualization-tools.md 5 for why (matplotlib has no Korean glyphs by
default, and every figure label here is already an English identifier).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ours vs. borrowed: the same two-colour split the contribution boundary table
# uses.  Hatching carries the same distinction when the page is printed grey.
OURS = "#2a78d6"
OURS_LIGHT = "#8ec4f2"
PAPER = "#8a8a8a"
BASE = "#b4b2a9"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
PAPER_HATCH = "///"

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def use_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee"])
        used = "scienceplots(science,ieee)"
    except ImportError:
        used = "builtin fallback"
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })
    return used


def strip(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=0)


def save(fig, name):
    """Vector for the report, PNG for quick viewing."""
    paths = []
    for ext in ("svg", "png"):
        path = os.path.join(FIGDIR, "%s.%s" % (name, ext))
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    print("  " + "  ".join(paths))
    return paths
