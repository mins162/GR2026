"""The whole die as a density map, with the routing crop marked on it.

This exists to answer one question about `plot_routing`: why its crop looks
lopsided, dense along the bottom and near-empty above.  The answer is that the
die is built that way and the crop sits on the seam.

Two things show up here.  The die is patchy on its own -- white rectangles are
macros, where the lower layers are blocked and only metal5 and up cross, and
15.4% of its GCells carry upper-layer routing with no metal2 or metal3 at all
while another 25.4% carry nothing.  And metal2/metal3 stop almost completely
past a line near the top: 120153 segments in the band y=400..450, 15984 in
y=450..500, 6831 in y=500..550.

The crop is drawn on top so the two can be read against each other.  It sits
across that line, which is not an accident: it was picked by searching for a
sparse window whose every border still carries routing, and on this die the
sparse ground is macro area and die edge.  Those two conditions together can
only be met on the seam between the cell rows and the macros.

Same source file and the same per-GCell segment count as `plot_routing`; the
crop rectangle is imported from it rather than copied, so it follows when the
window moves.
"""
import gzip

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import style
from plot_routing import SAMPLE, X0, X1, Y0, Y1

NAME = "die_overview"

# Coarser than the crop's 4: at die scale a 4-cell tile is under two pixels and
# the map turns to noise.
TILE = 8


def load(path=SAMPLE):
    """Per-GCell wire segment counts for the whole die, plus the metal2/3 share.

    Same rules as `plot_routing.load`: wires only, counted in every GCell they
    run through.  The second grid is metal2 and metal3 alone -- the layers the
    standard cells route on -- which is what makes the macro area legible.
    """
    total = np.zeros((1024, 1024), dtype=np.int32)
    low = np.zeros((1024, 1024), dtype=np.int32)
    in_net = False
    with gzip.open(path, "rt") as f:
        for line in f:
            head = line[0]
            if head == "(":
                in_net = True
                continue
            if head == ")":
                in_net = False
                continue
            if not in_net:
                continue
            x0, y0, l0, x1, y1, l1 = map(int, line.split())
            if l0 != l1:            # via
                continue
            for grid in (total, low) if l0 in (1, 2) else (total,):
                if x0 == x1:
                    grid[x0, y0:y1 + 1] += 1
                else:
                    grid[x0:x1 + 1, y0] += 1
    nx = np.flatnonzero(total.sum(1)).max() + 1
    ny = np.flatnonzero(total.sum(0)).max() + 1
    return total[:nx, :ny], low[:nx, :ny]


def build():
    total, low = load()
    ex, ey = total.shape
    # Pad up to whole tiles rather than trimming to them: the routed extent is
    # not a multiple of TILE, and trimming would drop a strip off the top and
    # right edges without saying so.  The pad is outside the routed area, so it
    # reads as the empty margin it is.
    nx, ny = -(-ex // TILE), -(-ey // TILE)
    padded = np.zeros((nx * TILE, ny * TILE), dtype=float)
    padded[:ex, :ey] = total
    tiles = padded.reshape(nx, TILE, ny, TILE).mean(axis=(1, 3))

    # Where metal2/metal3 give out.  Found rather than hardcoded so the line
    # cannot drift away from the data: the lowest row above which the cell-layer
    # density never climbs back to a fifth of the die's typical row.
    rows = low.sum(axis=0)
    weak = rows < 0.2 * np.median(rows[rows > 0])
    cliff = int(np.flatnonzero(~weak).max()) + 1

    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "density", plt.cm.Greys(np.linspace(0.0, 0.85, 256)))

    fig, ax = plt.subplots(figsize=(3.4, 4.4))
    im = ax.imshow(tiles.T, origin="lower", extent=(0, nx * TILE, 0, ny * TILE),
                   cmap=shade, vmin=0.0, vmax=float(np.percentile(tiles, 97)),
                   interpolation="nearest", aspect="equal")

    ax.axhline(cliff, color=style.OURS, lw=0.8, ls="--", zorder=2)
    ax.text(4, cliff + 8, "metal2/metal3 stop above here",
            color=style.OURS, fontsize=5.2)

    ax.add_patch(plt.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False,
                               edgecolor="#D55E00", linewidth=1.2, zorder=3))
    ax.text(X0 + 4, Y1 + 8, "routing_crop", color="#D55E00", fontsize=5.6)

    ax.set_title("mempool_tile_rank, whole die", fontsize=7)
    ax.set_xlabel("GCell x")
    ax.set_ylabel("GCell y")
    ax.tick_params(labelsize=5.2)
    ax.grid(False)

    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("wire segments per GCell (%dx%d tile mean)" % (TILE, TILE),
                  fontsize=5.2)
    bar.ax.tick_params(labelsize=5.0)
    bar.outline.set_visible(False)

    # The .out file carries no grid header -- only net names and segments -- so
    # the frame is the extent the routing actually reaches, not a declared die
    # size.
    ax.text(0.0, -0.15, "routed extent %dx%d GCells, opt run" % (ex, ey),
            transform=ax.transAxes, fontsize=5.2, color=style.MUTED)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
