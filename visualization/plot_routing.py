"""Routed wires over their own density, on a 150x150 GCell crop of the die.

Every other figure here comes from `data.py` timings; this one reads the router's
own output.  `mempool_tile_rank` is the design that fits -- 429x581 GCells and a
33 MB output, small enough to keep a gzipped copy in `sample/`.  `mempool_group`
is 884 MB and stays on the server.

The window is x=200..350, y=100..250, chosen by hand rather than by the search
earlier versions used.  It lands in the standard-cell heart of the die: 65619
wire segments over 22144 nets, 12.88 per GCell against the die average of 8.81,
and 1% of its cells empty.

At that density the figure is a solid block, which is the thing the crop exists
to avoid -- the density map is completely covered and individual routes cannot
be followed.  See `plot_die_overview` for where this sits; the sparse ground on
this die is macro area, not cell rows.

Draw order matters more than usual here.  Layers are painted low to high and the
lines are opaque, so metal4's 1449 segments sit on top of metal2's 34638 and the
figure reads green where the routing is overwhelmingly blue.  On a sparse crop
that ordering is what keeps the upper-layer trunks visible; at 1% empty it
inverts the picture.

Only wire segments are drawn.  A .out line is a via when its two layers differ,
and vias outnumber wires in any crop -- drawing them would bury the routing
under the pin map.
"""
import gzip
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import style

NAME = "routing_crop"

# Side of the square block of GCells the density map averages over.
TILE = 4

# Width of the plotting box in inches; the height follows the window's shape.
AXES_W = 3.0

# Sparsest 140x140 window that is sparse for the right reason.  Ranked on
# density, but only among windows that keep every border third carrying routing
# and stay under 30% empty cells -- without those two the ranking just walks off
# the edge of the routed area and buys its low density with dead frame.
#
# The bound on y comes from the same thing: routing thins out hard above y=522
# (a row there holds a fifth of what a row at y=500 does) while the die runs to
# 581, so windows reaching the top edge were excluded rather than rewarded.
#
# Set by hand to the region asked for, not by the density search the earlier
# windows used -- so none of the sparseness or framing constraints apply here.
X0, X1 = 200, 350
Y0, Y1 = 100, 250

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "sample", "mempool_tile_rank.opt.r1.out.gz")

# Keyed by metal number, not by rank within the window: which layers show up
# depends on where the crop lands (the dense window has metal2..6, the sparse
# one reaches metal8), and a rank-indexed palette would silently recolour every
# layer when the window moves.  Fixed here so two crops stay comparable.
#
# Okabe-Ito, so the set holds up for colourblind readers.  Distinct hues rather
# than one ramp: a ramp puts the two layers that carry the figure -- metal2 and
# metal3, 222k and 190k segments design-wide -- in neighbouring shades, and at
# 0.85 pt they were hard to tell apart where they cross.  The trade is that a
# greyscale print no longer reads layer order off lightness; hue carries the
# identity instead.  metal7 up holds 65 segments in the whole design, so those
# three are there to keep the palette total rather than to be read.
LAYER_COLOR = {
    2: "#0072B2",   # blue
    3: "#D55E00",   # vermillion
    4: "#009E73",   # green
    5: "#CC79A7",   # purple
    6: "#E69F00",   # orange
    7: "#56B4E9",   # sky blue
    8: "#000000",   # black
    9: "#F0E442",   # yellow
}



def load(path=SAMPLE):
    """Wire segments in the window by layer, the net count, and the density grid.

    Layer 0 is metal1 and carries no wires on this design (pins only), so the
    buckets that come back are metal2 and up.

    Density is wire segments per GCell.  A segment is counted in every GCell it
    runs through, not just where it starts -- a 40-cell trunk loads 40 cells, and
    counting it once would leave the channels it fills reading as empty.  Vias
    are left out for the same reason they are not drawn.  Segment coordinates in
    this format always run low to high (checked across all 428486 of them), so
    the ranges below need no min/max.
    """
    by_layer = {}
    density = np.zeros((X1 - X0, Y1 - Y0), dtype=np.int32)
    nets = set()
    net_id = -1
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
            if not in_net:          # a bare line between nets is the net name
                net_id += 1
                continue
            x0, y0, l0, x1, y1, l1 = map(int, line.split())
            if x1 < X0 or x0 > X1 or y1 < Y0 or y0 > Y1:
                continue
            nets.add(net_id)
            if l0 != l1:            # via
                continue
            by_layer.setdefault(l0, []).append([(x0, y0), (x1, y1)])
            # Clip to the window before depositing: a trunk can start outside it.
            cx0, cx1 = max(x0, X0) - X0, min(x1, X1 - 1) - X0
            cy0, cy1 = max(y0, Y0) - Y0, min(y1, Y1 - 1) - Y0
            if cx0 <= cx1 and cy0 <= cy1:
                density[cx0:cx1 + 1, cy0:cy1 + 1] += 1
    return by_layer, len(nets), density


def build():
    by_layer, nets, density = load()

    # The axes takes the window's shape (see set_box_aspect below).  Sizing the
    # to a fixed square left a wide crop floating in white; instead the axes is
    # pinned to AXES_W inches wide and the figure grows to fit whatever height
    # that implies, plus a constant band for the title, labels and legend.
    axes_h = AXES_W * (Y1 - Y0) / float(X1 - X0)
    fig, ax = plt.subplots(figsize=(AXES_W + 0.9, axes_h + 1.15))

    # Density underneath, wires on top.
    #
    # Shown as tile means, not raw GCells.  Per-GCell the hot cells are exactly
    # the cells the wires are drawn on, so the map hides under its own wires and
    # what survives is a stripe pattern rather than a field.  Averaging over
    # TILE spreads each channel into the space beside it, which is where there
    # is room to see it.
    #
    # Scaled to the 95th percentile of the tiles rather than the maximum: raw
    # cells run 0 to 28 with a median of 1, and against that top end every
    # ordinary cell washes out to white.  Tiles above p95 clip to the top shade.
    #
    # Greys and not a colour ramp: the wire hues are this figure's categorical
    # channel and a saturated colourmap would compete for the same reading.
    # Capped at 0.70 so the darkest tiles stay grey -- a black floor would
    # swallow metal2 and metal8.
    # Padded up to whole tiles, not trimmed to them, so a window whose sides are
    # not multiples of TILE keeps its last strip instead of losing it silently.
    # The pad sits outside the window and is never drawn -- imshow is given the
    # window's own extent below, so the padded tiles fall beyond the axis limits.
    nx, ny = -(-(X1 - X0) // TILE), -(-(Y1 - Y0) // TILE)
    padded = np.zeros((nx * TILE, ny * TILE), dtype=float)
    padded[:X1 - X0, :Y1 - Y0] = density
    tiles = padded.reshape(nx, TILE, ny, TILE).mean(axis=(1, 3))

    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "density", plt.cm.Greys(np.linspace(0.0, 0.70, 256)))
    im = ax.imshow(tiles.T, origin="lower",
                   extent=(X0, X0 + nx * TILE, Y0, Y0 + ny * TILE),
                   cmap=shade, vmin=0.0, vmax=float(np.percentile(tiles, 95)),
                   interpolation="nearest", zorder=0, aspect="auto")

    for layer in sorted(by_layer):                 # low layers first, high on top
        segs = by_layer[layer]
        # Opaque on purpose.  These were alpha 0.7 when the figure stood alone,
        # which let crossings darken; over the heatmap that same blending pulled
        # the wire colours toward whatever cell they sat on.
        #
        # Width tracks the window rather than being fixed: the axes is the same
        # size whatever the crop, so a line that reads at 140 GCells across is
        # most of a cell wide at 180 and the routing fuses into a block.  0.85
        # was tuned at 140, and the rest follows from it.
        # Rasterised inside the SVG.  These are tens of thousands of segments and
        # writing them as vector paths put the file at 10 MB -- 200x every other
        # figure here, and a fresh 10 MB blob in git history on every re-render.
        # Axes, text and colourbar stay vector; at the 300 dpi savefig uses, the
        # wires are indistinguishable either way.
        ax.add_collection(LineCollection(segs, colors=[LAYER_COLOR[layer + 1]],
                                         linewidths=0.85 * 140.0 / (X1 - X0),
                                         zorder=2, rasterized=True,
                                         label="metal%d" % (layer + 1)))

    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("wire segments per GCell (%dx%d tile mean)" % (TILE, TILE),
                  fontsize=5.2)
    bar.ax.tick_params(labelsize=5.0)
    bar.outline.set_visible(False)

    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    # Box aspect rather than data aspect.  Both give square GCells once the
    # limits are the window, but "equal" shrinks the axes inside a box that
    # keeps its original height, and the colourbar tracks that box -- on a wide
    # crop it ended up twice the height of the plot it labels.
    ax.set_box_aspect((Y1 - Y0) / float(X1 - X0))
    ax.set_title("mempool_tile_rank, routed wires over GCell density",
                 fontsize=7)
    ax.set_xlabel("GCell x")
    ax.set_ylabel("GCell y")
    ax.tick_params(labelsize=5.2)
    ax.grid(False)
    # Anchored by its top edge, not its bottom: "lower center" measures from the
    # legend's own bottom, so tightening the offset walked the legend up into
    # the x label instead of toward it.  The offsets are axes fractions, so they
    # are derived from the axes height in inches -- a fixed fraction that clears
    # the x label on a square crop lands on top of it on a short wide one.
    ax.legend(fontsize=4.9, ncol=len(by_layer), loc="upper center",
              bbox_to_anchor=(0.5, -0.35 / axes_h), handlelength=1.6,
              columnspacing=1.0)
    ax.text(0.0, -0.66 / axes_h, "%d nets, %dx%d GCell window, opt run"
            % (nets, X1 - X0, Y1 - Y0),
            transform=ax.transAxes, fontsize=5.2, color=style.MUTED)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
