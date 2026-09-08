"""The same crop as `plot_routing`, one panel per metal layer.

The single-panel version paints the layers on top of one another with opaque
lines.  On a sparse crop that is what keeps the upper-layer trunks visible, but
in a dense window it inverts the picture: metal4's 1449 segments cover metal2's
34638 and the figure reads green where the routing is overwhelmingly blue.
Here each layer gets its own frame, so a panel's darkness is its own layer's
count and nothing else.

The first panel is the density map with no wires over it.  In the overlay that
map is the thing that gets buried, and it is the only panel here that shows the
crop's shape whole.

Window, colours, tile size and the reader all come from `plot_routing` -- this
is that figure re-laid-out, not a second measurement of it.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import style
from plot_routing import LAYER_COLOR, TILE, X0, X1, Y0, Y1, load
from matplotlib.collections import LineCollection

NAME = "routing_layers"

# Width of one panel in inches.  The single-panel figure gives its axes 3.0 and
# sets line width from that; the lines here have to come down in proportion or a
# panel a third the width draws at three times the relative weight.
PANEL_W = 1.15


def build():
    by_layer, nets, density = load()
    layers = sorted(by_layer)

    nx, ny = -(-(X1 - X0) // TILE), -(-(Y1 - Y0) // TILE)
    padded = np.zeros((nx * TILE, ny * TILE), dtype=float)
    padded[:X1 - X0, :Y1 - Y0] = density
    tiles = padded.reshape(nx, TILE, ny, TILE).mean(axis=(1, 3))

    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "density", plt.cm.Greys(np.linspace(0.0, 0.70, 256)))

    ncol = 3
    nrow = -(-(len(layers) + 1) // ncol)          # +1 for the density panel
    panel_h = PANEL_W * (Y1 - Y0) / float(X1 - X0)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * PANEL_W + 0.7,
                                      nrow * panel_h + 0.95))
    flat = axes.ravel()

    im = flat[0].imshow(tiles.T, origin="lower",
                        extent=(X0, X0 + nx * TILE, Y0, Y0 + ny * TILE),
                        cmap=shade, vmin=0.0,
                        vmax=float(np.percentile(tiles, 95)),
                        interpolation="nearest", aspect="auto")
    flat[0].set_title("density, no wires", fontsize=5.6)

    lw = 0.85 * (140.0 / (X1 - X0)) * (PANEL_W / 3.0)
    for ax, layer in zip(flat[1:], layers):
        segs = by_layer[layer]
        # Rasterised for the same reason as in `plot_routing`: as vector paths
        # this figure was a 10 MB SVG against 30-65 KB for every other one.
        ax.add_collection(LineCollection(segs, colors=[LAYER_COLOR[layer + 1]],
                                         linewidths=lw, rasterized=True))
        ax.set_title("metal%d  %d segments" % (layer + 1, len(segs)),
                     fontsize=5.6)

    for i, ax in enumerate(flat):
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y0, Y1)
        ax.set_box_aspect((Y1 - Y0) / float(X1 - X0))
        ax.grid(False)
        ax.tick_params(labelsize=4.4, length=0)
        if i % ncol:                              # ticks on the left column only
            ax.set_yticklabels([])
        if i < len(flat) - ncol:                  # and on the bottom row only
            ax.set_xticklabels([])
    for ax in flat[len(layers) + 1:]:             # unused cells in the grid
        ax.set_visible(False)

    bar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("wire segments per GCell (%dx%d tile mean)" % (TILE, TILE),
                  fontsize=4.8)
    bar.ax.tick_params(labelsize=4.4)
    bar.outline.set_visible(False)

    fig.suptitle("mempool_tile_rank, routed wires by layer", fontsize=7)
    # Both of these default to the figure's bottom edge, where they land on top
    # of each other; the label is lifted to sit above the caption line.
    fig.supxlabel("GCell x", y=0.07, fontsize=5.6, color=style.MUTED)
    fig.supylabel("GCell y", fontsize=5.6, color=style.MUTED)
    fig.text(0.02, 0.01, "%d nets, %dx%d GCell window at x=%d, y=%d, opt run"
             % (nets, X1 - X0, Y1 - Y0, X0, Y0),
             fontsize=4.8, color=style.MUTED)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
