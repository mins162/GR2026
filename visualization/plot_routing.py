"""Routed wires as they actually sit on the die, over a 30x30 GCell crop.

Every other figure here comes from `data.py` timings; this one reads the router's
own output.  `mempool_tile_rank` is the design that fits -- 429x581 GCells and a
33 MB output, small enough to keep a gzipped copy in `sample/`.  `mempool_group`
is 884 MB and stays on the server.

Why a crop and not the whole die: 428k wire segments over 429x581 GCells renders
as a solid block at report size.  The window below holds 1610 of them, which is
the median density for a 30x30 window on this design -- dense enough to read as
real routing, sparse enough that individual wires stay apart.

Only wire segments are drawn.  A .out line is a via when its two layers differ,
and the crop holds 2511 of those against 1610 wires -- drawing them would bury
the routing under the pin map.
"""
import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import style

NAME = "routing_crop"

# Window picked by scanning every 30x30 window on the design and taking one at
# median segment density near the die centre, so the figure is not a lucky
# sparse corner.  Holds 1610 wires / 571 nets / metal2..metal6.
X0, X1 = 210, 240
Y0, Y1 = 240, 270

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "sample", "mempool_tile_rank.opt.r1.out.gz")


def load(path=SAMPLE):
    """Wire segments inside the window, bucketed by layer, plus the net count.

    Layer 0 is metal1 and carries no wires on this design (pins only), so the
    buckets that come back are metal2 and up.
    """
    by_layer = {}
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
    return by_layer, len(nets)


def build():
    by_layer, nets = load()

    # Layers are ordinal, so they get a sequential ramp rather than the
    # ours/borrowed split the other figures use.  viridis is monotonic in
    # lightness -- it survives a greyscale print -- and is cut at 0.85 because
    # the pale yellow tail would vanish against the page.
    ramp = plt.cm.viridis(np.linspace(0.0, 0.85, 8))

    fig, ax = plt.subplots(figsize=(3.3, 3.05))
    for layer in sorted(by_layer):                 # low layers first, high on top
        segs = by_layer[layer]
        # Wires stack: the guide is per GCell, so nets sharing an edge draw over
        # each other.  Partial alpha lets that pile-up darken instead of hiding.
        ax.add_collection(LineCollection(segs, colors=[ramp[layer - 1]],
                                         linewidths=0.85, alpha=0.7,
                                         label="metal%d" % (layer + 1)))

    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal")
    ax.set_title("mempool_tile_rank, routed wires", fontsize=7)
    ax.set_xlabel("GCell x")
    ax.set_ylabel("GCell y")
    ax.tick_params(labelsize=5.2)
    ax.grid(False)
    ax.legend(fontsize=4.9, ncol=len(by_layer), loc="lower center",
              bbox_to_anchor=(0.5, -0.30), handlelength=1.6, columnspacing=1.0)
    ax.text(0.0, -0.40, "%d nets in a %dx%d GCell window, opt run"
            % (nets, X1 - X0, Y1 - Y0),
            transform=ax.transAxes, fontsize=5.2, color=style.MUTED)
    return fig


if __name__ == "__main__":
    print(style.use_style())
    style.save(build(), NAME)
