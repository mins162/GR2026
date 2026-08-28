"""Numbers from docs/rtx3060-ab.md (RTX 3060 12GB, 60-run A/B matrix, 2026-08-26).

Transcribed by hand, section by section.  Every table below names the section it
came from so a re-measurement can be diffed against the source doc.  When a real
`runs.csv` is at hand, replace these literals rather than editing the plots.
"""

# rtx3060-ab.md 1 -- base vs opt anchors, ordered as the doc lists them.
ANCHORS = [
    ("mempool_group",      92.47, 28.28, 3.27),
    ("bsg_chip",           24.51, 11.52, 2.13),
    ("nvdla",               8.45,  3.21, 2.63),
    ("mempool_tile_rank",   3.27,  2.34, 1.40),
]

# rtx3060-ab.md 3 -- cumulative ladder on mempool_group.  `ours` marks what the
# contribution boundary table (contest-plan.md 3-1) claims as our own work.
LADDER = [
    ("base",                        92.47,   None, None),
    ("incremental vcost/presum",    46.29, -46.18, True),
    ("incremental wire commit",     40.83,  -5.46, True),
    ("GPU-FLUTE (serial)",          34.88,  -5.95, False),
    ("CPU/GPU overlap",             32.83,  -2.05, True),
    ("GPU batch generation",        28.83,  -4.00, True),
    ("tree-center",                 28.28,  -0.55, True),
]

# rtx3060-ab.md 2 -- leave-one-out margins.  The GPU-FLUTE algorithm row is
# derived (loo-flt minus loo-ovl), not measured directly; the doc says so.
LEAVE_ONE_OUT = [
    ("incremental vcost/presum",  47.08,  8.72, True),
    ("incremental wire commit",    6.89,  1.10, True),
    ("GPU-FLUTE algorithm",        5.58,  1.80, False),
    ("GPU batch generation",       4.14, -0.02, True),
    ("CPU/GPU overlap",            1.90,  1.01, True),
    ("tree-center + leaf peeling", 0.55,  0.25, True),
]

# rtx3060-ab.md 1 -- top-level stage split for mempool_group.
# base has no separate finish-nets row: rows under 2% of wall are folded.
STAGES = [
    ("input",                     4.67,  4.67),
    ("S1 route (L-shape)",       41.68, 10.90),
    ("S2 route (DAG/detour)",    43.89, 10.47),
    ("finish nets + output",      0.00,  1.43),
    ("other",                     2.23,  0.82),
]

# rtx3060-ab.md 1, "S1 RSMT decomposition".  Serial runs GPU then CPU; the
# overlapped run hides the whole GPU wall inside the CPU loop (tail = 0.000 s).
OVERLAP = {
    "mempool_group": {"gpu": 2.12, "cpu": 5.11, "overlapped_cpu": 5.16,
                      "gpu_wall": 2.197, "tail": 0.000},
    "bsg_chip":      {"gpu": 1.15, "cpu": 1.56, "overlapped_cpu": 1.61,
                      "gpu_wall": 1.160, "tail": 0.000},
}

CAPTION = "RTX 3060 12GB - one binary, runtime knobs only - 60 runs, 2026-08-26"
