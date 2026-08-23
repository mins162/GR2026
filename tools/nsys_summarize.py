#!/usr/bin/env python3
"""Turn `nsys stats` CSV dumps into the one table the vcost/presum question needs.

Reads a directory produced by tools/nsys_profile.sh and, for every
design x config it finds, prints where GPU time actually went: per-kernel time
from the driver, grouped into pipeline buckets, against the program's own
wall clock.  Kernel time and launch counts come from nsys, so nothing here
depends on the cudaEvent instrumentation inside the router.

Usage: tools/nsys_summarize.py <nsys_results_dir>
"""
import csv
import glob
import os
import re
import sys

# First match wins, so the specific names come before the substrings that would
# also swallow them (compute_presum_general is part of the commit path, not of
# the cost refresh that compute_presum does).
BUCKETS = [
    ("vcost rebuild",      ["update_vcost", "update_wcost"]),
    ("commit (demand)",    ["compute_presum_general", "commit_wire_demand",
                            "commit_via_demand", "commit_all_edge"]),
    ("presum (wire cost)", ["compute_presum"]),
    ("DP route",           ["Lshape_route_cuda", "Lshape_route_node_cuda"]),
    ("traceback",          ["get_routing_tree_cuda"]),
    ("GPU FLUTE",          ["break_kernel", "candidate_size_kernel", "estimate_break_kernel",
                            "finish_x_sort_kernel", "make_hanan_kernel", "make_x_sort_keys_kernel",
                            "merge_level_kernel", "solve_leaf_kernel", "tree_center_kernel",
                            "tree_size_kernel"]),
    ("congestion / score", ["extract_congestionView", "mark_overflow", "add_all_overflow"]),
    ("validation",         ["compare_vcost", "compare_presum", "compute_presum_reference"]),
    ("bookkeeping",        ["reset_dirty_state", "init_costs", "init_min_child_costs", "init_road"]),
]


def bucket_of(name):
    for label, keys in BUCKETS:
        if any(k in name for k in keys):
            return label
    return "other"


def find_col(header, *wanted):
    """Column index whose header contains any of `wanted` (nsys renames these)."""
    for i, h in enumerate(header):
        low = h.lower()
        if any(w in low for w in wanted):
            return i
    return None


def read_stats(path):
    """[(name, total_ns, instances)] from an nsys stats CSV."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        return []
    header = rows[0]
    # kernel reports call it "Name"; the NVTX reports call it "Range".
    i_name = find_col(header, "name", "range")
    i_time = find_col(header, "total time", "duration")
    i_num = find_col(header, "instances", "num calls", "count")
    if i_name is None or i_time is None:
        return []
    out = []
    for r in rows[1:]:
        if len(r) <= max(i_name, i_time):
            continue
        try:
            total = float(r[i_time].replace(",", ""))
        except ValueError:
            continue
        try:
            num = int(float(r[i_num].replace(",", ""))) if i_num is not None else 0
        except (ValueError, IndexError):
            num = 0
        out.append((r[i_name].strip(), total, num))
    return out


def read_log(path):
    """Wall clock, grid geometry and config line from the router's own stdout."""
    info = {"wall": None, "grid": None, "config": None, "batches": []}
    if not os.path.exists(path):
        return info
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("grid:"):
                info["grid"] = line.strip()
            elif line.startswith("config:"):
                info["config"] = line.strip()
            elif line.startswith("total "):
                m = re.match(r"total\s+([\d.]+) s", line)
                if m:
                    info["wall"] = float(m.group(1))
            elif "Generation" in line:
                nums = re.findall(r"\d+", line)
                if len(nums) >= 2:
                    info["batches"].append(int(nums[1]))
    return info


def fmt_s(ns):
    return "%8.2f" % (ns / 1e9)


def report(out_dir, tag):
    kern = (glob.glob(os.path.join(out_dir, tag + "_cuda_gpu_kern_sum.csv")) or
            glob.glob(os.path.join(out_dir, tag + "_gpukernsum.csv")))
    if not kern:
        return None
    rows = read_stats(kern[0])
    if not rows:
        return None
    log = read_log(os.path.join(out_dir, tag + ".log"))
    gpu_ns = sum(t for _, t, _ in rows)
    wall = log["wall"]

    print("=" * 78)
    print("%s" % tag)
    if log["config"]:
        print("  " + log["config"])
    if log["grid"]:
        print("  " + log["grid"])
    print("-" * 78)
    if wall:
        print("  wall clock (program self-report) : %8.2f s" % wall)
    print("  GPU busy (sum of kernel time)    : %8.2f s%s" % (
        gpu_ns / 1e9,
        "   (%.1f%% of wall)" % (100 * gpu_ns / 1e9 / wall) if wall else ""))
    if wall:
        print("  no kernel running                : %8.2f s   (%.1f%% of wall)" % (
            wall - gpu_ns / 1e9, 100 * (1 - gpu_ns / 1e9 / wall)))
    if log["batches"]:
        print("  batches per stage                : %s" %
              ", ".join(str(b) for b in log["batches"]))
    print()

    per_bucket = {}
    for name, total, num in rows:
        b = bucket_of(name)
        t, n = per_bucket.get(b, (0.0, 0))
        per_bucket[b] = (t + total, n + num)

    print("  %-20s %10s %8s %8s %12s" % ("bucket", "GPU time", "%GPU", "%wall", "launches"))
    for b, (t, n) in sorted(per_bucket.items(), key=lambda kv: -kv[1][0]):
        print("  %-20s %s s %7.1f%% %7s %12d" % (
            b, fmt_s(t), 100 * t / gpu_ns if gpu_ns else 0,
            "%.1f%%" % (100 * t / 1e9 / wall) if wall else "-", n))
    print()
    print("  top kernels")
    for name, total, num in sorted(rows, key=lambda r: -r[1])[:12]:
        # nsys prints the full signature; the argument list adds no information
        # here and pushes the namespace off the right edge.
        print("  %-46s %s s %12d" % (name.split("(")[0][:46], fmt_s(total), num))

    # Host side.  GPU busy well under wall clock means the answer is here, not
    # in the kernel table: a synchronize that dominates cudaLaunchKernel is the
    # host waiting on the GPU, while launch time dominating means the opposite.
    api = (glob.glob(os.path.join(out_dir, tag + "_cuda_api_sum.csv")) or
           glob.glob(os.path.join(out_dir, tag + "_cudaapisum.csv")))
    if api:
        arows = read_stats(api[0])
        if arows:
            print()
            print("  host-side CUDA API time (blocking calls include GPU wait)")
            for name, total, num in sorted(arows, key=lambda r: -r[1])[:8]:
                print("  %-46s %s s %12d%s" % (
                    name.split("(")[0][:46], fmt_s(total), num,
                    "  (%.1f%% of wall)" % (100 * total / 1e9 / wall) if wall else ""))

    nvtx = (glob.glob(os.path.join(out_dir, tag + "_nvtx_gpu_proj_sum.csv")) or
            glob.glob(os.path.join(out_dir, tag + "_nvtx_sum.csv")))
    if nvtx:
        nrows = read_stats(nvtx[0])
        if nrows:
            print()
            print("  NVTX ranges (GPU time projected onto the range)")
            for name, total, num in sorted(nrows, key=lambda r: -r[1])[:14]:
                print("  %-46s %s s %12d" % (name[:46], fmt_s(total), num))
    print()
    return {"tag": tag, "wall": wall, "gpu": gpu_ns, "buckets": per_bucket}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = sys.argv[1]
    tags = sorted({os.path.basename(p).split("_cuda_gpu_kern_sum")[0]
                                      .split("_gpukernsum")[0]
                   for p in glob.glob(os.path.join(out_dir, "*_cuda_gpu_kern_sum.csv")) +
                            glob.glob(os.path.join(out_dir, "*_gpukernsum.csv"))})
    if not tags:
        print("no kernel-summary CSVs in %s" % out_dir)
        return 1

    results = [r for r in (report(out_dir, t) for t in tags) if r]
    if len(results) < 2:
        return 0

    print("=" * 78)
    print("vcost + presum, side by side")
    print("=" * 78)
    print("  %-28s %9s %9s %9s %9s" % ("run", "wall", "GPU busy", "vcost", "presum"))
    for r in results:
        v = r["buckets"].get("vcost rebuild", (0, 0))[0] / 1e9
        p = r["buckets"].get("presum (wire cost)", (0, 0))[0] / 1e9
        print("  %-28s %8s %8.2fs %8.2fs %8.2fs" % (
            r["tag"], ("%.2fs" % r["wall"]) if r["wall"] else "-",
            r["gpu"] / 1e9, v, p))
    print()
    print("  Read it as: the two right-hand columns are what the optimization can")
    print("  ever remove.  Compare them against the wall column, not against each")
    print("  other -- an X% cut of a bucket that is 2% of wall clock is 0.02X% of")
    print("  the run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
