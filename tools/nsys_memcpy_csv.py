"""Dump the CPU<->GPU traffic of an nsys sqlite export as one compact CSV.

    python3 tools/nsys_memcpy_csv.py <profile.sqlite> <out.csv.gz> [t0 t1 ...]

Rows are (kind, start_s, end_s, bytes, name) on the nsys clock:
  K  GPU busy interval (union of every kernel, so 59k launches become a few
     thousand gaps-and-runs)
  H  Host-to-Device memcpy         D  Device-to-Host memcpy
  R  host-side cudaMemcpy call (wall time the CPU spent inside the API)
  A  any CUDA runtime call on the main thread (the one with the most calls);
     its complement is the time the CPU spent on its own work
  N  one row per kernel, with its name, inside each optional [t0, t1] window
     in seconds -- for zooming into a single batch
The .sqlite itself is gitignored (nsys_results_*/); the CSV is what the
visualization scripts read.
"""
import csv
import gzip
import sqlite3
import sys


def merged(rows):
    out = []
    for s, e in sorted(rows):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main(db_path, out_path, windows=()):
    db = sqlite3.connect(db_path)
    rows = []
    for t0, t1 in windows:
        for s, e, name in db.execute(
                "select k.start, k.end, n.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                "join StringIds n on k.shortName = n.id where k.start between ? and ?",
                (t0 * 1e9, t1 * 1e9)):
            rows.append(("N", s, e, 0, name))
    for s, e in merged(db.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")):
        rows.append(("K", s, e, 0))
    for s, e, b, kind in db.execute(
            "select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        rows.append(({1: "H", 2: "D"}.get(kind, "?"), s, e, b))
    for s, e in db.execute(
            "select r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds n on r.nameId = n.id where n.value like 'cudaMemcpy%'"):
        rows.append(("R", s, e, 0))
    main_tid = db.execute("select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
                          "group by globalTid order by count(*) desc limit 1").fetchone()[0]
    for s, e in merged(db.execute(
            "select start, end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?", (main_tid,))):
        rows.append(("A", s, e, 0))
    rows.sort(key=lambda r: r[1])
    with gzip.open(out_path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "start_s", "end_s", "bytes", "name"])
        for k, s, e, b, *name in rows:
            w.writerow([k, "%.6f" % (s / 1e9), "%.6f" % (e / 1e9), b] + (name or [""]))
    print("%s: %d rows" % (out_path, len(rows)))


if __name__ == "__main__":
    ts = [float(t) for t in sys.argv[3:]]
    main(sys.argv[1], sys.argv[2], list(zip(ts[::2], ts[1::2])))
