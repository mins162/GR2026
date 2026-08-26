#!/usr/bin/env python3
"""Turn an ab_matrix.sh result directory into runs.csv + SUMMARY.md.

Reads only the run logs and evaluator outputs, so it can be re-run after the
fact on any matrix directory.  Stage rows below the program's own print
threshold (2% of wall for parents, 1% for children) are absent from a log; they
are reported as folded ("--"), never as zero, and rows whose name changes with
the configuration (the S1 RSMT split) are summed into one comparable bucket.
"""
import csv
import os
import re
import sys
from collections import OrderedDict

STAGE_RE = re.compile(r"^(?P<name>.*?)\s\s+(?P<sec>\d+\.\d{2}) s\s+(?P<pct>\d+\.\d) %$")
GRID_RE = re.compile(r"^grid: L=(\d+) X=(\d+) Y=(\d+) cells=(\d+) tracks=(\d+)")
GEN_RE = re.compile(r"Generation \((GPU|CPU)\)\s+(\d+)\s+(\d+)\s+([\d.]+)")
SCORE_RE = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
OVERLAP_RE = re.compile(r"GPU wall=([\d.]+)s, tail beyond CPU loop=([\d.]+)s")
PHASES_RE = re.compile(r"serialized depth phases=(\d+)")
EVAL_TOTAL_RE = re.compile(r"^total cost ([\d.]+)")
EVAL_OPEN_RE = re.compile(r"^Number of open nets : (\d+)")
EVAL_INCOMPLETE_RE = re.compile(r"^Number of incompleted nets : (\d+)")

# The S1 RSMT rows are named after the schedule, so a config with the overlap on
# and one with it off do not share a row name.  Summing them gives the one
# number both configs can be compared on; the split itself is what the overlap
# A/B is about and is reported separately from the GPU wall / tail line.
RSMT_S1 = "S1: RSMT (CPU+GPU total)"

# Order used for the breakdown tables; anything unseen is appended.
STAGE_ORDER = [
    "input", "build CUDA database", "wait for net split",
    "Lshape route", RSMT_S1, "  S1: batch generation", "  S1: DAG build (DFS)",
    "  S1: GPU route batches",
    "DAG/detour route", "  S2: RSMT/DAG preprocessing", "  S2: detour generation",
    "  S2: batch generation", "  S2: host DAG prep + upload", "  S2: GPU route batches",
    "finish nets + final output", "close output", "other", "total",
]
ORDER_IDX = dict((name.strip(), i) for i, name in enumerate(STAGE_ORDER))


def stage_key(name):
    """Table order; rows the program grew since this list was written sort in
    just above the `other` row instead of after `total`."""
    return ORDER_IDX.get(name.strip(), ORDER_IDX["other"] - 0.5)

LADDER = ["base", "l1-vcp", "l2-cmt", "l3-flt", "l4-ovl", "l5-bgen", "opt"]
LADDER_LABEL = OrderedDict([
    ("base", "base (모두 off)"),
    ("l1-vcp", "+ incremental vcost/presum"),
    ("l2-cmt", "+ incremental wire-demand commit"),
    ("l3-flt", "+ GPU-FLUTE (serial)"),
    ("l4-ovl", "+ CPU/GPU 오버랩"),
    ("l5-bgen", "+ GPU batch generation"),
    ("opt", "+ tree-center (= opt)"),
])
LOO_LABEL = OrderedDict([
    ("loo-vcp", "incremental vcost/presum"),
    ("loo-ovl", "CPU/GPU 오버랩"),
    ("loo-bgen", "GPU batch generation"),
    ("loo-flt", "GPU-FLUTE 전체 (알고리즘+오버랩)"),
    ("l5-bgen", "tree-center + leaf peeling"),
    ("loo-cmt", "incremental wire-demand commit"),
])


def parse_log(path):
    run = {"stages": OrderedDict(), "gen": [], "scores": []}
    rsmt_s1 = None
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("config: "):
                run["config_line"] = line[len("config: "):]
                continue
            m = GRID_RE.match(line)
            if m:
                run["grid"] = "L=%s X=%s Y=%s cells=%s tracks=%s" % m.groups()
                continue
            m = GEN_RE.search(line)
            if m:
                run["gen"].append({"mode": m.group(1), "nets": int(m.group(2)),
                                   "batches": int(m.group(3)), "seconds": float(m.group(4))})
                continue
            m = SCORE_RE.match(line)
            if m:
                run["scores"].append([int(v) for v in m.groups()])
                continue
            m = OVERLAP_RE.search(line)
            if m:
                run["gpu_flute_wall"] = float(m.group(1))
                run["gpu_flute_tail"] = float(m.group(2))
                continue
            m = PHASES_RE.search(line)
            if m:
                run["depth_phases"] = int(m.group(1))
                continue
            m = STAGE_RE.match(line)
            if m:
                name, sec = m.group("name"), float(m.group("sec"))
                if name.startswith("other ("):
                    name = "other"
                if name.lstrip().startswith("S1: RSMT"):
                    rsmt_s1 = sec if rsmt_s1 is None else rsmt_s1 + sec
                    run["stages"].setdefault("  " + RSMT_S1, 0.0)
                    run["stages"]["  " + RSMT_S1] = rsmt_s1
                    run.setdefault("rsmt_rows", []).append((name.strip(), sec))
                    continue
                run["stages"][name] = sec
    run["total"] = run["stages"].get("total")
    if run["scores"]:
        run["score_s1"] = run["scores"][0][3]
        run["score_final"] = run["scores"][-1][3]
    return run


def parse_eval(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, errors="replace") as fh:
        for line in fh:
            for key, rx in (("eval_total", EVAL_TOTAL_RE), ("open_nets", EVAL_OPEN_RE),
                            ("incompleted", EVAL_INCOMPLETE_RE)):
                m = rx.match(line)
                if m:
                    out[key] = float(m.group(1)) if key == "eval_total" else int(m.group(1))
    return out


def load(outdir):
    rows = []
    with open(os.path.join(outdir, "runs.tsv")) as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            if not os.path.exists(rec["log"]):
                continue
            run = parse_log(rec["log"])
            run.update(parse_eval(rec.get("eval")))
            run["design"], run["config"] = rec["design"], rec["config"]
            run["rep"], run["status"] = int(rec["rep"]), rec["status"]
            run["gpu_busy"] = rec["gpu_busy"]
            rows.append(run)
    return rows


def median_run(runs):
    """The run whose total is the median; ties and even counts take the lower."""
    ok = [r for r in runs if r["status"] == "ok" and r.get("total") is not None]
    if not ok:
        return None
    ok.sort(key=lambda r: r["total"])
    return ok[(len(ok) - 1) // 2]


def fmt(value, digits=2):
    return "--" if value is None else ("%%.%df" % digits) % value


def pct(numerator, denominator):
    if not denominator:
        return "--"
    return "%+.1f%%" % (100.0 * numerator / denominator)


def write_csv(rows, path):
    stage_names = []
    for r in rows:
        for name in r["stages"]:
            if name not in stage_names:
                stage_names.append(name)
    stage_names.sort(key=stage_key)
    fields = ["design", "config", "rep", "status", "gpu_busy", "total", "score_s1",
              "score_final", "eval_total", "open_nets", "incompleted",
              "gpu_flute_wall", "gpu_flute_tail", "depth_phases",
              "s1_batches", "s2_batches", "s1_gen_mode", "s1_gen_s", "s2_gen_s",
              "grid", "config_line"] + stage_names
    with open(path, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row.update(r["stages"])
            for idx, key in ((0, "s1"), (1, "s2")):
                if len(r["gen"]) > idx:
                    row[key + "_batches"] = r["gen"][idx]["batches"]
                    row[key + "_gen_s"] = r["gen"][idx]["seconds"]
                    row[key + "_gen_mode"] = r["gen"][idx]["mode"]
            writer.writerow(row)


def breakdown_table(base, opt):
    names = []
    for run in (base, opt):
        for name in run["stages"]:
            if name not in names:
                names.append(name)
    names.sort(key=stage_key)
    lines = ["| 항목 | `base` | `opt` | 개선 |", "| --- | ---: | ---: | ---: |"]
    for name in names:
        b, o = base["stages"].get(name), opt["stages"].get(name)
        gain = pct(-(b - o), b) if (b is not None and o is not None and b) else "--"
        label = name.strip()
        if name.startswith("  "):
            label = "&nbsp;&nbsp;" + label
        elif label in ("Lshape route", "DAG/detour route", "total"):
            label = "**%s**" % label
        lines.append("| %s | %s | %s | %s |" % (label, fmt(b), fmt(o), gain))
    return lines


def design_section(design, runs, out):
    by_cfg = OrderedDict()
    for r in runs:
        by_cfg.setdefault(r["config"], []).append(r)
    med = OrderedDict((cfg, median_run(rs)) for cfg, rs in by_cfg.items())

    out.append("## %s" % design)
    out.append("")
    any_run = next((r for r in runs if r.get("grid")), None)
    if any_run:
        out.append("- grid : `%s`" % any_run["grid"])
    bad = [r for r in runs if r["status"] != "ok"]
    if bad:
        out.append("- **실패/불일치 %d건** : %s" %
                   (len(bad), ", ".join("%s r%d (%s)" % (r["config"], r["rep"], r["status"])
                                        for r in bad)))
    out.append("")

    out.append("### 재현성")
    out.append("")
    out.append("| config | n | min | median | max | 편차 |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cfg, rs in by_cfg.items():
        totals = sorted(r["total"] for r in rs if r["status"] == "ok" and r.get("total"))
        if not totals:
            out.append("| `%s` | 0 | -- | -- | -- | -- |" % cfg)
            continue
        spread = 100.0 * (totals[-1] - totals[0]) / totals[0] if totals[0] else 0.0
        out.append("| `%s` | %d | %s | %s | %s | %.1f%% |" %
                   (cfg, len(totals), fmt(totals[0]), fmt(med[cfg]["total"]),
                    fmt(totals[-1]), spread))
    out.append("")

    base, opt = med.get("base"), med.get("opt")
    if base and opt:
        out.append("### 1. 앵커 — base vs opt")
        out.append("")
        out.append("- **total %s s → %s s (%.2f×, %s)**" %
                   (fmt(base["total"]), fmt(opt["total"]),
                    base["total"] / opt["total"] if opt["total"] else 0,
                    pct(-(base["total"] - opt["total"]), base["total"])))
        if base["gen"] and opt["gen"]:
            out.append("- batch : `base` S1 %d / S2 %s · `opt` S1 %d / S2 %s" % (
                base["gen"][0]["batches"],
                base["gen"][1]["batches"] if len(base["gen"]) > 1 else "--",
                opt["gen"][0]["batches"],
                opt["gen"][1]["batches"] if len(opt["gen"]) > 1 else "--"))
        out.append("")
        out.extend(breakdown_table(base, opt))
        out.append("")

    if opt and base and any(med.get(cfg) for cfg in LOO_LABEL):
        span = base["total"] - opt["total"]
        out.append("### 2. 기여별 마진 (leave-one-out) — opt에서 하나씩 끔")
        out.append("")
        out.append("| 기여 | config | total | Δ vs opt | 전체 절감 대비 |")
        out.append("| --- | --- | ---: | ---: | ---: |")
        for cfg, label in LOO_LABEL.items():
            run = med.get(cfg)
            if not run:
                continue
            delta = run["total"] - opt["total"]
            out.append("| %s | `%s` | %s | **%+.2f s** | %s |" %
                       (label, cfg, fmt(run["total"]), delta,
                        "%.1f%%" % (100.0 * delta / span) if span else "--"))
        loo_flt, loo_ovl = med.get("loo-flt"), med.get("loo-ovl")
        if loo_flt and loo_ovl:
            out.append("| &nbsp;&nbsp;— GPU-FLUTE 알고리즘 단독 | 파생 | -- | **%+.2f s** | %s |" % (
                loo_flt["total"] - loo_ovl["total"],
                "%.1f%%" % (100.0 * (loo_flt["total"] - loo_ovl["total"]) / span) if span else "--"))
        out.append("")
        out.append("- Δ vs opt = 그 기여를 껐을 때 늘어나는 시간. 클수록 기여가 큼")
        out.append("- 마진의 합은 `base − opt` = %.2f s와 일치하지 않는다 (상호작용) — 차이는 §3과 대조" % span)
        out.append("")

    # Only a complete ladder is printable: with a rung missing the remaining
    # rows would still be labelled "+ <that rung's contribution>" while the step
    # actually spans everything skipped in between.
    if all(med.get(c) for c in LADDER):
        out.append("### 3. 누적 사다리 (base → opt)")
        out.append("")
        out.append("| 단계 | config | total | 이 단계 Δ | 누적 Δ |")
        out.append("| --- | --- | ---: | ---: | ---: |")
        prev = None
        for cfg in LADDER:
            run = med.get(cfg)
            if not run:
                continue
            step = "--" if prev is None else "%+.2f s" % (run["total"] - prev)
            cum = "%+.2f s" % (run["total"] - med["base"]["total"])
            out.append("| %s | `%s` | %s | %s | %s |" %
                       (LADDER_LABEL[cfg], cfg, fmt(run["total"]), step, cum))
            prev = run["total"]
        out.append("")

    out.append("### 4. 품질")
    out.append("")
    out.append("| config | rep | 프로그램 report | evaluator | open | incompleted |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cfg, rs in by_cfg.items():
        for r in sorted(rs, key=lambda r: r["rep"]):
            if "eval_total" not in r and "score_final" not in r:
                continue
            out.append("| `%s` | %d | %s | %s | %s | %s |" % (
                cfg, r["rep"],
                "{:,}".format(r["score_final"]) if r.get("score_final") else "--",
                "{:,.0f}".format(r["eval_total"]) if r.get("eval_total") else "--",
                r.get("open_nets", "--"), r.get("incompleted", "--")))
    out.append("")

    ovl_on, ovl_off = med.get("opt"), med.get("loo-ovl")
    if ovl_on and ovl_off and "gpu_flute_wall" in ovl_on:
        out.append("### 5. 오버랩 상세 (S1 RSMT)")
        out.append("")
        out.append("| config | GPU wall | tail beyond CPU loop | S1 RSMT 행 |")
        out.append("| --- | ---: | ---: | --- |")
        for cfg in ("opt", "loo-ovl", "loo-flt"):
            r = med.get(cfg)
            if not r:
                continue
            rows = " + ".join("%s %.2f s" % (n, s) for n, s in r.get("rsmt_rows", []))
            out.append("| `%s` | %s | %s | %s |" % (
                cfg, fmt(r.get("gpu_flute_wall"), 3), fmt(r.get("gpu_flute_tail"), 3),
                rows or "--"))
        out.append("")

    phases = [(cfg, r.get("depth_phases")) for cfg, r in med.items() if r and r.get("depth_phases")]
    if phases:
        out.append("- S2 critical path phases : " +
                   ", ".join("`%s` %d" % (c, p) for c, p in phases))
        out.append("")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load(outdir)
    if not rows:
        sys.exit("no runs found in %s" % outdir)
    write_csv(rows, os.path.join(outdir, "runs.csv"))

    out = ["# A/B 매트릭스 결과", "",
           "- 디렉터리 : `%s`" % os.path.abspath(outdir),
           "- 런 %d건 (실패/불일치 %d건)" % (len(rows), sum(1 for r in rows if r["status"] != "ok")),
           "- 각 표의 수치는 config별 **median 런**. 개별 런은 `runs.csv`",
           "- 환경 : `ENV.txt`", ""]
    designs = []
    for r in rows:
        if r["design"] not in designs:
            designs.append(r["design"])
    for design in designs:
        design_section(design, [r for r in rows if r["design"] == design], out)

    path = os.path.join(outdir, "SUMMARY.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
