#!/usr/bin/env python3
"""
Collect the per-tracker summaries of every evaluation into one CSV and print them as README tables.

Datasets (skipped when the run does not exist yet):
    val10         runs/eval            10 of the 67 val videos (eval_tracker.py --count 10)
    lasot10       runs/lasot_drone10   10 LaSOT drone sequences (4 test + 6 train)
    lasot_test    runs/lasot_drone10   only the 4 LaSOT test sequences of the run above (drone-2, -7, -13, -15)
FPS: val10 takes it from the timing pass runs/fps (each tracker alone on the GPU, 3 val videos); the LaSOT rows take
it from their own run, which was also one tracker at a time.
Type: "sot" = initialised on the GT box, runs alone; "det" = YOLO detector every frame (FPS includes it).

Out:       results/results.csv   one row per dataset x tracker
Run:       python metrics.py --dir runs/eval && python metrics.py --dir runs/lasot_drone10
           python collect_results.py            # writes results/results.csv, prints markdown tables
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_tracker import DET_TRACKERS
from metrics import mot_scores

LASOT_TEST = {"drone-2", "drone-7", "drone-13", "drone-15"}
OPE = ["mean_iou", "success_auc", "success_50", "precision_20", "norm_prec_auc", "sa"]
MOT = ["mota", "motp", "precision", "recall", "f1"]
HEAD = ["dataset", "tracker", "type", "sequences", "frames", "gt_frames", *OPE, *MOT, "fps"]
TABLE = [("AUC", "success_auc"), ("Success@0.5", "success_50"), ("Prec@20px", "precision_20"),
         ("Norm. prec", "norm_prec_auc"), ("SA", "sa"), ("MOTA", "mota"), ("F1", "f1"), ("FPS", "fps")]


def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def summarise(rows, keep=lambda seq: True):
    """metrics_per_sequence.csv rows -> per tracker: OPE / SA / FPS mean over sequences, MOT from summed counts."""
    by = defaultdict(list)
    for r in rows:
        if keep(r["sequence"]):
            by[r["tracker"]].append(r)
    out = {}
    for t, rs in by.items():
        f = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rs])
        tot = {k: int(f(k).sum()) for k in ("tp", "fp", "fn", "idsw", "gt_frames", "frames")}
        iou_sum = float(np.nansum(f("motp") * f("tp")))
        out[t] = {"sequences": len(rs), "frames": tot["frames"], "gt_frames": tot["gt_frames"],
                  **{k: float(np.nanmean(f(k))) for k in OPE + ["fps"]},
                  **mot_scores(tot["tp"], tot["fp"], tot["fn"], tot["idsw"], iou_sum, tot["gt_frames"])}
    return out


def main():
    fps_pass = {r["tracker"]: float(r["fps"]) for r in read("runs/fps/metrics_summary.csv")}
    lasot = read("runs/lasot_drone10/metrics_per_sequence.csv")
    datasets = {"val10": summarise(read("runs/eval/metrics_per_sequence.csv")),
                "lasot10": summarise(lasot),
                "lasot_test": summarise(lasot, lambda s: s.split("_")[0] in LASOT_TEST)}
    for t, m in datasets["val10"].items():
        m["fps"] = fps_pass.get(t, np.nan)

    rows = []
    for ds, res in datasets.items():
        for t, m in sorted(res.items(), key=lambda kv: -kv[1]["success_auc"]):
            rows.append({"dataset": ds, "tracker": t, "type": "det" if t in DET_TRACKERS else "sot", **m})
    rows.sort(key=lambda r: (list(datasets).index(r["dataset"]), -r["success_auc"]))
    Path("results").mkdir(exist_ok=True)
    fmt = lambda v: round(v, 4) if isinstance(v, float) else v
    with open("results/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for r in rows:
            w.writerow([fmt(r[k]) for k in HEAD])
    print(f"[saved] results/results.csv ({len(rows)} rows)\n")

    for ds in datasets:
        sel = [r for r in rows if r["dataset"] == ds]  # already sorted by AUC
        if not sel:
            continue
        best = {k: max(r[k] for r in sel) for _, k in TABLE}
        print(f"### {ds}\n")
        print("| Tracker | Type | " + " | ".join(n for n, _ in TABLE) + " |")
        print("|---" * (len(TABLE) + 2) + "|")
        for r in sel:
            cell = lambda k: (f"{r[k]:.0f}" if k == "fps" else f"{r[k]:.3f}")
            print(f"| {r['tracker']} | {r['type']} | " + " | ".join(f"**{cell(k)}**" if r[k] == best[k] else cell(k)
                                                               for _, k in TABLE) + " |")
        print(f"\n{sel[0]['sequences']} sequences, {sel[0]['frames']} frames, {sel[0]['gt_frames']} with target\n")

if __name__ == "__main__":
    main()
