#!/usr/bin/env python3
"""
Single-object tracking metrics from the per-frame CSVs written by export_csv.py --protocol gt (one row per frame,
columns frame, x1..cy, score, pred_exist, gt_exist, gt_*, iou, center_err, norm_center_err, match).

Per sequence and tracker (metrics_per_sequence.csv), then per tracker (metrics_summary.csv):
  OPE (OTB / LaSOT / UAV123), on frames where the target exists, using the box the tracker outputs every frame:
    mean_iou         mean IoU
    success_auc      area under the success plot: mean over IoU thresholds 0, 0.05, ..., 1 of P(IoU > t)
    success_50       P(IoU > 0.5)
    precision_20     P(center error <= 20 px)
    norm_prec_auc    LaSOT normalised precision: AUC of P(center error / GT size <= t), t in 0..0.5
  Anti-UAV (CVPR'20 / '21 challenge), over all frames, needs "no target" predictions:
    sa               state accuracy  sum(IoU_t * [v_t] + p_t * [not v_t]) / T  -  0.2 * (sum(p_t * [v_t]) / T*)^0.3
                     v_t = target visible, p_t = tracker reports no target (score < its lost threshold), T* = #v_t
  MOT-style (CLEAR MOT, one target, match = reported box with IoU >= 0.5):
    mota             1 - (FN + FP + IDSW) / #GT     (IDSW = 0: one id per sequence under the GT-init protocol)
    motp             mean IoU of matches
    precision/recall/f1   of the reported boxes vs GT at IoU >= 0.5
  fps               tracker only (from timing/), no video decode
Summary: OPE, SA and FPS are the mean over sequences (as the benchmarks report them); MOT counts are summed over all
frames first, then MOTA / MOTP / precision / recall / F1 are computed from the totals.

Run:       python metrics.py --dir runs/csv_gt
           python metrics.py --dir runs/val          # eval_tracker.py output, one sub-directory per tracker
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

IOU_T = np.linspace(0, 1, 21)
NORM_T = np.linspace(0, 0.5, 51)
MATCH_IOU = 0.5


def iou_xyxy(a, b):
    """IoU of boxes (..., 4) x1, y1, x2, y2."""
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2, y2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: np.clip(r[..., 2] - r[..., 0], 0, None) * np.clip(r[..., 3] - r[..., 1], 0, None)
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def sequence_metrics(rows):
    """rows: dicts from one tracker CSV (all frames). -> dict of metrics."""
    f = lambda k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    gt = f("gt_exist").astype(bool)
    has_box = ~np.isnan(f("x1"))
    pred = f("pred_exist").astype(bool)
    iou, cerr, ncerr = f("iou"), f("center_err"), f("norm_center_err")
    m = {"frames": len(rows), "gt_frames": int(gt.sum())}

    v = gt & has_box  # OPE: box on every frame after init; frames with no box (before init) count as a miss
    iou_v = np.where(v, iou, 0.0)[gt]
    cerr_v = np.where(v, cerr, np.inf)[gt]
    ncerr_v = np.where(v, ncerr, np.inf)[gt]
    n = max(gt.sum(), 1)
    m["mean_iou"] = iou_v.mean() if gt.any() else np.nan
    m["success_auc"] = np.mean([(iou_v > t).sum() / n for t in IOU_T]) if gt.any() else np.nan
    m["success_50"] = (iou_v > 0.5).sum() / n if gt.any() else np.nan
    m["precision_20"] = (cerr_v <= 20).sum() / n if gt.any() else np.nan
    m["norm_prec_auc"] = np.mean([(ncerr_v <= t).sum() / n for t in NORM_T]) if gt.any() else np.nan

    report = pred & has_box
    empty = ~report
    iou_r = np.where(report, np.nan_to_num(iou), 0.0)
    T, Tv = len(rows), max(gt.sum(), 1)
    m["sa"] = (np.sum(iou_r * gt + empty * ~gt) / T - 0.2 * (np.sum(empty * gt) / Tv) ** 0.3) if T else np.nan

    match = report & gt & (np.nan_to_num(iou) >= MATCH_IOU)
    m["tp"], m["fp"], m["fn"] = int(match.sum()), int((report & ~match).sum()), int((gt & ~match).sum())
    m["idsw"] = 0
    m["iou_sum_tp"] = float(np.nan_to_num(iou)[match].sum())
    m.update(mot_scores(m["tp"], m["fp"], m["fn"], m["idsw"], m["iou_sum_tp"], m["gt_frames"]))
    return m


def mot_scores(tp, fp, fn, idsw, iou_sum_tp, n_gt):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    return {"mota": 1 - (fn + fp + idsw) / n_gt if n_gt else np.nan,
            "motp": iou_sum_tp / tp if tp else np.nan,
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if tp else 0.0}


def read_timing(d):
    out = {}
    for p in [*(d / "timing").glob("*.csv"), *d.glob("*/timing/*.csv")]:  # export_csv.py / eval_tracker.py
        for r in csv.DictReader(open(p)):
            out[(p.stem, r["tracker"])] = float(r["ms_per_frame"])
    return out


COLS = ["mean_iou", "success_auc", "success_50", "precision_20", "norm_prec_auc", "sa",
        "mota", "motp", "precision", "recall", "f1", "fps"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="runs/csv_gt")
    ap.add_argument("--video-stem", default="visible", help="video file stem used in the CSV names")
    args = ap.parse_args()
    d = Path(args.dir)
    timing = read_timing(d)
    per_seq = []
    for p in sorted([*d.glob("*.csv"), *d.glob("*/*.csv")]):  # flat (export_csv.py) or <tracker>/ (eval_tracker.py)
        if p.stem.endswith("_gt") or p.name.startswith("metrics"):
            continue
        rows = list(csv.DictReader(open(p)))
        if not rows or "gt_exist" not in rows[0]:
            continue
        sep = f"_{args.video_stem}_"  # <sequence>_visible_<tracker>; tracker keys contain "_" (hit_s, ...)
        if sep not in p.stem:
            continue
        seq, tracker = p.stem.split(sep, 1)
        video = f"{seq}_{args.video_stem}"
        m = sequence_metrics(rows)
        ms = timing.get((video, tracker))
        m["fps"] = 1000 / ms if ms else np.nan
        per_seq.append({"sequence": video, "tracker": tracker, **m})

    fmt = lambda v: round(v, 4) if isinstance(v, float) else v
    head = ["sequence", "tracker", "frames", "gt_frames", *COLS, "tp", "fp", "fn", "idsw"]
    with open(d / "metrics_per_sequence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for m in sorted(per_seq, key=lambda m: (m["tracker"], m["sequence"])):
            w.writerow([fmt(m[k]) for k in head])

    by = defaultdict(list)
    for m in per_seq:
        by[m["tracker"]].append(m)
    summary = []
    for tracker, ms in by.items():
        mean = lambda k: float(np.nanmean([m[k] for m in ms]))
        tot = {k: sum(m[k] for m in ms) for k in ("tp", "fp", "fn", "idsw", "iou_sum_tp", "gt_frames", "frames")}
        s = {"tracker": tracker, "sequences": len(ms), "frames": tot["frames"], "gt_frames": tot["gt_frames"],
             **{k: mean(k) for k in ("mean_iou", "success_auc", "success_50", "precision_20", "norm_prec_auc",
                                    "sa", "fps")},
             **mot_scores(tot["tp"], tot["fp"], tot["fn"], tot["idsw"], tot["iou_sum_tp"], tot["gt_frames"]),
             **{k: tot[k] for k in ("tp", "fp", "fn", "idsw")}}
        summary.append(s)
    summary.sort(key=lambda s: -s["success_auc"])
    head = ["tracker", "sequences", "frames", "gt_frames", *COLS, "tp", "fp", "fn", "idsw"]
    with open(d / "metrics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for s in summary:
            w.writerow([fmt(s[k]) for k in head])

    # one table: per tracker its rows per video, then its AVERAGE row (= the metrics_summary.csv row)
    head = ["tracker", "sequence", "frames", "gt_frames", *COLS, "tp", "fp", "fn", "idsw"]
    with open(d / "metrics_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for s in summary:
            for m in sorted(by[s["tracker"]], key=lambda m: m["sequence"]):
                w.writerow([fmt(m[k]) for k in head])
            w.writerow([fmt(s[k]) if k != "sequence" else f"AVERAGE ({s['sequences']} videos)" for k in head])

    print(f"{'tracker':<13}" + "".join(f"{c:>13}" for c in COLS))
    for s in summary:
        print(f"{s['tracker']:<13}" + "".join(f"{s[c]:>13.3f}" for c in COLS))
    print(f"\n[saved] {d / 'metrics_per_sequence.csv'} ({len(per_seq)} rows), {d / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
