#!/usr/bin/env python3
"""
CLEAR MOT metrics (MOTA, IDSW, precision) and FPS per tracker, from the CSVs written by export_csv.py
(either protocol): <dir>/<sequence>_<video stem>_gt.csv and <dir>/<sequence>_<video stem>_<tracker>.csv, rows
frame,id,x1,y1,x2,y2,...; a tracker row counts as a reported box only if it has a box and, when the column exists,
pred_exist == 1.

Per frame, GT and tracker boxes are matched at IoU >= 0.5 (CLEAR MOT: a GT id keeps its tracker id from the previous
match while their IoU stays >= 0.5, the rest are assigned by the Hungarian algorithm on 1 - IoU):
    TP / FP / FN     matched tracker boxes / unmatched tracker boxes / unmatched GT boxes
    IDSW             a GT id matched to a different tracker id than at its last match
    MOTA             1 - (FN + FP + IDSW) / #GT boxes
    precision        TP / (TP + FP)
    fps              tracker time only (<dir>/timing/<video>.csv, ms_per_frame), no video decode
Per tracker the counts are summed over all sequences before MOTA / precision are computed; fps is the mean over
sequences.

Out:       <dir>/mot_metrics/<tracker>.csv   one row per sequence + a TOTAL row
           <dir>/mot_metrics/summary.csv     the TOTAL row of every tracker
Run:       python mot_metrics.py --dir runs/csv_gt
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from metrics import MATCH_IOU, iou_xyxy, read_timing


def load_boxes(path):
    """-> {frame: [(id, np.array([x1, y1, x2, y2])), ...]} of the reported boxes."""
    out = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["x1"] == "" or r.get("pred_exist", "1") in ("0", "0.0"):
            continue
        box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        out[int(r["frame"])].append((r["id"], box))
    return out


def clear_mot(gt, tr):
    """gt, tr: load_boxes() dicts of one sequence. -> dict of counts."""
    tp = fp = fn = idsw = n_gt = 0
    last = {}  # gt id -> tracker id at its last match
    for frame in sorted(set(gt) | set(tr)):
        g, t = gt.get(frame, []), tr.get(frame, [])
        n_gt += len(g)
        if not g or not t:
            fp, fn = fp + len(t), fn + len(g)
            continue
        iou = iou_xyxy(np.stack([b for _, b in g])[:, None], np.stack([b for _, b in t])[None])
        pairs = []
        # keep last frame's correspondences that still overlap enough
        t_idx = {tid: j for j, (tid, _) in enumerate(t)}
        for i, (gid, _) in enumerate(g):
            j = t_idx.get(last.get(gid))
            if j is not None and iou[i, j] >= MATCH_IOU:
                pairs.append((i, j))
        gi = [i for i in range(len(g)) if i not in {p[0] for p in pairs}]
        tj = [j for j in range(len(t)) if j not in {p[1] for p in pairs}]
        if gi and tj:
            sub = iou[np.ix_(gi, tj)]
            rows, cols = linear_sum_assignment(1 - sub)
            pairs += [(gi[r], tj[c]) for r, c in zip(rows, cols) if sub[r, c] >= MATCH_IOU]
        for i, j in pairs:
            gid, tid = g[i][0], t[j][0]
            if gid in last and last[gid] != tid:
                idsw += 1
            last[gid] = tid
        tp += len(pairs)
        fp += len(t) - len(pairs)
        fn += len(g) - len(pairs)
    return {"gt_boxes": n_gt, "tp": tp, "fp": fp, "fn": fn, "idsw": idsw}


def scores(c):
    return {"mota": 1 - (c["fn"] + c["fp"] + c["idsw"]) / c["gt_boxes"] if c["gt_boxes"] else np.nan,
            "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else np.nan}


HEAD = ["sequence", "mota", "idsw", "precision", "fps", "gt_boxes", "tp", "fp", "fn"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="runs/csv_gt")
    ap.add_argument("--video-stem", default="visible", help="video file stem used in the CSV names")
    ap.add_argument("--out", default=None, help="output dir (default <dir>/mot_metrics)")
    args = ap.parse_args()
    d = Path(args.dir)
    out = Path(args.out) if args.out else d / "mot_metrics"
    out.mkdir(parents=True, exist_ok=True)
    timing = read_timing(d)
    sep = f"_{args.video_stem}_"  # <sequence>_visible_<tracker>; tracker keys contain "_" (hit_s, ...)

    by = defaultdict(list)
    gt_cache = {}
    for p in sorted(d.glob(f"*{sep}*.csv")):
        seq, tracker = p.stem.split(sep, 1)
        if tracker == "gt":
            continue
        video = f"{seq}_{args.video_stem}"
        gt_path = d / f"{video}_gt.csv"
        if not gt_path.exists():
            print(f"[skip] {p.name}: no {gt_path.name}")
            continue
        if video not in gt_cache:
            gt_cache[video] = load_boxes(gt_path)
        c = clear_mot(gt_cache[video], load_boxes(p))
        ms = timing.get((video, tracker))
        by[tracker].append({"sequence": video, **c, **scores(c), "fps": 1000 / ms if ms else np.nan})

    fmt = lambda v: round(v, 4) if isinstance(v, float) else v
    summary = []
    for tracker, ms in sorted(by.items()):
        ms.sort(key=lambda m: m["sequence"])
        tot = {k: sum(m[k] for m in ms) for k in ("gt_boxes", "tp", "fp", "fn", "idsw")}
        fps = [m["fps"] for m in ms if not np.isnan(m["fps"])]
        total = {"sequence": f"TOTAL ({len(ms)} videos)", **tot, **scores(tot),
                 "fps": float(np.mean(fps)) if fps else np.nan}
        with open(out / f"{tracker}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEAD)
            for m in ms + [total]:
                w.writerow([fmt(m[k]) for k in HEAD])
        summary.append({"tracker": tracker, **total})

    summary.sort(key=lambda s: -np.nan_to_num(s["mota"], nan=-np.inf))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracker", "videos", *HEAD[1:]])
        for s in summary:
            w.writerow([s["tracker"], len(by[s["tracker"]]), *(fmt(s[k]) for k in HEAD[1:])])

    print(f"{'tracker':<14}{'MOTA':>9}{'IDSW':>7}{'Precision':>11}{'FPS':>9}")
    for s in summary:
        print(f"{s['tracker']:<14}{s['mota']:>9.3f}{s['idsw']:>7d}{s['precision']:>11.3f}{s['fps']:>9.1f}")
    print(f"\n[saved] {len(summary)} tracker CSVs + summary.csv -> {out}")


if __name__ == "__main__":
    main()
