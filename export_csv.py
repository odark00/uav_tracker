#!/usr/bin/env python3
"""
Write ground truth and tracker output of every video as CSVs (original-resolution pixels), for metrics.py.

--protocol gt (default, the standard one-pass evaluation of OTB / LaSOT / UAV123 / Anti-UAV): every single-object
tracker is initialised on the first ground-truth box (first frame with exist == 1) and then tracks every following
frame on its own, at full resolution, with no detector and no re-initialisation; one row per frame:
    frame,id,x1,y1,x2,y2,w,h,cx,cy,score,pred_exist,gt_exist,gt_x1,gt_y1,gt_x2,gt_y2,iou,center_err,norm_center_err,match
  box columns are empty before the init frame; score = the tracker's confidence (1 on the init frame);
  pred_exist = score >= the tracker's lost threshold in track.SOT (the tracker "reports" the target; the box itself is
  written either way); iou / center_err (px) / norm_center_err (center error / GT size, LaSOT) when both exist;
  match = pred_exist and IoU >= 0.5.
--protocol det: the track.py pipeline (detector-seeded SingleObject / MOT wrappers, as uav_tracker/export_csv.py),
  rows frame,id,x1,y1,x2,y2,w,h,cx,cy only for frames with a track.
GT CSV (both): frame,id,x1..cy for frames where exist == 1 (Anti-UAV json {"exist": [...], "gt_rect": [[x,y,w,h]]}).

Out:       <out>/<sequence>_<video stem>_gt.csv, <out>/<sequence>_<video stem>_<tracker>.csv,
           <out>/timing/<sequence>_<video stem>.csv  (tracker, frames, ms/frame; tracker time only)
A video whose CSVs all exist is skipped, so an interrupted run can be restarted. Then: python metrics.py --dir <out>

Run:       python export_csv.py --source .../val --trackers avtrack ortrack_deit
           python export_csv.py --source .../val --shard 0/4     # 4 processes, each takes every 4th video
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from metrics import MATCH_IOU
from track import ALL_TRACKERS, GROUPS, SOT, build_trackers, fit_frame

HEADER = ["frame", "id", "x1", "y1", "x2", "y2", "w", "h", "cx", "cy"]
HEADER_TRACK = HEADER + ["score", "pred_exist"]
HEADER_GT = HEADER + ["score", "pred_exist", "gt_exist", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "iou",
                      "center_err", "norm_center_err", "match"]


def row(frame, tid, x1, y1, x2, y2):
    return [frame, tid, *(round(float(v), 2) for v in (x1, y1, x2, y2, x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2))]


def write_csv(path, rows, header=HEADER):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tmp.rename(path)  # never leave a half-written CSV behind, so --resume can trust existing files


def load_gt(json_path):
    """-> list per frame (0-based) of GT (x1, y1, x2, y2) or None when the target is absent."""
    gt = json.loads(Path(json_path).read_text())
    return [(r[0], r[1], r[0] + r[2], r[1] + r[3]) if e and len(r) == 4 else None
            for e, r in zip(gt["exist"], gt["gt_rect"])]


def gt_rows(json_path):
    return [row(i, 1, *b) for i, b in enumerate(load_gt(json_path), 1) if b is not None]


def compare(frame, box, score, thr, g):
    """One --protocol gt row: prediction (x1, y1, x2, y2) or None, GT box or None, and the per-frame comparison."""
    pred = box is not None and score >= thr
    out = row(frame, 1, *box) if box is not None else [frame, 1] + [""] * 8
    out += [round(float(score), 4) if box is not None else "", int(pred), int(g is not None)]
    out += [round(float(v), 2) for v in g] if g is not None else [""] * 4
    if box is None or g is None:
        return out + ["", "", "", 0]
    ix = max(0.0, min(box[2], g[2]) - max(box[0], g[0])) * max(0.0, min(box[3], g[3]) - max(box[1], g[1]))
    union = (box[2] - box[0]) * (box[3] - box[1]) + (g[2] - g[0]) * (g[3] - g[1]) - ix
    iou = ix / union if union > 0 else 0.0
    dx, dy = (box[0] + box[2] - g[0] - g[2]) / 2, (box[1] + box[3] - g[1] - g[3]) / 2
    gw, gh = max(g[2] - g[0], 1e-6), max(g[3] - g[1], 1e-6)
    return out + [round(iou, 4), round(float(np.hypot(dx, dy)), 2), round(float(np.hypot(dx / gw, dy / gh)), 4),
                  int(pred and iou >= MATCH_IOU)]


def detector_ref(model, src, args, device):
    """No GT: the reference per frame = the detector's most confident box (conf >= --conf), in original pixels,
    and the init frame = the first one whose best box has conf >= --init-conf. -> (boxes or None per frame, start)."""
    cap = cv2.VideoCapture(str(src))
    ref, start, n = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = fit_frame(frame, 1280)  # detector input as in track.py
        s = frame.shape[1] / small.shape[1]
        d = model.predict(small, conf=args.conf, iou=args.iou, imgsz=args.imgsz, agnostic_nms=args.agnostic,
                          device=device, verbose=False)[0].boxes.data.cpu().numpy()
        if args.max_box and len(d):
            d = d[(d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]) <= args.max_box * small.shape[0] * small.shape[1]]
        if len(d):
            b = d[d[:, 4].argmax()]
            ref.append(tuple(float(v) * s for v in b[:4]))
            if start is None and b[4] >= args.init_conf:
                start = n
        else:
            ref.append(None)
        n += 1
    cap.release()
    return ref, start


def run_video_gt(src, gts, args, device, out, stem, start=None, ref_cols=True):
    """Standard OPE: init on the first GT box (or frame `start`), then every tracker runs alone on every frame."""
    models = {k: SOT[k][2](device) for k in args.trackers}
    rows = {k: [] for k in args.trackers}
    ms = {k: 0.0 for k in args.trackers}
    errors = {k: 0 for k in args.trackers}
    if start is None:
        start = next((i for i, g in enumerate(gts) if g is not None), None)
    cap = cv2.VideoCapture(str(src))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = gts[n] if n < len(gts) else None
        small = fit_frame(frame, args.max_side)
        s = frame.shape[1] / small.shape[1]
        for k, m in models.items():
            if start is None or n < start:
                box, score = None, 0.0
            elif n == start:
                x1, y1, x2, y2 = (v / s for v in g)
                m.init(small, (x1, y1, x2 - x1, y2 - y1))
                box, score = g, 1.0
            else:
                t0 = time.perf_counter()
                try:
                    _, (x, y, w, h), score = m.update(small)
                    box = (x * s, y * s, (x + w) * s, (y + h) * s)
                except Exception as e:  # e.g. a degenerate box; counts as "no box" for this frame
                    if not errors[k]:
                        print(f"[warn] {stem} {k} frame {n + 1}: {type(e).__name__}: {e}", flush=True)
                    errors[k] += 1
                    box, score = None, 0.0
                ms[k] += (time.perf_counter() - t0) * 1000
            if ref_cols:
                rows[k].append(compare(n + 1, box, score, SOT[k][3], g))
            elif box is not None:  # tracker output only (no GT): frame, box, score, reported
                rows[k].append(row(n + 1, 1, *box) + [round(float(score), 4), int(score >= SOT[k][3])])
        n += 1
    cap.release()
    tracked = max(n - (start or 0) - 1, 1)
    for k in args.trackers:
        write_csv(out / f"{stem}_{k}.csv", rows[k], header=HEADER_GT if ref_cols else HEADER_TRACK)
    write_csv(out / "timing" / f"{stem}.csv", [[k, tracked, round(ms[k] / tracked, 2), errors[k]] for k in args.trackers],
              header=["tracker", "frames", "ms_per_frame", "errors"])
    return n


def run_video(model, src, args, device, out, stem):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    trackers = build_trackers(args, fps, model.names, device)
    rows = {t.name: [] for t in trackers}
    ms = {t.name: 0.0 for t in trackers}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        small = fit_frame(frame, args.max_side)
        s = frame.shape[1] / small.shape[1]  # back to original-resolution pixels, like the GT
        res = model.predict(small, conf=args.det_low, iou=args.iou, imgsz=args.imgsz,
                            agnostic_nms=args.agnostic, device=device, verbose=False)[0]
        dets = res.boxes.data.cpu().numpy()
        if args.max_box:
            area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            dets = dets[area <= args.max_box * small.shape[0] * small.shape[1]]
        for t in trackers:
            t0 = time.perf_counter()
            tracks = t.update(small, dets)
            ms[t.name] += (time.perf_counter() - t0) * 1000
            rows[t.name] += [row(n, tid, *(v * s for v in box)) for tid, box, *_ in tracks]
    cap.release()
    for key, t in zip(args.trackers, trackers):
        write_csv(out / f"{stem}_{key}.csv", rows[t.name])
    write_csv(out / "timing" / f"{stem}.csv", [[k, n, round(ms[t.name] / max(n, 1), 2)]
                                               for k, t in zip(args.trackers, trackers)],
              header=["tracker", "frames", "ms_per_frame"])
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, or directory searched recursively for --pattern")
    ap.add_argument("--pattern", default="visible.mp4", help="video file name to look for in a directory")
    ap.add_argument("--trackers", nargs="+", default=["avtrack"], choices=ALL_TRACKERS + list(GROUPS))
    ap.add_argument("--model", default="YOLOv8n_HuggingFace_TomSmaildrone-yolo-v1.pt")
    ap.add_argument("--protocol", default="gt", choices=["gt", "det"],
                    help="gt: standard OPE, init on the first GT box (default); det: track.py pipeline")
    ap.add_argument("--out", default=None, help="default runs/csv_gt (gt) / runs/csv (det)")
    ap.add_argument("--shard", default="0/1", help="i/n: process only every n-th video starting at i")
    ap.add_argument("--ref", default="json", choices=["json", "det"],
                    help="json = <video>.json GT (default); det = no GT: init on the detector's first box with "
                         "conf >= --init-conf, then tracker output only (frame,id,x1..cy,score,pred_exist)")
    ap.add_argument("--init-conf", type=float, default=0.5, help="--ref det: detector conf to initialise on")
    # same defaults as track.py
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--det-low", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-side", type=int, default=None, help="default: 0 = full resolution (gt), 1280 (det)")
    ap.add_argument("--no-agnostic", dest="agnostic", action="store_false")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--max-box", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.trackers = list(dict.fromkeys(x for t in args.trackers for x in GROUPS.get(t, [t])))
    gt_mode = args.protocol == "gt"
    if gt_mode and any(t not in SOT for t in args.trackers):
        ap.error("--protocol gt needs single-object trackers (MOT trackers cannot be initialised on a GT box)")
    args.out = args.out or ("runs/csv_gt" if gt_mode else "runs/csv")
    args.max_side = (0 if gt_mode else 1280) if args.max_side is None else args.max_side

    src = Path(args.source)
    videos = sorted(src.rglob(args.pattern)) if src.is_dir() else [src]
    i, n = (int(v) for v in args.shard.split("/"))
    videos = videos[i::n]
    out = Path(args.out).resolve()  # some tracker loaders chdir into third_party/
    (out / "timing").mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    det_ref = gt_mode and args.ref == "det"
    model = YOLO(args.model) if not gt_mode or det_ref else None

    for k, video in enumerate(videos, 1):
        stem = f"{video.parent.name}_{video.stem}"
        wanted = [out / f"{stem}_{t}.csv" for t in args.trackers] + ([] if det_ref else [out / f"{stem}_gt.csv"])
        if all(p.exists() for p in wanted):
            print(f"[{k}/{len(videos)}] {stem}: done already, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        if det_ref:
            ref, start = detector_ref(model, video, args, device)  # only its init box is used
            frames = run_video_gt(video, ref, args, device, out, stem, start=start, ref_cols=False)
        elif gt_mode:
            write_csv(out / f"{stem}_gt.csv", gt_rows(video.with_suffix(".json")))
            frames = run_video_gt(video, load_gt(video.with_suffix(".json")), args, device, out, stem)
        else:
            write_csv(out / f"{stem}_gt.csv", gt_rows(video.with_suffix(".json")))
            frames = run_video(model, video, args, device, out, stem)
        print(f"[{k}/{len(videos)}] {stem}: {frames} frames, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
