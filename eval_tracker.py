#!/usr/bin/env python3
"""
Run single-object trackers on every video of a dataset (standard one-pass evaluation: init on the first GT box, then
the tracker runs alone on every frame, as export_csv.py --protocol gt) and score them per video:
    auc        area under the success plot: mean over IoU thresholds 0, 0.05, ..., 1 of P(IoU > t)   (OTB / LaSOT)
    p20        precision: P(center error <= 20 px)
    p_norm     normalised precision: AUC of P(center error / GT size <= t), t in 0..0.5          (LaSOT / TrackingNet)
    accuracy   per-frame state accuracy: (TP + TN) / frames; TP = reported box with IoU >= 0.5 on a target frame,
               TN = nothing reported on a frame without the target
    f_score    F1 of precision / recall of the reported boxes vs GT at IoU >= 0.5
    fps        tracker time only (no video decode); for the detector-driven trackers below detector + tracker
auc / p20 / p_norm use the box the tracker outputs on every frame where the target exists; accuracy / f_score only
count boxes the tracker reports (score >= its lost threshold in track.SOT). The AVERAGE row is the mean over videos.

Detector-driven trackers (YOLO --model on every frame, frame fitted to --det-side as in track.py):
    kalman     track.KalmanModel: initialised on the first GT box, then corrected with the detection nearest to its
               prediction each frame
    deepsort   multi-object (MobileNet appearance + Kalman), bytetrack (two-stage IoU matching incl. low-score boxes),
    bytetrack  sort (Kalman + Hungarian IoU): they cannot be initialised on a box, so the target is the first
    sort       confirmed track with IoU >= 0.5 to the GT from the init frame on; only that id is followed afterwards
               (no re-association, so an id switch loses the target). Coasting tracks (Kalman prediction while the
               detection is missing) count as reported.

Out:       <out>/<tracker>/<sequence>_<video stem>_<tracker>.csv   per-frame rows (export_csv.py format)
           <out>/<tracker>/timing/<sequence>_<video stem>.csv
           <out>/<tracker>/metrics.csv   one row per video + AVERAGE row
           <out>/summary.csv             AVERAGE row of every tracker in <out>
Finished videos are skipped, so an interrupted run can be restarted.

Run:       python eval_tracker.py --tracker sutrack
           python eval_tracker.py --tracker sutrack asymtrack avtrack --source .../val --out runs/eval
           python eval_tracker.py --tracker sutrack asymtrack avtrack --count 10     # 10 videos spread over the set
           python eval_tracker.py --tracker sutrack --metrics-only     # re-score existing CSVs, no tracking
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import cv2

from export_csv import HEADER_GT, compare, load_gt, run_video_gt, write_csv
from metrics import sequence_metrics
from track import SOT, SORT, ByteTrack, DeepSORT, fit_frame

DET_TRACKERS = ["kalman", "deepsort", "bytetrack", "sort"]

COLS = ["auc", "p20", "p_norm", "accuracy", "f_score", "fps"]


def video_metrics(rows, ms_per_frame):
    m = sequence_metrics(rows)
    gt = np.array([r["gt_exist"] == "1" for r in rows])
    report = np.array([r["pred_exist"] == "1" and r["x1"] != "" for r in rows])
    tn = int((~gt & ~report).sum())
    return {"frames": m["frames"], "gt_frames": m["gt_frames"],
            "auc": m["success_auc"], "p20": m["precision_20"], "p_norm": m["norm_prec_auc"],
            "accuracy": (m["tp"] + tn) / m["frames"] if m["frames"] else np.nan,
            "f_score": m["f1"], "fps": 1000 / ms_per_frame if ms_per_frame else np.nan}


def detect(model, frame, args, device):
    """-> (N, 6) x1, y1, x2, y2, conf, cls, as track.py (low conf kept, implausibly large boxes dropped)."""
    d = model.predict(frame, conf=args.det_low, iou=args.iou, imgsz=args.imgsz, agnostic_nms=True,
                      device=device, verbose=False)[0].boxes.data.cpu().numpy()
    area = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    return d[area <= args.max_box * frame.shape[0] * frame.shape[1]]


def run_video_det(video, gts, tracker, model, args, device, out, stem):
    """kalman / deepsort / bytetrack / sort on detector output; rows as export_csv.run_video_gt. -> frames."""
    start = next((i for i, g in enumerate(gts) if g is not None), None)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    m = {"kalman": lambda: SOT["kalman"][2](device),
         "deepsort": lambda: DeepSORT(fps, model.names, args.conf, args.patience),
         "bytetrack": lambda: ByteTrack(fps, model.names, args.patience),
         "sort": lambda: SORT(model.names, args.conf, args.patience)}[tracker]()
    thr = SOT["kalman"][3] if tracker == "kalman" else 0.0
    rows, det_ms, trk_ms, target, n = [], 0.0, 0.0, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = gts[n] if n < len(gts) else None
        small = fit_frame(frame, args.det_side)
        s = frame.shape[1] / small.shape[1]
        box, score = None, 0.0
        if start is not None and n >= start:
            t0 = time.perf_counter()
            dets = detect(model, small, args, device)
            t1 = time.perf_counter()
            if tracker == "kalman":
                if n == start:
                    x1, y1, x2, y2 = (v / s for v in g)
                    m.init(small, (x1, y1, x2 - x1, y2 - y1))
                    box, score = g, 1.0
                else:
                    _, (x, y, w, h), score = m.update(small, dets)
                    box = (x * s, y * s, (x + w) * s, (y + h) * s)
            else:
                tracks = {tid: tuple(v * s for v in b) for tid, b, *_ in m.update(small, dets)}
                if target is None and g is not None and tracks:
                    ious = {tid: compare(0, b, 1.0, 0, g)[17] for tid, b in tracks.items()}
                    best = max(ious, key=ious.get)
                    target = best if ious[best] >= 0.5 else None
                if target in tracks:
                    box, score = tracks[target], 1.0
            t2 = time.perf_counter()
            if n > start:
                det_ms, trk_ms = det_ms + (t1 - t0) * 1000, trk_ms + (t2 - t1) * 1000
        rows.append(compare(n + 1, box, score, thr, g))
        n += 1
    cap.release()
    tracked = max(n - (start or 0) - 1, 1)
    write_csv(out / f"{stem}_{tracker}.csv", rows, header=HEADER_GT)
    write_csv(out / "timing" / f"{stem}.csv",
              [[tracker, tracked, round((det_ms + trk_ms) / tracked, 2), round(det_ms / tracked, 2),
                round(trk_ms / tracked, 2)]], header=["tracker", "frames", "ms_per_frame", "det_ms", "trk_ms"])
    return n


def write_rows(path, head, rows):
    fmt = lambda v: round(v, 4) if isinstance(v, float) else v
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in rows:
            w.writerow([fmt(r[k]) for k in head])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", nargs="+", required=True, choices=list(SOT) + DET_TRACKERS)
    ap.add_argument("--source", default="/home/daryna/Downloads/val-20260930T095918Z-1-001/val",
                    help="dataset dir with <sequence>/<video stem>.mp4 + .json")
    ap.add_argument("--video-stem", default="visible")
    ap.add_argument("--count", type=int, default=0, help="only N videos spread evenly over the sorted list; 0 = all")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--max-side", type=int, default=0, help="tracker input size of the longer side; 0 = full res")
    ap.add_argument("--metrics-only", action="store_true", help="score existing CSVs, do not run trackers")
    ap.add_argument("--device", default=None)
    # detector, for kalman / deepsort (same defaults as track.py)
    ap.add_argument("--model", default="YOLOv8n_HuggingFace_TomSmaildrone-yolo-v1.pt")
    ap.add_argument("--det-side", type=int, default=1280, help="frame size (longer side) for detector + tracker")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--det-low", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-box", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out).resolve()  # some tracker loaders chdir into third_party/
    videos = sorted(Path(args.source).glob(f"*/{args.video_stem}.mp4"))
    if not videos:
        ap.error(f"no */{args.video_stem}.mp4 under {args.source}")
    if 0 < args.count < len(videos):
        videos = [videos[i] for i in np.linspace(0, len(videos) - 1, args.count).round().astype(int)]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = None

    for tracker in args.tracker:
        d = out / tracker
        (d / "timing").mkdir(parents=True, exist_ok=True)
        per_video = []
        for k, video in enumerate(videos, 1):
            stem = f"{video.parent.name}_{video.stem}"
            path, tpath = d / f"{stem}_{tracker}.csv", d / "timing" / f"{stem}.csv"
            if not (path.exists() and tpath.exists()):
                if args.metrics_only:
                    continue
                t0 = time.perf_counter()
                gts = load_gt(video.with_suffix(".json"))
                if tracker in DET_TRACKERS:
                    if detector is None:
                        from ultralytics import YOLO
                        detector = YOLO(str(Path(args.model).resolve()))
                    n = run_video_det(video, gts, tracker, detector, args, device, d, stem)
                else:
                    run_args = argparse.Namespace(trackers=[tracker], max_side=args.max_side)
                    n = run_video_gt(video, gts, run_args, device, d, stem)
                print(f"[{tracker} {k}/{len(videos)}] {stem}: {n} frames, {time.perf_counter() - t0:.0f} s", flush=True)
            rows = list(csv.DictReader(open(path)))
            ms = next((float(r["ms_per_frame"]) for r in csv.DictReader(open(tpath)) if r["tracker"] == tracker), None)
            per_video.append({"video": stem, **video_metrics(rows, ms)})
        if not per_video:
            print(f"[{tracker}] no results")
            continue
        avg = {"video": f"AVERAGE ({len(per_video)} videos)",
               **{k: sum(m[k] for m in per_video) for k in ("frames", "gt_frames")},
               **{k: float(np.nanmean([m[k] for m in per_video])) for k in COLS}}
        write_rows(d / "metrics.csv", ["video", *COLS, "frames", "gt_frames"], per_video + [avg])
        print(f"[saved] {d / 'metrics.csv'}")

    # summary of every tracker evaluated into <out> so far
    summary = []
    for m in sorted(out.glob("*/metrics.csv")):
        last = list(csv.DictReader(open(m)))[-1]
        summary.append({"tracker": m.parent.name, "videos": last["video"].split("(")[1].split()[0],
                        **{k: float(last[k]) for k in COLS}})
    summary.sort(key=lambda s: -s["auc"])
    write_rows(out / "summary.csv", ["tracker", "videos", *COLS], summary)
    print(f"\n{'tracker':<14}{'videos':>7}" + "".join(f"{c:>10}" for c in COLS))
    for s in summary:
        print(f"{s['tracker']:<14}{s['videos']:>7}" + "".join(f"{s[c]:>10.3f}" for c in COLS))
    print(f"[saved] {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
