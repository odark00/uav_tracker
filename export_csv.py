#!/usr/bin/env python3
"""
Write ground truth and tracker output of one video as two CSVs in the same format (original-resolution pixels):
    frame,id,x1,y1,x2,y2,w,h,cx,cy
frame is 1-based; x1,y1,x2,y2 = box corners, w,h = size, cx,cy = box center. GT comes from <video>.json next to the
video (Anti-UAV format: {"exist": [...], "gt_rect": [[x, y, w, h], ...]}), one target with id 1, only frames where
exist == 1. The tracker runs exactly as in track.py (same detector, same SingleObject / MOT wrappers).

Run:       python export_csv.py --source .../visible.mp4 --tracker avtrack
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from track import ALL_TRACKERS, build_trackers, fit_frame

HEADER = ["frame", "id", "x1", "y1", "x2", "y2", "w", "h", "cx", "cy"]


def row(frame, tid, x1, y1, x2, y2):
    return [frame, tid, *(round(float(v), 2) for v in (x1, y1, x2, y2, x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2))]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"[saved] {path} ({len(rows)} rows)")


def gt_rows(json_path):
    gt = json.loads(Path(json_path).read_text())
    return [row(i, 1, x, y, x + w, y + h) for i, (e, r) in enumerate(zip(gt["exist"], gt["gt_rect"]), 1)
            if e and len(r) == 4 for x, y, w, h in [r]]


def tracker_rows(model, src, args, device):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tracker = build_trackers(args, fps, model.names, device)[0]
    rows, n = [], 0
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
        for tid, box, *_ in tracker.update(small, dets):
            rows.append(row(n, tid, *(v * s for v in box)))
    cap.release()
    print(f"[run] {tracker.name}: {n} frames")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file; GT is read from the .json with the same stem")
    ap.add_argument("--gt", default=None, help="GT json (default: <source>.json)")
    ap.add_argument("--tracker", default="avtrack", choices=ALL_TRACKERS)
    ap.add_argument("--model", default="YOLOv8n_HuggingFace_TomSmaildrone-yolo-v1.pt")
    ap.add_argument("--out", default="runs/csv")
    # same defaults as track.py
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--det-low", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--no-agnostic", dest="agnostic", action="store_false")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--max-box", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.trackers = [args.tracker]

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{src.parent.name}_{src.stem}"
    write_csv(out / f"{stem}_gt.csv", gt_rows(args.gt or src.with_suffix(".json")))

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = tracker_rows(YOLO(args.model), src, args, device)
    write_csv(out / f"{stem}_{args.tracker}.csv", rows)


if __name__ == "__main__":
    main()
