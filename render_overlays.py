#!/usr/bin/env python3
"""
Overlay videos of ground truth vs tracker output from the CSVs of export_csv.py --protocol gt (no tracker is re-run).

For each selected sequence the video is decoded once and one video per tracker is written:
    <out>/<tracker>/<sequence>.mp4
GT box green (dashed when the target is absent: nothing drawn), prediction in orange when the tracker reports the
target (score >= its lost threshold) and grey when it doesn't; header with frame, IoU, center error, score, and the
running mean IoU. By default 10 sequences spread evenly over the sorted list, the same for every tracker.

Run:       python render_overlays.py --source .../val --csv runs/csv_gt
           python render_overlays.py --source .../val --sequences 20190925_101846_1_4 20190926_133516_1_6
           python render_overlays.py --source test_videos --pattern "*.mp4" --csv runs/csv_test --ref det
             (no GT: the reference box is the detector's, from export_csv.py --ref det, labelled "detector")
           python render_overlays.py --source test_videos --pattern "*.mp4" --csv runs/csv_test --ref none --combined
             (every tracker on one video, one colour each: <out>/all/<sequence>.mp4)
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

GT_COLOR, PRED_COLOR, OFF_COLOR = (0, 220, 0), (0, 140, 255), (160, 160, 160)


def read_rows(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[int(r["frame"])] = r
    return rows


def num(r, k):
    return float(r[k]) if r and r.get(k, "") != "" else None


def label(img, text, org, scale, color=(255, 255, 255)):
    th = max(1, round(2 * scale))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, th, cv2.LINE_AA)


def draw(frame, n, r, name, stats, scale, ref="GT"):
    img = frame
    s = max(img.shape[:2]) / 1920 * 1.6  # text size relative to the (resized) frame's longer side
    lw = max(1, round(2 * s))
    if num(r, "gt_x1") is not None:
        g = [round(num(r, k) * scale) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        cv2.rectangle(img, g[:2], g[2:], GT_COLOR, lw)
    if num(r, "x1") is not None:
        p = [round(num(r, k) * scale) for k in ("x1", "y1", "x2", "y2")]
        on = r["pred_exist"] == "1"
        cv2.rectangle(img, p[:2], p[2:], PRED_COLOR if on else OFF_COLOR, lw)
    iou, cerr, score = num(r, "iou"), num(r, "center_err"), num(r, "score")
    if iou is not None:
        stats["iou"].append(iou)
    miou = np.mean(stats["iou"]) if stats["iou"] else float("nan")
    if ref is None:  # tracker output only
        lines = [f"{name}  frame {n}", "no box" if score is None else
                 f"score {score:.2f} " + ("(reported)" if r["pred_exist"] == "1" else "(lost)")]
    else:
        lines = [f"{name}  frame {n}",
                 (f"{ref}: absent" if r is None or r["gt_exist"] != "1" else f"{ref}: present")
                 + ("" if score is None else f"  | score {score:.2f} " + ("(reported)" if r["pred_exist"] == "1" else "(lost)")),
                 ("" if iou is None else f"IoU {iou:.2f}  center err {cerr:.0f}px  ") + f"mean IoU {miou:.3f}"]
    band = img[:round((20 + 26 * len(lines)) * s), :round(560 * s)]  # dark band over the camera's own OSD text
    band[:] = (band * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        label(img, t, (round(10 * s), round((28 + 26 * i) * s)), s)
    x, y = round(10 * s), img.shape[0] - round(12 * s)
    legend = ((ref, GT_COLOR),) * (ref is not None) + (("pred", PRED_COLOR), ("pred (lost)", OFF_COLOR))
    for text, color in legend:  # legend, measured
        label(img, text, (x, y), s, color)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, max(1, round(2 * s)) + 2)[0][0] + round(16 * s)
    return img


# 12 distinct BGR colours for --combined (one per tracker)
PALETTE = [(255, 99, 71), (0, 165, 255), (0, 255, 255), (50, 205, 50), (255, 0, 255), (255, 191, 0),
           (147, 20, 255), (0, 128, 255), (203, 192, 255), (128, 128, 0), (0, 0, 255), (255, 255, 255)]


def draw_all(img, n, rows, colors, scale, ref="GT"):
    """--combined: every tracker's box on one frame, labelled by name; lost boxes dashed-thin; legend on the right."""
    s = max(img.shape[:2]) / 1920 * 1.6
    lw = max(1, round(2 * s))
    fs = 0.45 * s
    any_r = next((r for r in (rs.get(n) for rs in rows.values()) if r), None)
    if ref and num(any_r, "gt_x1") is not None:
        g = [round(num(any_r, k) * scale) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        cv2.rectangle(img, g[:2], g[2:], GT_COLOR, lw + 1)
    for t, rs in rows.items():
        r = rs.get(n)
        if num(r, "x1") is None:
            continue
        p = [round(num(r, k) * scale) for k in ("x1", "y1", "x2", "y2")]
        on = r["pred_exist"] == "1"
        cv2.rectangle(img, p[:2], p[2:], colors[t], lw if on else 1)
        if on:
            cv2.putText(img, t, (p[0], max(p[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, fs, colors[t], 1, cv2.LINE_AA)
    # legend: name + reported / lost / no box
    lh = round(22 * s)
    band = img[:round(10 * s) + lh * (len(rows) + 1), -round(260 * s):]
    band[:] = (band * 0.35).astype(np.uint8)
    x0 = img.shape[1] - round(250 * s)
    label(img, f"frame {n}", (x0, lh), s * 0.8)
    for i, (t, rs) in enumerate(rows.items(), 2):
        r = rs.get(n)
        state = "-" if num(r, "x1") is None else ("" if r["pred_exist"] == "1" else " (lost)")
        label(img, t + state, (x0, lh * i), s * 0.8, colors[t])
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/home/daryna/Downloads/val-20260930T095918Z-1-001/val")
    ap.add_argument("--csv", default="runs/csv_gt")
    ap.add_argument("--out", default="runs/overlays")
    ap.add_argument("--video-stem", default="visible")
    ap.add_argument("--pattern", default=None, help="plain videos instead of <seq>/<video-stem>.mp4, e.g. '*.mp4'")
    ap.add_argument("--ref", default="gt", choices=["gt", "none"], help="none: no GT, tracker boxes only")
    ap.add_argument("--sequences", nargs="*", help="default: --count sequences spread over the sorted list")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--trackers", nargs="*", help="default: every tracker that has CSVs")
    ap.add_argument("--width", type=int, default=960, help="output size of the longer side (aspect kept); 0 = original")
    ap.add_argument("--combined", action="store_true",
                    help="one video per sequence with every tracker on it: <out>/all/<sequence>.mp4")
    args = ap.parse_args()

    src, csv_dir, out = Path(args.source), Path(args.csv), Path(args.out)
    ref_label = None if args.ref == "none" else "GT"
    if args.pattern:  # plain videos: name = file stem, CSV stem = <folder>_<file stem> as export_csv.py writes it
        videos = {v.stem: (v, f"{v.parent.name}_{v.stem}") for v in sorted(src.glob(args.pattern))}
    else:
        videos = {v.parent.name: (v, f"{v.parent.name}_{v.stem}") for v in sorted(src.glob(f"*/{args.video_stem}.mp4"))}
    seqs = list(videos)
    if not args.sequences:
        idx = np.linspace(0, len(seqs) - 1, min(args.count, len(seqs))).round().astype(int)
        args.sequences = [seqs[i] for i in dict.fromkeys(idx)]
    if args.trackers:
        trackers = args.trackers
    else:
        found = set()
        for _, stem in videos.values():
            found |= {p.stem[len(stem) + 1:] for p in csv_dir.glob(f"{stem}_*.csv")}
        trackers = sorted(found - {"gt"})

    for k, seq in enumerate(args.sequences, 1):
        video, stem = videos[seq]
        have = [t for t in trackers if (csv_dir / f"{stem}_{t}.csv").exists()]
        if not have:
            print(f"[{k}/{len(args.sequences)}] {seq}: no CSVs yet, skipped")
            continue
        rows = {t: read_rows(csv_dir / f"{stem}_{t}.csv") for t in have}
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = args.width / max(W, H) if args.width else 1.0
        size = (round(W * scale), round(H * scale))
        if args.combined:
            colors = {t: PALETTE[trackers.index(t) % len(PALETTE)] for t in have}
            (out / "all").mkdir(parents=True, exist_ok=True)
            w = cv2.VideoWriter(str(out / "all" / f"{seq}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale != 1 else frame
                w.write(draw_all(small, n, rows, colors, scale, ref_label))
            cap.release()
            w.release()
            print(f"[{k}/{len(args.sequences)}] {seq}: {n} frames, {len(have)} trackers -> {out}/all/{seq}.mp4",
                  flush=True)
            continue
        writers, stats = {}, {t: {"iou": []} for t in have}
        for t in have:
            (out / t).mkdir(parents=True, exist_ok=True)
            writers[t] = cv2.VideoWriter(str(out / t / f"{seq}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale != 1 else frame
            for t in have:
                writers[t].write(draw(small.copy(), n, rows[t].get(n), t, stats[t], scale, ref_label))
        cap.release()
        for w in writers.values():
            w.release()
        print(f"[{k}/{len(args.sequences)}] {seq}: {n} frames x {len(have)} trackers -> {out}/<tracker>/{seq}.mp4",
              flush=True)


if __name__ == "__main__":
    main()
