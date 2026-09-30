#!/usr/bin/env python3
"""
Real-time UAV detection on videos with a YOLOv8 model.

Run:      python infer.py                                  # all videos in ./test_videos
          python infer.py --source test_videos/clip.mp4 --conf 0.3
          python infer.py --save --no-show                 # write annotated videos to ./runs/infer
          python infer.py --source 0                       # webcam

Keys:     q / Esc = quit, n = next video, space = pause/resume
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
COLORS = {0: (0, 200, 255), 1: (0, 0, 255)}  # BGR per class id

ROOT = Path(__file__).resolve().parent
MODELS = {
    1: "1)_YOLOv8n_Hugging_Face_TomSmaildrone-yolo-v1.pt",
    2: "2)_YOLO12m_shahed136_last_2025_EDTH-Warsaw-shahed136-detector.pt",
    3: "3)_YOLOv8_YOLO8_drone_detector.pt.pt",  # TODO: зараз копія №1, замінити на справжні ваги
    4: "4)_YOLO11n_UAV-finetune.pt",
    5: "5)_YOLOv8n_51ep-16-GPU.pt",
    6: "6)_YOLO11L_best.pt",
    7: "7)_YOLOv5_anti_uav_jittor.pt",
}


def resolve_model(ai, model):
    """Повертає шлях до вагів: цифра 1-7 -> ai/<файл>, інакше явний --model."""
    if ai is not None:
        return str(ROOT / "ai" / MODELS[ai])
    if model:
        p = Path(model)
        if not p.is_absolute() and not p.exists():
            cand = ROOT / p
            if cand.exists():
                return str(cand)
        return model
    try:
        raw = input(f"Введіть номер моделі 1-7 {sorted(MODELS)} [Enter=1]: ").strip()
    except EOFError:
        raw = ""
    num = int(raw) if raw.isdigit() and int(raw) in MODELS else 1
    return str(ROOT / "ai" / MODELS[num])


def collect_sources(source):
    if source.isdigit():
        return [int(source)]
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    return [p]


def draw(frame, result, names, fps):
    boxes = result.boxes
    for (x1, y1, x2, y2), conf, cls in zip(
        boxes.xyxy.int().tolist(), boxes.conf.tolist(), boxes.cls.int().tolist()
    ):
        color = COLORS.get(cls, (0, 255, 0))
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    hud = f"FPS {fps:5.1f} | detections: {len(boxes)}"
    cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def run_video(model, src, args, device):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[skip] cannot open {src}")
        return True
    name = src.name if isinstance(src, Path) else f"cam{src}"
    print(f"[run] {name}")

    writer = None
    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out_path = out_dir / f"{Path(name).stem}_det.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    win = "UAV detection"
    fps, prev = 0.0, time.perf_counter()
    keep_going, paused = True, False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(
                frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                device=device, verbose=False,
            )[0]
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
            prev = now
            frame = draw(frame, result, model.names, fps)
            if writer:
                writer.write(frame)

        if args.show:
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                keep_going = False
                break
            if key == ord("n"):
                break
            if key == ord(" "):
                paused = not paused
                prev = time.perf_counter()

    cap.release()
    if writer:
        writer.release()
        print(f"[saved] {out_path}")
    return keep_going


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ai", type=int, default=None, choices=sorted(MODELS),
                    help="номер моделі 1-7 з папки ai/ (якщо не задано і немає --model — запитає цифру)")
    ap.add_argument("--source", default="test_videos", help="video file, directory, or webcam index")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (auto if omitted)")
    ap.add_argument("--save", action="store_true", help="save annotated videos")
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--no-show", dest="show", action="store_false", help="disable live window")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model(args.ai, args.model)
    print(f"[model] {model_path}")
    model = YOLO(model_path)
    print(f"model classes: {model.names} | device: {device}")

    sources = collect_sources(args.source)
    if not sources:
        print(f"no videos found in {args.source}")
        return
    for src in sources:
        if not run_video(model, src, args, device):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
