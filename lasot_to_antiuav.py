#!/usr/bin/env python3
"""
Convert LaSOT sequences (img/%08d.jpg + groundtruth.txt + full_occlusion.txt + out_of_view.txt) to the Anti-UAV layout
that eval_tracker.py reads: <out>/<split>/<sequence>/visible.mp4 + visible.json {"exist": [...], "gt_rect": [[x,y,w,h]]}.
exist = 0 on frames LaSOT marks fully occluded or out of view (its box is kept in gt_rect but not scored).
The video is lossless H.264 (yuv444p, qp 0) at 30 fps, so trackers see the original JPEG pixels.
Sequences listed in <lasot>/testing_set.txt go to <out>/test, the rest to <out>/train (SUTrack, MCITrack and most
other deep trackers are trained on the LaSOT train split, so only test/ is a fair comparison).

Run:       python lasot_to_antiuav.py --lasot ~/Downloads/LaSOT --category drone --out ~/Downloads/LaSOT/drone_eval
           python eval_tracker.py --tracker sutrack --source ~/Downloads/LaSOT/drone_eval/test --out runs/lasot_drone_test
"""
import argparse
import json
import subprocess
from pathlib import Path


def read_flags(path):
    return [int(v) for v in path.read_text().replace("\n", ",").split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lasot", required=True, help="LaSOT root with <category>/<category>-N/ and testing_set.txt")
    ap.add_argument("--category", default="drone")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.lasot).expanduser()
    test = set((root / "testing_set.txt").read_text().split())
    seqs = sorted((root / args.category).glob(f"{args.category}-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    for seq in seqs:
        d = Path(args.out).expanduser() / ("test" if seq.name in test else "train") / seq.name
        if (d / "visible.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        rects = [[int(float(v)) for v in line.split(",")] for line in (seq / "groundtruth.txt").read_text().split()]
        hidden = [a or b for a, b in zip(read_flags(seq / "full_occlusion.txt"), read_flags(seq / "out_of_view.txt"))]
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(seq / "img/%08d.jpg"),
                        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(d / "visible.mp4")], check=True)
        exist = [int(not h and r[2] > 0 and r[3] > 0) for h, r in zip(hidden, rects)]
        (d / "visible.json").write_text(json.dumps({"exist": exist, "gt_rect": rects}))
        print(f"{d.parent.name}/{seq.name}: {len(rects)} frames, {sum(exist)} with the target", flush=True)


if __name__ == "__main__":
    main()
