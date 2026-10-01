#!/usr/bin/env python3
"""
Compare UAV trackers on the same YOLOv8 detections, drawn side by side in real time.

Multi-object ("mot", fed by the detector every frame):
           bytetrack  - IoU + Kalman, two-stage matching incl. low-score boxes (ultralytics BYTETracker)
           botsort    - ByteTrack + camera motion compensation, sparse optical flow (ultralytics BOTSORT)
           deepsort   - Kalman + MobileNet appearance embeddings (deep-sort-realtime)
           sort       - Kalman + Hungarian IoU matching, no appearance (own numpy implementation)
Single-object, classic ("classic"):
           csrt       - discriminative correlation filter + channel/spatial reliability (OpenCV contrib)
           kcf        - kernelized correlation filter (OpenCV contrib)
           mosse      - minimum output sum of squared error filter, fastest (OpenCV contrib legacy)
           mil        - online multiple-instance learning (OpenCV)
           tld        - tracking-learning-detection (OpenCV contrib legacy)
           meanshift  - hue/saturation histogram back-projection + mean shift (OpenCV)
           camshift   - same, with adaptive window size (OpenCV)
           ivt        - incremental PCA subspace + particle filter (own implementation, Ross et al. 2008)
Single-object, Bayesian filters on the detections ("filter"):
           kalman     - constant-velocity Kalman filter, gated nearest detection
           particle   - particle filter, colour histogram x detection likelihood
Single-object, deep learning ("deep"):
           ortrack_deit - ORTrack-DeiT, occlusion-robust ViT for UAVs, CVPR 2025 (third_party/ORTrack, GPU)
           asymtrack  - AsymTrack-B, asymmetric Siamese EfficientMod, AAAI 2025 (third_party/AsymTrack, PyTorch on GPU)
           sutrack    - SUTrack-T224, unified single-ViT tracker (Fast-iTPN tiny), AAAI 2025 (third_party/SUTrack, GPU)
           sutrack_b  - SUTrack-B224, same with the Fast-iTPN base encoder and 2 templates (online update)
           focustrack - FocusTrack, OSTrack ViT-B with adaptive search region for anti-UAV (third_party/FocusTrack, GPU)
           mcitrack_b - MCITrack-B224, Fast-iTPN base + Mamba context neck, AAAI 2025 (third_party/MCITrack, GPU)
           avtrack    - AVTrack-DeiT, adaptive ViT (third_party/siam_zoo float ONNX, CPU, onnxruntime)
The detector runs once per frame and every tracker gets the same detections (boxes larger than --max-box of the
frame are dropped). Single-object trackers start on the most confident detection; a Kalman motion model on top
carries them through occlusions (cables, branches) and re-acquires the same target from nearby detections,
see SingleObject.

Run:       python track.py                                        # all trackers, all videos in ./test_videos
           python track.py --trackers bytetrack sutrack_b --source test_videos/clip.mp4
           python track.py --trackers deep                         # one group: mot / classic / filter / deep
           python track.py --save --no-show                       # write grid videos to ./runs/track

Keys:      q / Esc = quit, n = next video, space = pause/resume
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from infer import collect_sources

ROOT = Path(__file__).resolve().parent
ORTRACK_DIR = ROOT / "third_party/ORTrack"
ASYMTRACK_DIR = ROOT / "third_party/AsymTrack"
SUTRACK_DIR = ROOT / "third_party/SUTrack"
FOCUSTRACK_DIR = ROOT / "third_party/FocusTrack"
MCITRACK_DIR = ROOT / "third_party/MCITrack"
SIAM_ZOO_DIR = ROOT / "third_party/siam_zoo"


def id_color(tid):
    rng = np.random.default_rng(int(tid) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


# ------------------------------------------------------------------------------------------------ multi-object
class ByteTrack:
    name = "ByteTrack"

    def __init__(self, fps, names, coast):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        self.tracker = BYTETracker(cfg)  # keeps lost tracks for track_buffer=30 frames
        self.names, self.coast = names, coast

    def update(self, frame, dets):
        from ultralytics.engine.results import Boxes

        out = self.tracker.update(Boxes(dets, frame.shape[:2]), frame)
        # rows: x1, y1, x2, y2, id, score, cls, det_idx
        tracks = [(int(r[4]), r[:4], float(r[5]), self.names[int(r[6])]) for r in out]
        # lost tracks (no detection this frame, e.g. UAV on cables): draw their Kalman prediction, same id
        fid = self.tracker.frame_id
        tracks += [(t.track_id, t.xyxy, 0.0, self.names[int(t.cls)] + " (coast)") for t in self.tracker.lost_stracks
                   if fid - t.end_frame <= self.coast]
        return tracks


class BoTSORT(ByteTrack):
    """ByteTrack + global motion compensation (sparse optical flow warps the Kalman states when the camera pans).
    ReID stays off (botsort.yaml default): its "auto" model needs the detector's features, which aren't passed."""
    name = "BoT-SORT"

    def __init__(self, fps, names, coast):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
        self.tracker = BOTSORT(cfg)
        self.names, self.coast = names, coast


class DeepSORT:
    name = "DeepSORT"

    def __init__(self, fps, names, conf, coast):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(max_age=int(fps), n_init=3, embedder="mobilenet", half=True,
                                bgr=True, embedder_gpu=torch.cuda.is_available())
        self.names, self.conf, self.coast = names, conf, coast

    def update(self, frame, dets):
        raw = [([x1, y1, x2 - x1, y2 - y1], c, int(k)) for x1, y1, x2, y2, c, k in dets if c >= self.conf]
        tracks = self.tracker.update_tracks(raw, frame=frame)
        # unmatched confirmed tracks = Kalman prediction ("coast"), so the id stays visible through occlusions
        return [(int(t.track_id), t.to_ltrb(), t.det_conf or 0.0,
                 self.names[t.det_class] + (" (coast)" if t.time_since_update > 0 else ""))
                for t in tracks if t.is_confirmed() and t.time_since_update <= self.coast]


class KalmanBox:
    """SORT's constant-velocity Kalman filter. State [cx, cy, area, aspect, vx, vy, v_area], boxes in/out as xyxy."""

    F = np.eye(7) + np.eye(7, k=4)
    H = np.eye(4, 7)
    R = np.diag([1.0, 1.0, 10.0, 10.0])
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

    def __init__(self, box):
        self.x = np.r_[self._z(box), 0.0, 0.0, 0.0]
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    @staticmethod
    def _z(box):
        x1, y1, x2, y2 = box[:4]
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        return np.array([x1 + w / 2, y1 + h / 2, w * h, w / h])

    def box(self):
        cx, cy, s, r = self.x[:4]
        w = np.sqrt(max(s * r, 1.0))
        h = max(s, 1.0) / w
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        y = self._z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


def iou_matrix(a, b):
    """Pairwise IoU of xyxy boxes a (N, 4) and b (M, 4)."""
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class SORT:
    """Bewley et al. 2016: Kalman prediction + Hungarian matching on IoU, no appearance.
    iou_thr 0.2 (paper: 0.3) to match ByteTrack's match_thresh; max_age 30 like ByteTrack's track_buffer."""

    name = "SORT"

    def __init__(self, names, conf, coast, max_age=30, min_hits=3, iou_thr=0.2):
        self.names, self.conf, self.coast = names, conf, coast
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_thr
        self.tracks, self.next_id, self.frame = [], 1, 0

    def update(self, frame, dets):
        from scipy.optimize import linear_sum_assignment

        self.frame += 1
        dets = dets[dets[:, 4] >= self.conf]
        preds = np.array([t["kf"].predict() for t in self.tracks]).reshape(-1, 4)
        matched_t, matched_d = set(), set()
        if len(preds) and len(dets):
            iou = iou_matrix(preds, dets[:, :4])
            for ti, di in zip(*linear_sum_assignment(-iou)):
                if iou[ti, di] >= self.iou_thr:
                    matched_t.add(ti)
                    matched_d.add(di)
                    t = self.tracks[ti]
                    t["kf"].update(dets[di])
                    t.update(since=0, streak=t["streak"] + 1, conf=float(dets[di, 4]), cls=int(dets[di, 5]))
                    t["confirmed"] = t["confirmed"] or t["streak"] >= self.min_hits
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.update(since=t["since"] + 1, streak=0)
        for di in set(range(len(dets))) - matched_d:
            self.tracks.append({"kf": KalmanBox(dets[di]), "id": self.next_id, "since": 0, "streak": 1, "confirmed": False,
                                "conf": float(dets[di, 4]), "cls": int(dets[di, 5])})
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t["since"] <= self.max_age]
        out = [(t["id"], t["kf"].box(), t["conf"], self.names[t["cls"]]) for t in self.tracks
               if t["since"] == 0 and (t["streak"] >= self.min_hits or self.frame <= self.min_hits)]
        # confirmed tracks that missed this frame: Kalman prediction (not in the SORT paper; keeps the id visible)
        return out + [(t["id"], t["kf"].box(), 0.0, self.names[t["cls"]] + " (coast)") for t in self.tracks
                      if t["confirmed"] and 0 < t["since"] <= self.coast]


# ------------------------------------------------------------------------------------------------ single-object
def sample_target(img, box, factor, out_sz):
    """STARK/LightFC crop: square of side sqrt(w*h)*factor around the box, zero-padded, resized to out_sz."""
    x, y, w, h = box
    sz = int(np.ceil(np.sqrt(w * h) * factor))
    x1, y1 = round(x + 0.5 * w - sz * 0.5), round(y + 0.5 * h - sz * 0.5)
    x2, y2 = x1 + sz, y1 + sz
    H, W = img.shape[:2]
    px1, py1 = max(0, -x1), max(0, -y1)
    px2, py2 = max(x2 - W + 1, 0), max(y2 - H + 1, 0)
    crop = img[y1 + py1:y2 - py2, x1 + px1:x2 - px2]
    crop = cv2.copyMakeBorder(crop, py1, py2, px1, px2, cv2.BORDER_CONSTANT)
    return cv2.resize(crop, (out_sz, out_sz)), out_sz / sz


@contextmanager
def repo_lib(repo, pkg="lib"):
    """LightFC, ORTrack, AsymTrack and pyCFTrackers are all a top-level package called `lib` (TCTrack: `pysot`):
    import from `repo` with its own `pkg`, then put back whatever `pkg` was loaded before, so all of them can run in
    the same process. Objects built inside keep working afterwards, they hold references to their own modules."""
    own = lambda k: k == pkg or k.startswith(pkg + ".")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if own(k)}
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        sys.path.remove(str(repo))
        for k in [k for k in sys.modules if own(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def load_module(name, path):
    """Import a single file under a unique module name (for third-party files called config.py, tracker.py, ...)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def box_from_center_head(head, out, window, search_size, rf, state, W, H, margin):
    """STARK/OSTrack center head -> (x, y, w, h) in the frame, as in LightFC/ORTrack track(); plus peak score."""
    score = float(out["score_map"].max())
    box = head.cal_bbox(window * out["score_map"], out["size_map"], out["offset_map"]).view(-1, 4).mean(0)
    return crop_box_to_frame(box, search_size, rf, state, W, H, margin), score


def crop_box_to_frame(box, search_size, rf, state, W, H, margin):
    """(cx, cy, w, h) normalised to the search crop -> clipped (x, y, w, h) in the frame (map_box_back + clip_box)."""
    cx, cy, w, h = (box * search_size / rf).tolist()
    half = 0.5 * search_size / rf
    cx += state[0] + 0.5 * state[2] - half
    cy += state[1] + 0.5 * state[3] - half
    x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
    x1, y1 = min(max(0, x1), W - margin), min(max(0, y1), H - margin)  # clip_box
    x2, y2 = min(max(margin, x2), W), min(max(margin, y2), H)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


class ORTrackModel:
    """Port of third_party/ORTrack/lib/test/tracker/ortrack.py (CVPR 2025, occlusion-robust ViT for UAVs) that
    also returns the peak score. Unlike LightFC the template goes through the ViT together with every search crop,
    so the template patch (not a feature) is cached."""

    reid = True  # search box in self.state: can be re-run around a re-id candidate (SingleObject)

    def __init__(self, device, cfg_name="deit_tiny_distilled_patch16_224"):
        with repo_lib(ORTRACK_DIR):
            from lib.config.ortrack.config import cfg, update_config_from_file
            from lib.models.ortrack import build_ortrack

            update_config_from_file(str(ORTRACK_DIR / f"experiments/ortrack/{cfg_name}.yaml"))
            net = build_ortrack(cfg, training=False)
            ckpt = torch.load(ORTRACK_DIR / f"output/checkpoints/train/ortrack/Model/{cfg_name}/ORTrack_ep0300.pth.tar",
                              map_location="cpu", weights_only=False)  # pickles lib.train.admin.local settings
        net.load_state_dict(ckpt["net"], strict=True)
        self.net = net.to(device).eval()
        self.device = device
        self.t, self.distill = cfg.TEST, cfg.MODEL.IS_DISTILL
        fs = self.t.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        hann = torch.hann_window(fs + 2, periodic=False)[1:-1]  # == hann2d(centered=True)
        self.window = (hann[:, None] * hann[None, :]).to(device)[None, None]
        self.mean = torch.tensor(cfg.DATA.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.STD, device=device).view(1, 3, 1, 1)

    def _tensor(self, patch):
        t = torch.from_numpy(patch).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (t - self.mean) / self.std

    def init(self, frame, box):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        z, _ = sample_target(rgb, box, self.t.TEMPLATE_FACTOR, self.t.TEMPLATE_SIZE)
        self.z = self._tensor(z)
        self.state = list(box)

    @torch.no_grad()
    def update(self, frame):
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, rf = sample_target(rgb, self.state, self.t.SEARCH_FACTOR, self.t.SEARCH_SIZE)
        out = self.net(template=self.z, search=self._tensor(x), is_distill=self.distill)
        self.state, score = box_from_center_head(self.net.box_head, out, self.window, self.t.SEARCH_SIZE, rf,
                                                 self.state, W, H, margin=10)  # clip_box margin, as in ORTrack
        return True, self.state, score


class AsymTrackModel:
    """Port of third_party/AsymTrack/lib/test/tracker/AsymTrack.py (AAAI 2025, asymmetric Siamese EfficientMod).
    Its STARK corner head has no confidence output, so the score is the geometric mean of the two corner heatmaps'
    softmax peaks: a sharp single corner -> high, a flat or split heatmap (target gone / occluded) -> low."""

    reid = True  # search box in self.state: can be re-run around a re-id candidate (SingleObject)

    def __init__(self, device, cfg_name="base"):
        with repo_lib(ASYMTRACK_DIR):
            from lib.config.AsymTrack.config import cfg, update_config_from_file
            from lib.models.AsymTrack import build_asymtrack
            from lib.utils.box_ops import box_xyxy_to_cxcywh

            update_config_from_file(str(ASYMTRACK_DIR / f"experiments/AsymTrack/{cfg_name}.yaml"))
            cfg.TEST_MODE = True  # skip the ImageNet backbone download, the checkpoint has everything
            net = build_asymtrack(cfg)  # GPU only: the corner head calls .cuda() in __init__
        ckpt = torch.load(ASYMTRACK_DIR / f"checkpoints/models/AsymTrack/{cfg_name}/AsymTrack_ep0500.pth.tar",
                          map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["net"], strict=True)
        net.backbone.switch_to_deploy()
        self.net = net.to(device).eval()
        self.device = device
        self.t = cfg.TEST
        self.to_cxcywh = box_xyxy_to_cxcywh
        self.mean = torch.tensor(cfg.DATA.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.STD, device=device).view(1, 3, 1, 1)

    def _tensor(self, patch):
        t = torch.from_numpy(patch).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (t - self.mean) / self.std

    def init(self, frame, box):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        z, _ = sample_target(rgb, box, self.t.TEMPLATE_FACTOR, self.t.TEMPLATE_SIZE)
        self.z = self._tensor(z)
        self.state = list(box)

    @torch.no_grad()
    def update(self, frame):
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, rf = sample_target(rgb, self.state, self.t.SEARCH_FACTOR, self.t.SEARCH_SIZE)
        xyxy, p_tl, p_br = self._heads(self._tensor(x))
        score = float(torch.sqrt(p_tl.max() * p_br.max()))
        self.state = crop_box_to_frame(self.to_cxcywh(xyxy).view(-1, 4).mean(0), self.t.SEARCH_SIZE, rf,
                                       self.state, W, H, margin=10)  # clip_box margin, as in AsymTrack / HiT
        return True, self.state, score

    def _heads(self, x):
        """= forward_backbone + forward_head (LINEAR neck), but the corner head also returns its two heatmaps."""
        feat = self.net.forward_backbone([x, self.z], train=False)[-1]
        B, h, w, C = feat.shape
        mem = self.net.bottleneck(feat.view(B, h * w, C))
        fs = self.net.feat_sz_s
        opt = mem.unsqueeze(-1).permute(0, 3, 2, 1).contiguous().view(-1, mem.shape[-1], fs, fs)
        return self.net.box_head(opt, return_dist=True)


class SUTrackModel:
    """Port of third_party/SUTrack/lib/test/tracker/sutrack.py (AAAI 2025, unified single ViT, Fast-iTPN encoder)
    for plain RGB: the RGB crop is duplicated into the 6-channel multi-modal input, as sutrack.py does, and the text
    branch gets an empty prompt. The T224 checkpoint contains its CLIP ViT-L/14 text encoder, so CLIP is built from
    those weights instead of clip.load() downloading it; with no prompt the text embedding is constant, so it is
    computed once. T224 uses one template; B224 and the bigger configs keep TEST.NUM_TEMPLATES templates and replace
    the newest one every UPDATE_INTERVALS frames when the windowed peak exceeds UPDATE_THRESHOLD, as sutrack.py does.
    Score = peak of the centre heatmap."""

    reid = True  # search box in self.state: can be re-run around a re-id candidate (SingleObject)

    def __init__(self, device, cfg_name="sutrack_t224"):
        import clip

        ckpt = torch.load(SUTRACK_DIR / f"checkpoints/train/sutrack/{cfg_name}/SUTRACK_ep0180.pth.tar",
                          map_location="cpu", weights_only=False)["net"]
        prefix = "text_encoder.clip."
        clip_sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        load = clip.load  # = clip.load(jit=False) on the weights we already have
        clip.load = lambda name, device="cpu": (clip.model.build_model(clip_sd).to(device), None)
        try:
            with repo_lib(SUTRACK_DIR):
                from lib.config.sutrack.config import cfg, update_config_from_file
                from lib.models.sutrack import build_sutrack, encoder as encoder_mod

                update_config_from_file(str(SUTRACK_DIR / f"experiments/sutrack/{cfg_name}.yaml"))
                encoder_mod.is_main_process = lambda: False  # = pretrained=False: no ImageNet Fast-iTPN weights
                net = build_sutrack(cfg)
        finally:
            clip.load = load
        net.load_state_dict(ckpt, strict=True)
        self.net = net.to(device).eval()
        self.device = device
        self.t = cfg.TEST
        fs = self.t.SEARCH_SIZE // cfg.MODEL.ENCODER.STRIDE
        hann = torch.hann_window(fs + 2, periodic=False)[1:-1]  # == hann2d(centered=True)
        self.window = (hann[:, None] * hann[None, :]).to(device)[None, None]
        self.num_templates = self.t.NUM_TEMPLATES
        self.update_interval = self.t.UPDATE_INTERVALS.DEFAULT
        self.update_threshold = self.t.UPDATE_THRESHOLD.DEFAULT
        self.mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=device).view(1, 6, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225] * 2, device=device).view(1, 6, 1, 1)
        with torch.no_grad():
            self.text_src = self.net.forward_textencoder(text_data=torch.zeros(1, 77, dtype=torch.long, device=device))

    def _tensor(self, patch):
        t = torch.from_numpy(patch).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (t.repeat(1, 2, 1, 1) - self.mean) / self.std  # RGB -> 6 channels (RGB + the same RGB as "X")

    def _template(self, rgb, box):
        """-> (template tensor, its box normalised to the crop) = transform_image_to_crop(box, box, rf, ts, True)."""
        z, rf = sample_target(rgb, box, self.t.TEMPLATE_FACTOR, self.t.TEMPLATE_SIZE)
        ts = self.t.TEMPLATE_SIZE
        cx, cy = (ts - 1) / 2, (ts - 1) / 2
        w, h = box[2] * rf, box[3] * rf
        anno = torch.tensor([[cx - w / 2, cy - h / 2, w, h]], device=self.device, dtype=torch.float32) / (ts - 1)
        return self._tensor(z), anno

    def init(self, frame, box):
        z, anno = self._template(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box)
        self.z, self.z_anno = [z] * self.num_templates, [anno]  # as sutrack.py: one annotation until the first update
        self.state = list(box)
        self.frame_id = 0

    @torch.no_grad()
    def update(self, frame):
        H, W = frame.shape[:2]
        self.frame_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, rf = sample_target(rgb, self.state, self.t.SEARCH_FACTOR, self.t.SEARCH_SIZE)
        enc = self.net.forward_encoder(self.z, [self._tensor(x)], self.z_anno, self.text_src, None)
        out = self.net.forward_decoder(feature=enc)
        self.state, score = box_from_center_head(self.net.decoder, out, self.window, self.t.SEARCH_SIZE, rf,
                                                 self.state, W, H, margin=10)  # clip_box margin, as in SUTrack
        if (self.num_templates > 1 and self.frame_id % self.update_interval == 0
                and float((self.window * out["score_map"]).max()) > self.update_threshold):
            z, anno = self._template(rgb, self.state)  # keep the first template, replace the newest
            self.z, self.z_anno = [*self.z, z], [*self.z_anno, anno]
            if len(self.z) > self.num_templates:
                self.z.pop(1)
            if len(self.z_anno) > self.num_templates:
                self.z_anno.pop(1)
        return True, self.state, score


class FocusTrackModel:
    """Port of third_party/FocusTrack/lib/test/tracker/focustrack.py (anti-UAV, OSTrack ViT-B + mask decoder + a
    "target in view" classification head). Search Region Adjustment: when the head says the target is out of the
    crop (p < T_LOGITS) and the heatmap peak is below T_SCORE, the search factor grows by ENLARGE_STEP per frame up to
    MAX_SEARCH_FACTOR and the Hann window is switched off; it snaps back once the target is found. No grid warping
    (TEST.SEARCH.USE_GRID is off in the released configs). Score = peak of the centre heatmap."""

    reid = True  # search box in self.state: can be re-run around a re-id candidate (SingleObject)

    def __init__(self, device, cfg_name="focustrack_stage2"):
        import types

        if "mmseg" not in sys.modules:  # its SegViT decoder only subclasses mmseg's BaseDecodeHead for 2 attributes
            class BaseDecodeHead(torch.nn.Module):
                def __init__(self, in_channels, channels, num_classes, **kwargs):
                    super().__init__()
                    self.in_channels, self.channels, self.num_classes = in_channels, channels, num_classes
                    self.conv_seg, self.loss_decode = torch.nn.Identity(), torch.nn.Identity()  # deleted by ATMHead

            names = ["mmseg", "mmseg.models", "mmseg.models.decode_heads", "mmseg.models.decode_heads.decode_head"]
            for n in names:
                sys.modules.setdefault(n, types.ModuleType(n))
            sys.modules[names[-1]].BaseDecodeHead = BaseDecodeHead
        with repo_lib(FOCUSTRACK_DIR):
            from lib.config.focustrack.config import cfg, update_config_from_file
            from lib.models.focustrack import build_focustrack

            update_config_from_file(str(FOCUSTRACK_DIR / f"experiments/focustrack/{cfg_name}.yaml"))
            net = build_focustrack(cfg, training=False)
            path = next((FOCUSTRACK_DIR / "output/checkpoints/train/focustrack").rglob(f"{cfg_name}/FocusTrack_ep*.pth.tar"))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["net"], strict=True)
        self.net = net.to(device).eval()
        self.device = device
        self.t = cfg.TEST
        fs = self.t.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        hann = torch.hann_window(fs + 2, periodic=False)[1:-1]  # == hann2d(centered=True)
        self.window = (hann[:, None] * hann[None, :]).to(device)[None, None]
        self.mean = torch.tensor(cfg.DATA.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.STD, device=device).view(1, 3, 1, 1)

    def _tensor(self, patch):
        t = torch.from_numpy(patch).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (t - self.mean) / self.std

    def init(self, frame, box):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        z, _ = sample_target(rgb, box, self.t.TEMPLATE.FACTOR, self.t.TEMPLATE.SIZE)
        self.z = self._tensor(z)
        self.state = list(box)
        self.search_factor, self.use_hann = self.t.SEARCH.FACTOR, True

    @torch.no_grad()
    def update(self, frame):
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        size = self.t.SEARCH.SIZE
        x, rf = sample_target(rgb, self.state, self.search_factor, size)
        out = self.net(template=self.z, search=self._tensor(x), training=False)
        window = self.window if self.use_hann else 1.0
        self.state, score = box_from_center_head(self.net.box_head, out, window, size, rf,
                                                 self.state, W, H, margin=10)  # clip_box margin, as in FocusTrack
        if self.t.USE_REGION_ADJUST:
            in_view = float(torch.softmax(out["logits"], dim=1)[0, -1])
            if in_view < self.t.T_LOGITS and score < self.t.T_SCORE:
                self.search_factor = min(self.search_factor + self.t.ENLARGE_STEP, self.t.MAX_SEARCH_FACTOR)
                self.use_hann = False
            else:
                self.search_factor, self.use_hann = self.t.SEARCH.FACTOR, True
        return True, self.state, score


class MCITrackModel:
    """Port of third_party/MCITrack/lib/test/tracker/mcitrack.py (AAAI 2025, Fast-iTPN encoder + Mamba "time neck"
    whose hidden state carries context from frame to frame). Online settings are the repo's UAV123 ones (TEST.*.UAV):
    every frame with a windowed peak > UPT is pushed to a memory bank of MB templates, every INTER frames the
    NUM_TEMPLATES - 1 online templates are re-sampled evenly from it, and the hidden state is reset whenever the peak
    drops below UPH. Score = peak of the centre heatmap."""

    reid = True  # search box in self.state: can be re-run around a re-id candidate (SingleObject)

    def __init__(self, device, cfg_name="mcitrack_b224", preset="UAV"):
        with repo_lib(MCITRACK_DIR):
            from lib.config.mcitrack.config import cfg, update_config_from_file
            from lib.models.mcitrack import build_mcitrack, encoder as encoder_mod

            update_config_from_file(str(MCITRACK_DIR / f"experiments/mcitrack/{cfg_name}.yaml"))
            encoder_mod.is_main_process = lambda: False  # = pretrained=False: no ImageNet Fast-iTPN weights
            net = build_mcitrack(cfg)
        ckpt = torch.load(MCITRACK_DIR / f"checkpoints/train/mcitrack/{cfg_name}/MCITRACK_ep{cfg.TEST.EPOCH:04d}.pth.tar",
                          map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["net"], strict=True)
        self.net = net.to(device).eval()
        self.device = device
        self.t = cfg.TEST
        self.n_layers = cfg.MODEL.NECK.N_LAYERS
        self.num_templates = self.t.NUM_TEMPLATES
        self.upt, self.uph = self.t.UPT[preset], self.t.UPH[preset]
        self.inter, self.mb = self.t.INTER[preset], self.t.MB[preset]
        fs = self.t.SEARCH_SIZE // cfg.MODEL.ENCODER.STRIDE
        hann = torch.hann_window(fs + 2, periodic=False)[1:-1]  # == hann2d(centered=True)
        self.window = (hann[:, None] * hann[None, :]).to(device)[None, None]
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _tensor(self, patch):
        t = torch.from_numpy(patch).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (t - self.mean) / self.std

    def _template(self, rgb, box):
        """-> (template tensor, its box normalised to the crop) = transform_image_to_crop(box, box, rf, ts, True)."""
        z, rf = sample_target(rgb, box, self.t.TEMPLATE_FACTOR, self.t.TEMPLATE_SIZE)
        ts = self.t.TEMPLATE_SIZE
        cx, cy = (ts - 1) / 2, (ts - 1) / 2
        w, h = box[2] * rf, box[3] * rf
        anno = torch.tensor([[cx - w / 2, cy - h / 2, w, h]], device=self.device, dtype=torch.float32) / (ts - 1)
        return self._tensor(z), anno

    def init(self, frame, box):
        z, anno = self._template(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box)
        self.z, self.z_anno = [z] * self.num_templates, [anno] * self.num_templates
        self.mem, self.mem_anno = list(self.z), list(self.z_anno)
        self.h_state = [None] * self.n_layers
        self.state = list(box)
        self.frame_id = 0

    @torch.no_grad()
    def update(self, frame):
        H, W = frame.shape[:2]
        self.frame_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, rf = sample_target(rgb, self.state, self.t.SEARCH_FACTOR, self.t.SEARCH_SIZE)
        enc = self.net.forward_encoder(self.z, [self._tensor(x)], self.z_anno)
        _, feat, h = self.net.forward_neck(enc, list(self.h_state))
        out = self.net.forward_decoder(feature=feat)
        self.state, score = box_from_center_head(self.net.decoder, out, self.window, self.t.SEARCH_SIZE, rf,
                                                 self.state, W, H, margin=10)  # clip_box margin, as in MCITrack
        conf = float((self.window * out["score_map"]).max())
        self.h_state = h if conf >= self.uph else [None] * self.n_layers
        if self.num_templates > 1 and conf > self.upt:
            z, anno = self._template(rgb, self.state)
            self.mem.append(z)
            self.mem_anno.append(anno)
            if len(self.mem) > self.mb:
                self.mem.pop(0)
                self.mem_anno.pop(0)
        if self.frame_id % self.inter == 0:  # keep the first template, re-sample the others from the memory bank
            step = len(self.mem) // self.num_templates
            for i in range(1, self.num_templates):
                self.z = [*self.z[:1], *self.z[2:], self.mem[step * i]]
                self.z_anno = [*self.z_anno[:1], *self.z_anno[2:], self.mem_anno[step * i]]
        return True, self.state, score


class OpenCVModel:
    """Any cv2 Tracker. CSRT/KCF/MOSSE have no confidence score, so their score is just 1/0 from `ok`."""

    def __init__(self, factory):
        self.factory = factory

    def init(self, frame, box):
        self.tracker = self.factory()  # fresh instance on every (re)init
        self.tracker.init(frame, tuple(int(round(v)) for v in box))

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        get_score = getattr(self.tracker, "getTrackingScore", None)
        return ok, list(box), float(get_score()) if get_score else float(ok)


class SiamZooModel:
    """Tracker from third_party/siam_zoo/siamtrack.py (float ONNX on the CPU via onnxruntime)."""

    def __init__(self, name):
        sys.path.insert(0, str(SIAM_ZOO_DIR))
        from siamtrack import make_tracker

        self.tracker = make_tracker(name, backend="onnx", models=str(SIAM_ZOO_DIR / "models"))

    def init(self, frame, box):
        self.tracker.init(frame, [float(v) for v in box])

    def update(self, frame):
        box, score = self.tracker.update(frame)
        return True, list(box), float(score)


def xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def crop(img, cx, cy, w, h):
    H, W = img.shape[:2]
    x1, y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x2, y2 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    return img[y1:y2, x1:x2] if x2 - x1 >= 2 and y2 - y1 >= 2 else None


class KalmanModel:
    """Single-object constant-velocity Kalman filter (SORT's KalmanBox) on detections: each frame it predicts,
    then corrects with the detection nearest to the prediction (gated by distance). No detection -> it coasts on
    the prediction and the score is 0, so SingleObject re-seeds after `patience` coasting frames."""

    uses_dets = True

    def __init__(self, gate=3.0):
        self.gate = gate  # max center distance, in target sizes

    def init(self, frame, box):
        x, y, w, h = box
        self.kf = KalmanBox([x, y, x + w, y + h])

    def update(self, frame, dets):
        pred = self.kf.predict()
        if len(dets):
            pc = (pred[:2] + pred[2:]) / 2
            dist = np.linalg.norm((dets[:, :2] + dets[:, 2:4]) / 2 - pc, axis=1)
            i = int(dist.argmin())
            if dist[i] < self.gate * max(np.sqrt((pred[2] - pred[0]) * (pred[3] - pred[1])), 8):
                self.kf.update(dets[i])
                return True, xywh(self.kf.box()), float(dets[i, 4])
        return True, xywh(pred), 0.0


class ParticleFilterModel:
    """Particle filter over (cx, cy, vx, vy) with a fixed box size. Weight = HSV colour-histogram similarity to the
    template (Bhattacharyya) x closeness to the nearest detection, so it can follow the target on appearance alone
    and is pulled back to the detector when it fires. Box size follows matched detections. Score = similarity."""

    uses_dets = True

    def __init__(self, n=300, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    @staticmethod
    def _hist(hsv, cx, cy, w, h):
        patch = crop(hsv, cx, cy, w, h)
        if patch is None:
            return None
        hist = cv2.calcHist([patch], [0, 1], None, [16, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)

    def init(self, frame, box):
        x, y, w, h = box
        self.wh = np.array([max(w, 4.0), max(h, 4.0)])
        self.p = np.zeros((self.n, 4))
        self.p[:, :2] = [x + w / 2, y + h / 2]
        self.ref = self._hist(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), x + w / 2, y + h / 2, *self.wh)

    def update(self, frame, dets):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        size = float(np.sqrt(self.wh.prod()))
        # predict: constant velocity + noise
        self.p[:, 2:] += self.rng.normal(0, 0.05 * size + 1, (self.n, 2))
        self.p[:, :2] += self.p[:, 2:] + self.rng.normal(0, 0.15 * size + 2, (self.n, 2))
        # weight: appearance x detection
        bh = np.array([1.0 if (hh := self._hist(hsv, cx, cy, *self.wh)) is None
                       else cv2.compareHist(self.ref, hh, cv2.HISTCMP_BHATTACHARYYA) for cx, cy in self.p[:, :2]])
        w = np.exp(-20 * bh ** 2)
        if len(dets):
            centers = (dets[:, :2] + dets[:, 2:4]) / 2
            d2 = ((self.p[:, None, :2] - centers[None]) ** 2).sum(-1).min(1)
            w *= 0.05 + np.exp(-d2 / (2 * size ** 2))
        w = w / w.sum() if w.sum() > 0 else np.full(self.n, 1 / self.n)
        est = w @ self.p
        # follow the detector's box size when a detection is close to the estimate
        if len(dets):
            dist = np.linalg.norm(centers - est[:2], axis=1)
            i = int(dist.argmin())
            if dist[i] < 2 * size:
                self.wh = 0.7 * self.wh + 0.3 * np.maximum(dets[i, 2:4] - dets[i, :2], 4)
        # systematic resampling when the effective sample size drops
        if 1 / (w ** 2).sum() < self.n / 2:
            idx = np.searchsorted(np.cumsum(w), (self.rng.random() + np.arange(self.n)) / self.n)
            self.p = self.p[np.minimum(idx, self.n - 1)]
        hist = self._hist(hsv, est[0], est[1], *self.wh)
        score = 0.0 if hist is None else 1 - cv2.compareHist(self.ref, hist, cv2.HISTCMP_BHATTACHARYYA)
        return True, [est[0] - self.wh[0] / 2, est[1] - self.wh[1] / 2, *self.wh], float(score)


class MeanShiftModel:
    """OpenCV meanShift / CamShift on the back-projection of the target's hue-saturation histogram.
    CamShift also adapts the window size. Score = mean back-projection inside the window (0..1)."""

    def __init__(self, cam=False):
        self.cam = cam
        self.term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def init(self, frame, box):
        x, y, w, h = (int(round(v)) for v in box)
        self.win = (x, y, max(w, 2), max(h, 2))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = hsv[y:y + self.win[3], x:x + self.win[2]]
        # skip only dark pixels: the OpenCV sample also drops low saturation, but a grey UAV on blue sky IS the grey
        mask = cv2.inRange(roi, (0, 0, 32), (180, 255, 255))
        self.hist = cv2.calcHist([roi], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        if self.cam:
            _, win = cv2.CamShift(prob, self.win, self.term)
        else:
            _, win = cv2.meanShift(prob, self.win, self.term)
        ok = win[2] >= 2 and win[3] >= 2
        if ok:
            self.win = win
        x, y, w, h = self.win
        score = float(prob[y:y + h, x:x + w].mean() / 255) if ok else 0.0
        return ok, [x, y, w, h], score


class IVTModel:
    """IVT (Ross et al., IJCV 2008), compact version: particle filter over (cx, cy, scale); each candidate patch
    (32x32 grey) is scored by its reconstruction error in a PCA subspace of the target's appearance. The subspace
    is updated incrementally every `batch` frames (sequential Karhunen-Loeve with mean update and forgetting)."""

    def __init__(self, n=300, patch=32, k=16, batch=5, forget=0.95, seed=0):
        self.n, self.ps, self.k, self.batch, self.f = n, patch, k, batch, forget
        self.rng = np.random.default_rng(seed)

    def _patch(self, gray, cx, cy, s):
        p = crop(gray, cx, cy, self.w0 * s, self.h0 * s)
        return None if p is None else cv2.resize(p, (self.ps, self.ps), interpolation=cv2.INTER_AREA).ravel() / 255.0

    def _err(self, P):
        D = P - self.mu
        if self.U is not None:
            D = D - (D @ self.U) @ self.U.T
        return (D ** 2).mean(1)

    def init(self, frame, box):
        x, y, w, h = box
        self.w0, self.h0 = max(w, 4.0), max(h, 4.0)
        self.state = np.array([x + w / 2, y + h / 2, 1.0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mu = self._patch(gray, *self.state)
        self.U, self.S, self.n_seen, self.buf = None, None, 1, []

    def _update_subspace(self):
        B = np.array(self.buf).T  # d x m
        m, n = B.shape[1], self.f * self.n_seen
        mu_b = B.mean(1)
        extra = np.sqrt(n * m / (n + m)) * (mu_b - self.mu)
        cols = [B - mu_b[:, None], extra[:, None]]
        if self.U is not None:
            cols.insert(0, self.f * self.U * self.S)
        U, S, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
        self.U, self.S = U[:, :self.k], S[:self.k]
        self.mu = (n * self.mu + m * mu_b) / (n + m)
        self.n_seen, self.buf = n + m, []

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = np.sqrt(self.w0 * self.h0) * self.state[2]
        cand = self.state + self.rng.normal(0, 1, (self.n, 3)) * [0.2 * size + 2, 0.2 * size + 2, 0.01]
        # without a bound the scale shrinks onto flat sky, which any subspace reconstructs perfectly
        cand[:, 2] = np.clip(cand[:, 2], 0.7, 1.5)
        patches = [self._patch(gray, *c) for c in cand]
        valid = np.array([p is not None for p in patches])
        if not valid.any():
            return False, [*(self.state[:2] - [self.w0 / 2, self.h0 / 2]), self.w0, self.h0], 0.0
        err = np.full(self.n, np.inf)
        err[valid] = self._err(np.array([p for p in patches if p is not None]))
        best = int(err.argmin())  # IVT takes the MAP particle
        self.state = cand[best]
        self.buf.append(patches[best])
        if len(self.buf) >= self.batch:
            self._update_subspace()
        w, h = self.w0 * self.state[2], self.h0 * self.state[2]
        score = float(np.exp(-err[best] / 0.01))  # mean squared reconstruction error of 0.01 -> 0.37
        return True, [self.state[0] - w / 2, self.state[1] - h / 2, w, h], score


# name -> (display name, kind, model builder, lost-score threshold).
# kind: "classic" (hand-crafted features), "filter" (Bayesian filters on detections), "deep" (learned)
SOT = {
    "csrt":      ("CSRT",      "classic", lambda d: OpenCVModel(cv2.TrackerCSRT_create), 0.5),
    "kcf":       ("KCF",       "classic", lambda d: OpenCVModel(cv2.TrackerKCF_create), 0.5),
    "mosse":     ("MOSSE",     "classic", lambda d: OpenCVModel(cv2.legacy.TrackerMOSSE_create), 0.5),
    "mil":       ("MIL",       "classic", lambda d: OpenCVModel(cv2.TrackerMIL_create), 0.5),
    "tld":       ("TLD",       "classic", lambda d: OpenCVModel(cv2.legacy.TrackerTLD_create), 0.5),
    "meanshift": ("MeanShift", "classic", lambda d: MeanShiftModel(cam=False), 0.1),
    "camshift":  ("CAMShift",  "classic", lambda d: MeanShiftModel(cam=True), 0.1),
    "ivt":       ("IVT",       "classic", lambda d: IVTModel(), 0.1),
    "kalman":    ("Kalman",    "filter",  lambda d: KalmanModel(), 0.01),
    "particle":  ("Particle",  "filter",  lambda d: ParticleFilterModel(), 0.05),
    "ortrack_deit": ("ORTrack-DeiT", "deep", lambda d: ORTrackModel(d, "deit_tiny_patch16_224"), 0.3),
    "asymtrack": ("AsymTrack-B", "deep",  lambda d: AsymTrackModel(d, "base"), 0.3),
    "sutrack":   ("SUTrack-T", "deep",    lambda d: SUTrackModel(d), 0.3),
    "sutrack_b": ("SUTrack-B", "deep",    lambda d: SUTrackModel(d, "sutrack_b224"), 0.3),
    "focustrack": ("FocusTrack", "deep",  lambda d: FocusTrackModel(d), 0.3),
    "mcitrack_b": ("MCITrack-B", "deep",  lambda d: MCITrackModel(d, "mcitrack_b224"), 0.3),
    "avtrack":   ("AVTrack",   "deep",    lambda d: SiamZooModel("avtrack_deit"), 0.3),
}
MOT = ["bytetrack", "botsort", "deepsort", "sort"]
GROUPS = {"mot": MOT, **{g: [k for k, v in SOT.items() if v[1] == g] for g in ("classic", "filter", "deep")}}
ALL_TRACKERS = [*MOT, *SOT]


def max_iou(box, boxes):
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return float((inter / (area(np.asarray(box)) + area(boxes) - inter + 1e-9)).max())


def center_size(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 0)), 1.0)


class SingleObject:
    """Runs a single-object tracker with a Kalman motion model (KalmanBox) on top, for occlusions such as a UAV
    crossing cables or flying behind branches.

    healthy frame  tracker ok, score >= lost_thr, box consistent with the Kalman prediction (no jump, no size
                   explosion), and no confident detection near the prediction that the box misses
                   -> Kalman is corrected with the tracker box
    bad frame      -> the box coasts on the Kalman prediction; any detection near the prediction, even a weak one
                   (the detector's confidence drops while the UAV is on the cables), re-acquires the target: the
                   tracker is re-initialised on it with the SAME id
    lost           after `patience` bad frames in a row, or when a strong detection elsewhere keeps contradicting
                   the tracker for `patience` frames (slow drift onto background) -> re-seed on the best detection,
                   new id. Detections far from the prediction never break a healthy track (false positives).
    Boxes scoring below `min_score` (coasting ones score 0) are not reported; the track itself goes on underneath,
    so the id is kept when the score recovers.

    Re-identification (reid_thr > 0, deep trackers with `reid = True`): a detection only re-acquires the target if
    the tracker itself, with its current template, finds the target around it with a score >= reid_thr (template
    matching with the tracker's own network) at a similar size, on `reid_confirm` frames in a row at a steady
    position (one missed frame allowed). A small UAV and a bird score alike on one frame, but the birds of a flock
    are detected at different places from frame to frame. Then the track goes on from there with the SAME id and
    the template is kept (no re-init on the detection). Once a target exists, a lost one is searched for this way
    over all detections in the frame (UAV behind a building, coming out elsewhere); a UAV seen as steadily that does
    not match it is a different one and gets a new id."""

    def __init__(self, name, model, conf, lost_thr, patience, names, gate=3.0, strong=0.6, min_score=0.0,
                 reid_thr=0.0, reid_max=5, reid_confirm=5):
        self.name, self.model, self.min_score = name, model, min_score
        self.reid_max, self.reid_confirm = reid_max, reid_confirm
        self.reid_thr = reid_thr if getattr(model, "reid", False) else 0.0  # others keep the plain re-seeding
        self.reid_log = []  # (candidate det, score) of the last verification, for debugging
        self.chain = None  # re-id candidate being confirmed: [box, score, frames seen, frames missed]
        self.conf, self.lost_thr, self.patience = conf, lost_thr, patience
        self.names, self.gate, self.strong = names, gate, strong
        self.tid, self.active, self.label = 0, False, ""

    def _start(self, frame, det, new_id):
        x1, y1, x2, y2, c, k = det
        self.model.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self.kf = KalmanBox(det)
        self.bad = self.contradicted = 0
        self.size = center_size(det)[2]
        if new_id:
            self.tid += 1
            self.label = self.names[int(k)]
        self.active = True
        return [(self.tid, (x1, y1, x2, y2), float(c), self.label)]

    def _verify(self, frame, cands):
        """Run the tracker on a search region centred on each candidate (most confident first), leaving its state
        untouched. -> [(det, box xyxy, score, match)]: match = the tracker finds the target on this candidate with
        score >= reid_thr and at 0.5-2x the target's last size (the network rescales every crop to the template, so
        it does not see size: a far dot and a close UAV can both score high)."""
        m, out, self.reid_log = self.model, [], []
        snap = lambda d: {k: (list(v) if isinstance(v, list) else v) for k, v in d.items()}
        saved = snap(vars(m))
        for d in cands[np.argsort(-cands[:, 4])][:self.reid_max]:
            m.state = xywh(d[:4])
            ok, (x, y, w, h), score = m.update(frame)
            vars(m).clear()
            vars(m).update(snap(saved))
            box = np.array([x, y, x + w, y + h])
            dcx, dcy, dsz = center_size(d)
            cx, cy, sz = center_size(box)
            on_it = ok and np.hypot(cx - dcx, cy - dcy) < max(dsz, 16)  # found it on this candidate, not next to it
            score = float(score) if on_it else 0.0
            match = score >= self.reid_thr and 0.5 < sz / self.size < 2.0
            self.reid_log.append((d, score))
            out.append((d, box if on_it else d[:4], score, match))
        return out

    def _reacquire(self, frame, cands, new_ok):
        """Re-id step: verify the candidates and follow the most promising one (a match first) over the frames, as
        long as it moves less than 3/4 of its size per frame (one missed frame allowed). After reid_confirm frames
        -> _resume if it matched the target on all but one of them; with `new_ok`, a different UAV (seen steadily,
        mean detector confidence >= conf, but not the target) -> _start with a new id. Else None."""
        res = self._verify(frame, cands) if len(cands) else []
        best = max(res, key=lambda r: (r[3], r[2]), default=None)
        c = self.chain
        if c is not None:
            ccx, ccy, csz = center_size(c["box"])
            step = [r for r in res if np.hypot(*np.subtract(center_size(r[1])[:2], (ccx, ccy))) < 0.75 * max(csz, 8)]
            if step and not (best[3] and not max(step, key=lambda r: (r[3], r[2]))[3]):
                det, box, score, match = max(step, key=lambda r: (r[3], r[2]))
                c.update(det=det, box=box, score=score, n=c["n"] + 1, miss=0, match=c["match"] + match,
                         conf=c["conf"] + det[4])
            elif not step and c["miss"] < 1:
                c["miss"] += 1
            else:
                c = None  # lost it, or a match showed up elsewhere
        if c is None and best is not None:
            det, box, score, match = best
            c = dict(det=det, box=box, score=score, n=1, miss=0, match=int(match), conf=det[4])
        self.chain = c
        if c is None or c["n"] < self.reid_confirm:
            return None
        if c["match"] >= c["n"] - 1:
            return self._resume(c["box"], c["score"])
        if new_ok and c["conf"] / c["n"] >= self.conf:
            self.chain = None
            return self._start(frame, c["det"], new_id=True)
        return None

    def _resume(self, box, score):
        """Continue the same track at `box`, keeping the tracker's template."""
        self.model.state = xywh(box)
        self.kf = KalmanBox(box)
        self.bad = self.contradicted = 0
        self.size = center_size(box)[2]
        self.active, self.chain = True, None
        return [(self.tid, tuple(box), score, self.label)]

    def update(self, frame, dets):
        return [t for t in self._track(frame, dets) if t[2] >= self.min_score]

    def _track(self, frame, dets):
        good = dets[dets[:, 4] >= self.conf]
        if self.active:
            pred = self.kf.predict()
            if getattr(self.model, "uses_dets", False):
                ok, (x, y, w, h), score = self.model.update(frame, dets)
            else:
                ok, (x, y, w, h), score = self.model.update(frame)
            box = np.array([x, y, x + w, y + h])

            pcx, pcy, psz = center_size(pred)
            gate = self.gate * max(psz, 16)
            cx, cy, sz = center_size(box)
            jumped = np.hypot(cx - pcx, cy - pcy) > gate
            # a sudden size change is a jump to something else, unless (with re-id, which checks the size of what it
            # picks up) the tracker is confident: a fixed-wing banking shows its full wingspan from one frame to the next
            if not 0.5 < sz / psz < 2.0 and not (self.reid_thr and score >= self.strong):
                jumped = True

            near = None  # detection closest to the prediction, within the gate
            if len(dets):
                dist = np.hypot((dets[:, 0] + dets[:, 2]) / 2 - pcx, (dets[:, 1] + dets[:, 3]) / 2 - pcy)
                i = int(dist.argmin())
                if dist[i] < gate:
                    near = dets[i]
            missed = near is not None and near[4] >= self.conf and max_iou(box, near[None, :4]) < 0.1

            strong = good[good[:, 4] >= self.strong]
            self.contradicted = self.contradicted + 1 if len(strong) and max_iou(box, strong[:, :4]) < 0.1 else 0
            if self.reid_thr and score >= self.strong:
                self.contradicted = 0  # a confident tracker is not overruled by another UAV elsewhere; re-id can fix

            if self.contradicted > self.patience:  # drifted: a strong detection elsewhere, long enough
                self.active = False
            elif ok and score >= self.lost_thr and not jumped and not missed:
                self.kf.update(box)
                self.bad, self.size = 0, sz
                return [(self.tid, tuple(box), score, self.label)]
            elif near is not None and not self.reid_thr:  # re-acquire the same target from the detector
                return self._start(frame, near, new_id=False)
            elif near is not None and (out := self._reacquire(frame, near[None], new_ok=False)):
                return out
            else:
                self.bad += 1
                if self.bad <= self.patience:
                    return [(self.tid, tuple(pred), 0.0, self.label + " (coast)")]
                self.active = False

        if self.reid_thr and self.tid:  # lost: only the same target, wherever it shows up again
            self.bad += 1
            return self._reacquire(frame, dets, new_ok=True) or []
        if len(good):
            return self._start(frame, good[good[:, 4].argmax()], new_id=True)
        return []


# ------------------------------------------------------------------------------------------------ runner
def build_trackers(args, fps, names, device):
    out = []
    for t in args.trackers:
        if t == "bytetrack":
            out.append(ByteTrack(fps, names, args.patience))
        elif t == "botsort":
            out.append(BoTSORT(fps, names, args.patience))
        elif t == "deepsort":
            out.append(DeepSORT(fps, names, args.conf, args.patience))
        elif t == "sort":
            out.append(SORT(names, args.conf, args.patience))
        else:
            name, _, build, thr = SOT[t]
            out.append(SingleObject(name, build(device), args.conf, thr, args.patience, names,
                                    min_score=getattr(args, "min_score", 0.0),
                                    reid_thr=getattr(args, "reid", 0.0)))
    return out


def ema(prev, value, a=0.1):
    return value if prev is None else (1 - a) * prev + a * value


def grid_shape(n, frame_w, frame_h, max_w, max_h):
    """(rows, cols) that gives the biggest panels for n frames of this aspect ratio in a max_w x max_h window."""
    def scale(cols):
        rows = (n + cols - 1) // cols
        return min(max_w / (cols * frame_w), max_h / (rows * frame_h))
    cols = max(range(1, n + 1), key=scale)
    return (n + cols - 1) // cols, cols


def fit_size(w, h, max_side):
    """Size after downscaling (never crop, never upscale) so the longer side is at most max_side; 0 = off."""
    s = min(1.0, max_side / max(w, h)) if max_side else 1.0
    return round(w * s), round(h * s)


def fit_frame(frame, max_side):
    h, w = frame.shape[:2]
    size = fit_size(w, h, max_side)
    return frame if size == (w, h) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def panel_size(n, frame_w, frame_h, max_w, max_h):
    """Largest panel size (keeping the frame's aspect ratio) so the whole grid fits in max_w x max_h."""
    rows, cols = grid_shape(n, frame_w, frame_h, max_w, max_h)
    s = min(max_w / (cols * frame_w), max_h / (rows * frame_h))
    return int(frame_w * s), int(frame_h * s), cols


def draw_panel(frame, tracks, hud, panel_w):
    """Boxes with id / label / score, and the `hud` lines on a dark band at the top."""
    img = frame.copy()
    s = img.shape[1] / panel_w * 0.75  # the panel is shrunk to panel_w px, so draw text big enough to survive it
    fs, lw = 0.5 * s, max(1, round(1.5 * s))
    for tid, (x1, y1, x2, y2), score, label in tracks:
        x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
        color = id_color(tid)
        text = f"#{tid} {label} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, text, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw)
    band = img[:int((12 + 22 * len(hud)) * s)]
    band[:] = (band * 0.45).astype(np.uint8)
    for i, line in enumerate(hud):
        org = (int(8 * s), int((24 + 22 * i) * s))
        cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (0, 0, 0), lw * 3)
        cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (255, 255, 255), lw)
    return img


def video_gt(src, scale):
    """Ground truth of <video>.json (Anti-UAV: exist + gt_rect x, y, w, h) per frame as (x1, y1, x2, y2) in the
    working frame (original pixels * scale), None on frames without the target; None if there is no json."""
    path = src.with_suffix(".json") if isinstance(src, Path) else None
    if path is None or not path.exists():
        return None
    gt = json.loads(path.read_text())
    return [tuple(v * scale for v in (r[0], r[1], r[0] + r[2], r[1] + r[3])) if e and len(r) == 4 else None
            for e, r in zip(gt["exist"], gt["gt_rect"])]


class Precision:
    """Running precision: share of all boxes shown so far that match the GT target at IoU >= 0.5."""

    def __init__(self):
        self.tp = self.n = 0

    def add(self, boxes, g):
        self.n += len(boxes)
        self.tp += int(g is not None and len(boxes) > 0 and max_iou(np.array(g), np.array(boxes)) >= 0.5)

    def __str__(self):
        return f"precision {self.tp / self.n:.2f} ({self.tp}/{self.n})" if self.n else "precision -"


def make_grid(panels, pw, ph, cols):
    rows = (len(panels) + cols - 1) // cols
    grid = np.zeros((ph * rows, pw * cols, 3), np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = cv2.resize(p, (pw, ph))
    return grid


def run_video(model, src, args, device):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[skip] cannot open {src}")
        return True
    name = src.name if isinstance(src, Path) else f"cam{src}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    trackers = build_trackers(args, fps, model.names, device)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    work_w, work_h = fit_size(src_w, src_h, args.max_side)
    panel_w, panel_h, cols = panel_size(len(trackers), work_w, work_h, args.width, args.height)
    if (work_w, work_h) != (src_w, src_h):
        print(f"[resize] {src_w}x{src_h} -> {work_w}x{work_h} (--max-side {args.max_side})")
    print(f"[run] {name} | trackers: {', '.join(t.name for t in trackers)}")

    stats = {t.name: {"ms": [], "frames": 0, "ids": set(), "ema": None, "prec": Precision()} for t in trackers}
    gt = video_gt(src, work_w / src_w)
    det_prec, det_ema = Precision(), None
    det_ms_all, writer, out_path, grid = [], None, None, None
    keep_going, paused, n = True, False, 0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame = fit_frame(frame, args.max_side)
            n += 1
            t0 = time.perf_counter()
            res = model.predict(frame, conf=args.det_low, iou=args.iou, imgsz=args.imgsz,
                                agnostic_nms=args.agnostic, device=device, verbose=False)[0]
            dets = res.boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
            if args.max_box:  # drop implausibly large boxes (frame-sized false positives on branches/cables)
                area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                dets = dets[area <= args.max_box * frame.shape[0] * frame.shape[1]]
            det_ms = (time.perf_counter() - t0) * 1000
            det_ms_all.append(det_ms)
            det_ema = ema(det_ema, det_ms)
            g = gt[n - 1] if gt is not None and n <= len(gt) else None
            shown = dets[dets[:, 4] >= args.conf]  # detections at the confidence that starts / feeds tracks
            det_prec.add(shown[:, :4], g)
            det_q = str(det_prec) if gt is not None else f"conf {shown[:, 4].max():.2f}" if len(shown) else "conf -"

            panels = []
            for t in trackers:
                t0 = time.perf_counter()
                tracks = t.update(frame, dets)
                ms = (time.perf_counter() - t0) * 1000
                s = stats[t.name]
                s["ms"].append(ms)
                s["frames"] += bool(tracks)
                s["ids"].update(tid for tid, *_ in tracks)
                s["ema"] = ema(s["ema"], ms)  # smooth ms, then invert for FPS
                s["prec"].add([b for _, b, *_ in tracks], g)
                trk_q = str(s["prec"]) if gt is not None else \
                    f"score {max(sc for *_, sc, _ in tracks):.2f}" if tracks else "score -"
                hud = [f"{t.name} | total {1000 / (det_ema + s['ema']):.0f} FPS (detector + tracker)",
                       f"Detector: every frame, {det_ema:.1f} ms = {1000 / det_ema:.0f} FPS | {det_q}",
                       f"Tracker:  every frame, {s['ema']:.1f} ms = {1000 / s['ema']:.0f} FPS | {trk_q}"]
                panels.append(draw_panel(frame, tracks, hud, panel_w))
            grid = make_grid(panels, panel_w, panel_h, cols)

            if args.save and writer is None:
                out_dir = Path(args.out)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{Path(name).stem}_track.mp4"
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (grid.shape[1], grid.shape[0]))
            if writer:
                writer.write(grid)

        if args.show and grid is not None:
            cv2.imshow("UAV tracking", grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                keep_going = False
                break
            if key == ord("n"):
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer:
        writer.release()
        print(f"[saved] {out_path}")

    warm = slice(5, None) if n > 10 else slice(None)  # skip warm-up frames in timing
    det = np.mean(det_ms_all[warm])
    print(f"\n{name}: {n} frames | detector {det:.1f} ms/frame" + (f" | detector {det_prec}" if gt else ""))
    print(f"{'tracker':<11} {'ms/frame':>9} {'FPS':>7} {'frames w/ track':>16} {'unique IDs':>11}"
          + (f" {'precision':>10}" if gt else ""))
    for tname, s in stats.items():
        ms = np.mean(s["ms"][warm])
        print(f"{tname:<11} {ms:>9.2f} {1000 / (det + ms):>7.1f} "
              f"{s['frames']:>9}/{n:<6} {len(s['ids']):>11}"
              + (f" {s['prec'].tp / max(s['prec'].n, 1):>10.3f}" if gt else ""))
    print()
    return keep_going


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="YOLOv8n_HuggingFace_TomSmaildrone-yolo-v1.pt")
    ap.add_argument("--source", default="test_videos", help="video file, directory, or webcam index")
    ap.add_argument("--trackers", nargs="+", default=ALL_TRACKERS, choices=ALL_TRACKERS + list(GROUPS),
                    help="tracker names and/or groups: mot, classic, filter, deep")
    ap.add_argument("--conf", type=float, default=0.25, help="detection conf to start tracks / feed DeepSORT")
    ap.add_argument("--det-low", type=float, default=0.3, help="lowest detection conf kept (ByteTrack 2nd stage)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-side", type=int, default=1280,
                    help="downscale frames whose longer side is bigger (aspect kept, no crop); 0 = off")
    ap.add_argument("--no-agnostic", dest="agnostic", action="store_false",
                    help="per-class NMS (default is class-agnostic: the model often puts a quadcopter AND a "
                         "fixed-wing box on the same UAV, which splits tracks)")
    ap.add_argument("--patience", type=int, default=10,
                    help="frames a lost track keeps being drawn at its Kalman prediction (all trackers)")
    ap.add_argument("--min-score", type=float, default=0.25,
                    help="hide single-object tracker boxes scoring below this (incl. coasting ones); 0 = show all")
    ap.add_argument("--reid", type=float, default=0.5,
                    help="deep single-object trackers: a lost target is only picked up again (same id) where the "
                         "tracker's own template match scores at least this; 0 = off (re-seed on any detection)")
    ap.add_argument("--max-box", type=float, default=0.1,
                    help="ignore detections larger than this fraction of the frame area; 0 = off")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (auto if omitted)")
    ap.add_argument("--width", type=int, default=1600, help="max width of the side-by-side window")
    ap.add_argument("--height", type=int, default=900, help="max height of the side-by-side window")
    ap.add_argument("--save", action="store_true", help="save the grid video")
    ap.add_argument("--out", default="runs/track")
    ap.add_argument("--no-show", dest="show", action="store_false", help="disable live window")
    args = ap.parse_args()

    args.trackers = list(dict.fromkeys(x for t in args.trackers for x in GROUPS.get(t, [t])))
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
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
