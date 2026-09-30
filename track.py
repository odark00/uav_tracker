#!/usr/bin/env python3
"""
Compare UAV trackers on the same YOLOv8 detections, drawn side by side in real time.

Multi-object ("mot", fed by the detector every frame):
           bytetrack  - IoU + Kalman, two-stage matching incl. low-score boxes (ultralytics BYTETracker)
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
           lightfc    - MobileNetV2 Siamese + center head (third_party/LightFC, PyTorch on GPU)
           nanotrack  - NanoTrack v2 (OpenCV TrackerNano)
           vittrack   - lightweight ViT tracker (OpenCV TrackerVit)
           lighttrack - NAS-found Siamese tracker   \
           avtrack    - AVTrack-DeiT, adaptive ViT   |
           siamfc     - SiamFC (AlexNet)             |  third_party/siam_zoo float ONNX, CPU (onnxruntime)
           siamrpn    - SiamRPN (AlexNet)            |
           siamrpnpp  - SiamRPN++ (MobileNetV2)      |
           dasiamrpn  - DaSiamRPN                    |
           mixformer  - MixFormerV2-S               /
The detector runs once per frame and every tracker gets the same detections (boxes larger than --max-box of the
frame are dropped). Single-object trackers start on the most confident detection; a Kalman motion model on top
carries them through occlusions (cables, branches) and re-acquires the same target from nearby detections,
see SingleObject.

Run:       python track.py                                        # all trackers, all videos in ./test_videos
           python track.py --trackers bytetrack lightfc --source test_videos/clip.mp4
           python track.py --trackers deep                         # one group: mot / classic / filter / deep
           python track.py --save --no-show                       # write grid videos to ./runs/track

Keys:      q / Esc = quit, n = next video, space = pause/resume
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from infer import MODELS, collect_sources, resolve_model

ROOT = Path(__file__).resolve().parent
LIGHTFC_DIR = ROOT / "third_party/LightFC"
LIGHTFC_CFG = "mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou"
NANOTRACK_DIR = ROOT / "third_party/nanotrack"
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


class LightFCModel:
    """Port of third_party/LightFC/lib/test/tracker/lightfc.py that also returns the peak score."""

    def __init__(self, device):
        sys.path.insert(0, str(LIGHTFC_DIR))
        from lib.models import LightFC
        from lib.utils.load import load_yaml

        cfg = load_yaml(str(LIGHTFC_DIR / f"experiments/lightfc/{LIGHTFC_CFG}.yaml"))
        net = LightFC(cfg=cfg, env_num=None, training=False)
        ckpt = torch.load(LIGHTFC_DIR / "checkpoints/lightfc_ep0400.pth.tar", map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["net"], strict=True)
        for m in list(net.backbone.modules()) + list(net.head.modules()):
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        self.net = net.to(device).eval()
        self.device = device
        self.t = cfg.TEST
        fs = self.t.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        hann = torch.hann_window(fs + 2, periodic=False)[1:-1]  # == LightFC hann1d(centered=True)
        self.window = (hann[:, None] * hann[None, :]).to(device)[None, None]
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _tensor(self, patch):
        t = torch.from_numpy(patch).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        return (t - self.mean) / self.std

    @torch.no_grad()
    def init(self, frame, box):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        z, _ = sample_target(rgb, box, self.t.TEMPLATE_FACTOR, self.t.TEMPLATE_SIZE)
        self.z_feat = self.net.forward_backbone(self._tensor(z))
        self.state = list(box)

    @torch.no_grad()
    def update(self, frame):
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, rf = sample_target(rgb, self.state, self.t.SEARCH_FACTOR, self.t.SEARCH_SIZE)
        out = self.net.forward_tracking(z_feat=self.z_feat, x=self._tensor(x))
        score = float(out["score_map"].max())
        resp = self.window * out["score_map"]
        cx, cy, w, h = (self.net.head.cal_bbox(resp, out["size_map"], out["offset_map"]).view(-1, 4).mean(0)
                        * self.t.SEARCH_SIZE / rf).tolist()
        half = 0.5 * self.t.SEARCH_SIZE / rf
        cx += self.state[0] + 0.5 * self.state[2] - half
        cy += self.state[1] + 0.5 * self.state[3] - half
        x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
        m = 2  # clip_box margin, as in LightFC
        x1, y1 = min(max(0, x1), W - m), min(max(0, y1), H - m)
        x2, y2 = min(max(m, x2), W), min(max(m, y2), H)
        self.state = [x1, y1, max(m, x2 - x1), max(m, y2 - y1)]
        return True, self.state, score


def nano_params():
    p = cv2.TrackerNano_Params()
    p.backbone = str(NANOTRACK_DIR / "nanotrack_backbone_sim.onnx")
    p.neckhead = str(NANOTRACK_DIR / "nanotrack_head_sim.onnx")
    return p


def vit_params():
    p = cv2.TrackerVit_Params()
    p.net = str(SIAM_ZOO_DIR / "models/vittrack_cv/vittrack.onnx")
    return p


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
    "lightfc":   ("LightFC",   "deep",    lambda d: LightFCModel(d), 0.3),
    "nanotrack": ("NanoTrack", "deep",    lambda d: OpenCVModel(lambda: cv2.TrackerNano_create(nano_params())), 0.3),
    "vittrack":  ("VitTrack",  "deep",    lambda d: OpenCVModel(lambda: cv2.TrackerVit_create(vit_params())), 0.3),
    "lighttrack": ("LightTrack", "deep",  lambda d: SiamZooModel("lighttrack"), 0.3),
    "avtrack":   ("AVTrack",   "deep",    lambda d: SiamZooModel("avtrack_deit"), 0.3),
    "siamfc":    ("SiamFC",    "deep",    lambda d: SiamZooModel("siamfc"), 2.0),  # raw xcorr response, ~4-7 on target
    "siamrpn":   ("SiamRPN",   "deep",    lambda d: SiamZooModel("siamrpn_alex"), 0.3),
    "siamrpnpp": ("SiamRPN++", "deep",    lambda d: SiamZooModel("siamrpnpp_mobilev2"), 0.3),
    "dasiamrpn": ("DaSiamRPN", "deep",    lambda d: SiamZooModel("dasiamrpn"), 0.3),
    "mixformer": ("MixFormerV2", "deep",  lambda d: SiamZooModel("mixformerv2_s"), 0.1),
}
MOT = ["bytetrack", "deepsort", "sort"]
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
                   new id. Detections far from the prediction never break a healthy track (false positives)."""

    def __init__(self, name, model, conf, lost_thr, patience, names, gate=3.0, strong=0.6):
        self.name, self.model = name, model
        self.conf, self.lost_thr, self.patience = conf, lost_thr, patience
        self.names, self.gate, self.strong = names, gate, strong
        self.tid, self.active, self.label = 0, False, ""

    def _start(self, frame, det, new_id):
        x1, y1, x2, y2, c, k = det
        self.model.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self.kf = KalmanBox(det)
        self.bad = self.contradicted = 0
        if new_id:
            self.tid += 1
            self.label = self.names[int(k)]
        self.active = True
        return [(self.tid, (x1, y1, x2, y2), float(c), self.label)]

    def update(self, frame, dets):
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
            jumped = np.hypot(cx - pcx, cy - pcy) > gate or not 0.5 < sz / psz < 2.0

            near = None  # detection closest to the prediction, within the gate
            if len(dets):
                dist = np.hypot((dets[:, 0] + dets[:, 2]) / 2 - pcx, (dets[:, 1] + dets[:, 3]) / 2 - pcy)
                i = int(dist.argmin())
                if dist[i] < gate:
                    near = dets[i]
            missed = near is not None and near[4] >= self.conf and max_iou(box, near[None, :4]) < 0.1

            strong = good[good[:, 4] >= self.strong]
            self.contradicted = self.contradicted + 1 if len(strong) and max_iou(box, strong[:, :4]) < 0.1 else 0

            if self.contradicted > self.patience:  # drifted: a strong detection elsewhere, long enough
                self.active = False
            elif ok and score >= self.lost_thr and not jumped and not missed:
                self.kf.update(box)
                self.bad = 0
                return [(self.tid, tuple(box), score, self.label)]
            elif near is not None:  # re-acquire the same target from the detector
                return self._start(frame, near, new_id=False)
            else:
                self.bad += 1
                if self.bad <= self.patience:
                    return [(self.tid, tuple(pred), 0.0, self.label + " (coast)")]
                self.active = False

        if len(good):
            return self._start(frame, good[good[:, 4].argmax()], new_id=True)
        return []


# ------------------------------------------------------------------------------------------------ runner
def build_trackers(args, fps, names, device):
    out = []
    for t in args.trackers:
        if t == "bytetrack":
            out.append(ByteTrack(fps, names, args.patience))
        elif t == "deepsort":
            out.append(DeepSORT(fps, names, args.conf, args.patience))
        elif t == "sort":
            out.append(SORT(names, args.conf, args.patience))
        else:
            name, _, build, thr = SOT[t]
            out.append(SingleObject(name, build(device), args.conf, thr, args.patience, names))
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


def draw_panel(frame, name, tracks, fps, panel_w):
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
    lines = [f"{name} | FPS {fps:.0f} | objects {len(tracks)}"]  # FPS = total, detector + tracker
    for i, hud in enumerate(lines):
        org = (int(8 * s), int((24 + 22 * i) * s))
        cv2.putText(img, hud, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (0, 0, 0), lw * 3)
        cv2.putText(img, hud, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (255, 255, 255), lw)
    return img


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
    if args.panel_width:
        _, cols = grid_shape(len(trackers), work_w, work_h, args.width, args.height)
        panel_w = args.panel_width
        panel_h = round(panel_w * work_h / work_w)
    else:
        panel_w, panel_h, cols = panel_size(len(trackers), work_w, work_h, args.width, args.height)
    if (work_w, work_h) != (src_w, src_h):
        print(f"[resize] {src_w}x{src_h} -> {work_w}x{work_h} (--max-side {args.max_side})")
    print(f"[run] {name} | trackers: {', '.join(t.name for t in trackers)}")

    stats = {t.name: {"ms": [], "frames": 0, "ids": set(), "ema": None} for t in trackers}
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

            panels = []
            for t in trackers:
                t0 = time.perf_counter()
                tracks = t.update(frame, dets)
                ms = (time.perf_counter() - t0) * 1000
                s = stats[t.name]
                s["ms"].append(ms)
                s["frames"] += bool(tracks)
                s["ids"].update(tid for tid, *_ in tracks)
                s["ema"] = ema(s["ema"], det_ms + ms)  # smooth ms, then invert for FPS
                panels.append(draw_panel(frame, t.name, tracks, 1000 / s["ema"], panel_w))
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
    print(f"\n{name}: {n} frames | detector {det:.1f} ms/frame")
    print(f"{'tracker':<11} {'ms/frame':>9} {'FPS':>7} {'frames w/ track':>16} {'unique IDs':>11}")
    for tname, s in stats.items():
        ms = np.mean(s["ms"][warm])
        print(f"{tname:<11} {ms:>9.2f} {1000 / (det + ms):>7.1f} "
              f"{s['frames']:>9}/{n:<6} {len(s['ids']):>11}")
    print()
    return keep_going


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ai", type=int, default=None, choices=sorted(MODELS),
                    help="номер моделі 1-7 з папки ai/ (якщо не задано і немає --model — запитає цифру)")
    ap.add_argument("--source", default="test_videos", help="video file, directory, or webcam index")
    ap.add_argument("--trackers", nargs="+", default=None, choices=ALL_TRACKERS + list(GROUPS),
                    help="REQUIRED: tracker names and/or groups (mot, classic, filter, deep)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection conf to start tracks / feed DeepSORT")
    ap.add_argument("--det-low", type=float, default=0.1, help="lowest detection conf kept (ByteTrack 2nd stage)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-side", type=int, default=1280,
                    help="downscale frames whose longer side is bigger (aspect kept, no crop); 0 = off")
    ap.add_argument("--no-agnostic", dest="agnostic", action="store_false",
                    help="per-class NMS (default is class-agnostic: the model often puts a quadcopter AND a "
                         "fixed-wing box on the same UAV, which splits tracks)")
    ap.add_argument("--patience", type=int, default=30,
                    help="frames a lost track keeps being drawn at its Kalman prediction (all trackers)")
    ap.add_argument("--max-box", type=float, default=0.1,
                    help="ignore detections larger than this fraction of the frame area; 0 = off")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (auto if omitted)")
    ap.add_argument("--width", type=int, default=1600, help="max width of the side-by-side window")
    ap.add_argument("--height", type=int, default=900, help="max height of the side-by-side window")
    ap.add_argument("--panel-width", type=int, default=0,
                    help="panel width in px (one video cell); 0 = auto fit to --width/--height")
    ap.add_argument("--save", action="store_true", help="save the grid video")
    ap.add_argument("--out", default="runs/track")
    ap.add_argument("--no-show", dest="show", action="store_false", help="disable live window")
    args = ap.parse_args()

    args.trackers = list(dict.fromkeys(x for t in args.trackers for x in GROUPS.get(t, [t]))) if args.trackers \
        else [input(f"Введіть трекер {ALL_TRACKERS}\nабо групу {list(GROUPS)}: ").strip().lower()]
    args.trackers = list(dict.fromkeys(x for t in args.trackers for x in GROUPS.get(t, [t])))
    bad = [t for t in args.trackers if t not in ALL_TRACKERS]
    if bad:
        ap.error(f"невідомий трекер: {bad}. Доступні: {ALL_TRACKERS} + групи {list(GROUPS)}")
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
