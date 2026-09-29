#!/usr/bin/env python3
"""Siamese single-object trackers split for the Hailo-8: template / search / head networks + CPU correlation.

Same code on the workstation (--backend onnx, float) and on the Pi (--backend hailo, HEFs).
models/<tracker>/spec.json (from export_models.py) describes the parts, the correlations and the tracker settings.

    python3 siamtrack.py nanotrack_v2 zr_field_videos --backend hailo --no-video
    python3 siamtrack.py siamfc zr_field_videos/145_vid_1.mp4 --backend onnx
Per clip: <out>/<tracker>/<clip>_track.jsonl  {"fn", "box": [x,y,w,h], "score", "iou"}, summary.json.
Protocol (same as ~/nanotrack/nanotrack_video.py): init on the first GT box, track every later frame,
score IoU with GT on every GT frame after the init frame.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------ backends
class OnnxBackend:
    """Float reference. Inputs/outputs NCHW float32."""

    def __init__(self, mdir, spec):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("SIAM_ORT_THREADS", "0"))  # 0 = all cores
        # hackaton_kpi: idle ORT threads busy-spin on every core after each run, which slowed the other trackers
        # in track.py (run one after another in the same process) by up to 20x
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = {p: ort.InferenceSession(f"{mdir}/{p}.onnx", so, providers=["CPUExecutionProvider"])
                     for p in spec["parts"]}

        self.norm = spec.get("norm")  # LightTrack: ImageNet mean/std (on the Hailo it's in the model script)

    def run(self, part, feeds):
        s = self.sess[part]
        if self.norm and part in ("template", "search", "search_big"):  # LightTrack, SE-SiamFC
            m = np.array(self.norm["mean"], np.float32)[None, :, None, None]
            sd = np.array(self.norm["std"], np.float32)[None, :, None, None]
            feeds = {k: (v - m) / sd for k, v in feeds.items()}
        outs = s.run(None, {k: v.astype(np.float32) for k, v in feeds.items()})
        return {o.name: v for o, v in zip(s.get_outputs(), outs)}

    def close(self):
        pass


class HailoBackend:
    """One HEF per part, sharing one VDevice (scheduler). Inputs/outputs NCHW float32 at the API; HailoRT
    quantizes/dequantizes (FLOAT32 format), the chip sees NHWC. hef/<part>.io.json maps ONNX names to HEF names."""

    def __init__(self, mdir, spec):
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdev = VDevice(params)
        self.parts = {}
        for part in spec["parts"]:
            io = json.load(open(f"{mdir}/hef/{part}.io.json"))
            im = self.vdev.create_infer_model(f"{mdir}/hef/{part}.hef")
            for i in im.inputs:
                i.set_format_type(FormatType.FLOAT32)
            for o in im.outputs:
                o.set_format_type(FormatType.FLOAT32)
            cm = im.configure()
            # fixed buffers: HailoRT maps them for DMA, so they must live as long as the bindings
            ins = {i.name: np.empty(i.shape, np.float32) for i in im.inputs}
            outs = {o.name: np.empty(o.shape, np.float32) for o in im.outputs}
            b = cm.create_bindings(input_buffers=ins, output_buffers=outs)
            self.parts[part] = (im, cm, b, ins, outs, io)

    def run(self, part, feeds):
        im, cm, b, ins, outs, io = self.parts[part]
        for name, arr in feeds.items():
            x = arr[0].transpose(1, 2, 0) if arr.ndim == 4 else arr  # images NCHW -> NHWC; tokens (1, N, C) as-is
            np.copyto(ins[io["inputs"][name]], x.reshape(ins[io["inputs"][name]].shape))
        cm.run([b], 10000)
        res = {}
        for name, hname in io["outputs"].items():
            a = outs[hname]
            res[name] = (a[None] if a.ndim == 3 else a.reshape(1, *a.shape[-3:])).transpose(0, 3, 1, 2)
        return res

    def close(self):
        """Release in order (bindings -> configured models -> device); letting the GC do it can segfault."""
        for part in list(self.parts):
            im, cm, b, ins, outs, io = self.parts.pop(part)
            del b
            if hasattr(cm, "shutdown"):
                cm.shutdown()
            del cm, im
        self.vdev.release()


# ------------------------------------------------------------------------------------------------ correlations (CPU)
def corr_pw(k, s):
    """pixel-wise: (C,h,w) x (C,H,W) -> (h*w, H, W)"""
    c, h, w = k.shape
    _, H, W = s.shape
    return (k.reshape(c, h * w).T @ s.reshape(c, H * W)).reshape(h * w, H, W)


def _windows(s, kh, kw, pads=(0, 0, 0, 0)):
    if any(pads):  # ONNX pads: top, left, bottom, right
        s = np.pad(s, ((0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    return sliding_window_view(s, (kh, kw), axis=(1, 2))  # (C, H', W', kh, kw)


def corr_dw(k, s, pads=(0, 0, 0, 0)):
    """depthwise: (C,kh,kw) x (C,H,W) -> (C,H',W'). cv2.filter2D per channel (it correlates, anchor top-left):
    3.2 ms for 256x29x29 * 5x5 on the Pi 5, vs 10.8 ms for a batched matmul over sliding windows."""
    c, kh, kw = k.shape
    if any(pads):  # ONNX pads: top, left, bottom, right
        s = np.pad(s, ((0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    s = s.astype(np.float32, copy=False)
    k = k.astype(np.float32, copy=False)
    ho, wo = s.shape[1] - kh + 1, s.shape[2] - kw + 1
    out = np.empty((c, ho, wo), np.float32)
    for i in range(c):
        out[i] = cv2.filter2D(s[i], -1, k[i], anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)[:ho, :wo]
    return out


def corr_up(K, s):
    """up-channel / full: (A,C,kh,kw) x (C,H,W) -> (A,H',W')"""
    a, c, kh, kw = K.shape
    win = _windows(s, kh, kw)
    _, ho, wo = win.shape[:3]
    cols = win.transpose(0, 3, 4, 1, 2).reshape(c * kh * kw, ho * wo)
    return (K.reshape(a, -1) @ cols).reshape(a, ho, wo)


def corr_full(k, s):
    """SiamFC full xcorr, (1,C,kh,kw) x (C,H,W) -> (1,H',W'): sum of per-channel cv2.filter2D (3.7 ms for 256x6x6 on
    22x22 on the Pi 5, numpy im2col 5.8 ms, OpenCV DNN conv 6.7 ms)."""
    _, c, kh, kw = k.shape
    ho, wo = s.shape[1] - kh + 1, s.shape[2] - kw + 1
    s = s.astype(np.float32, copy=False)
    k = k.astype(np.float32, copy=False)
    out = np.zeros((ho, wo), np.float32)
    for i in range(c):
        out += cv2.filter2D(s[i], -1, k[0, i], anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)[:ho, :wo]
    return out[None]


def conv2d(x, w, b):
    """plain conv (valid, stride 1) for the DaSiamRPN kernel layers at init: (C,H,W) x (O,C,kh,kw) -> (O,H',W')"""
    return corr_up(w, x) + b[:, None, None]


# ------------------------------------------------------------------------------------------------ trackers
def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    """pysot / NanoTrack crop: square of original_sz around pos, mean-colour padding, resized to model_sz."""
    sz = original_sz
    c = (original_sz + 1) / 2
    xmin = np.floor(pos[0] - c + 0.5)
    xmax = xmin + sz - 1
    ymin = np.floor(pos[1] - c + 0.5)
    ymax = ymin + sz - 1
    r, cc, k = im.shape
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    # Pad only the crop, not the whole frame (the pysot code pads the full 1920x1080 image: ~10 ms on the Pi).
    x0, y0, x1, y1 = max(xmin, 0), max(ymin, 0), min(xmax + 1, cc), min(ymax + 1, r)
    pads = (y0 - ymin, ymax + 1 - y1, x0 - xmin, xmax + 1 - x1)  # top, bottom, left, right
    if x1 <= x0 or y1 <= y0:  # entirely outside the frame
        patch = np.empty((ymax - ymin + 1, xmax - xmin + 1, k), np.uint8)
        patch[:] = avg_chans
    elif any(pads):
        patch = cv2.copyMakeBorder(im[y0:y1, x0:x1], *pads, cv2.BORDER_CONSTANT,
                                   value=[int(v) for v in avg_chans])  # like the uint8 assignment it replaces
    else:
        patch = im[y0:y1, x0:x1]
    if model_sz != original_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch


def blob(img):
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)


class Siamese:
    def __init__(self, spec, backend, mdir):
        self.spec, self.net = spec, backend
        self.t_net = self.t_corr = 0.0
        if spec.get("kernels"):
            self.kern = dict(np.load(f"{mdir}/{spec['kernels']}"))
        # dw / up correlations as one OpenCV DNN conv whose weights are the template kernel (set in _template_feats)
        self.dnn = {}
        if not os.environ.get("SIAM_NO_DNN"):
            for c in spec["corr"]:
                if c.get("dnn"):
                    net = cv2.dnn.readNetFromONNX(f"{mdir}/{c['dnn']}")
                    lid = net.getLayerId(next(n for n in net.getLayerNames() if "xcorr" in n))
                    self.dnn[c["out"]] = (net, lid)

    def _run(self, part, feeds):
        t = time.perf_counter()
        r = self.net.run(part, feeds)
        self.t_net += time.perf_counter() - t
        return r

    def _corr(self, kf, sf):
        t = time.perf_counter()
        out = {}
        for c in self.spec["corr"]:
            k, s = kf[c["k"]], sf[c["s"]][0]
            if c["out"] in self.dnn:
                net = self.dnn[c["out"]][0]
                net.setInput(np.ascontiguousarray(s[None], dtype=np.float32))
                out[c["out"]] = net.forward()
            elif c["type"] == "pw":
                out[c["out"]] = corr_pw(k[0], s)[None]
            elif c["type"] == "full":
                out[c["out"]] = corr_full(k, s)[None]
            elif c["type"] == "dw":
                out[c["out"]] = corr_dw(k[0], s, c.get("pads", (0, 0, 0, 0)))[None]
            elif c["type"] == "up":  # (A,C,kh,kw)
                out[c["out"]] = corr_up(k, s)[None]
        self.t_corr += time.perf_counter() - t
        return out

    def _template_feats(self, z_crop):
        kf = self._run("template", {"z": blob(z_crop)})
        if self.spec["name"] == "dasiamrpn":  # kernel layers on the CPU, once per init
            zf = kf["zf"][0]
            A = self.spec["corr"]
            kr = conv2d(zf, self.kern["w_r1"], self.kern["b_r1"])  # (5120,4,4)
            kc = conv2d(zf, self.kern["w_cls1"], self.kern["b_cls1"])  # (2560,4,4)
            kf = {"k_reg": kr.reshape(A[0]["anchors"], -1, *kr.shape[1:]),
                  "k_cls": kc.reshape(A[1]["anchors"], -1, *kc.shape[1:])}
        for c in self.spec["corr"]:
            if c["out"] in self.dnn:
                net, lid = self.dnn[c["out"]]
                k = kf[c["k"]]
                w = k[0][:, None] if c["type"] == "dw" else k  # dw: (C,1,kh,kw); up: (A,C,kh,kw)
                net.setParam(lid, 0, np.ascontiguousarray(w, dtype=np.float32))
        return kf


class RPNLike(Siamese):
    """NanoTrack (points, ltrb) and SiamRPN / DaSiamRPN (anchors) share the pysot tracking loop."""

    def __init__(self, spec, backend, mdir):
        super().__init__(spec, backend, mdir)
        S = spec["score_size"]
        hann = np.outer(np.hanning(S), np.hanning(S)).flatten()
        if spec["kind"] == "points":
            st = spec["stride"]
            ori = -(S // 2) * st
            x, y = np.meshgrid(ori + st * np.arange(S), ori + st * np.arange(S))
            self.points = np.stack([x.flatten(), y.flatten()], 1).astype(np.float32)
            self.window = hann
        else:
            self.anchors = self._anchors(spec)
            self.window = np.tile(hann, len(spec["ratios"]) * len(spec["scales"]))

    @staticmethod
    def _anchors(spec):
        stride, S = spec["stride"], spec["score_size"]
        base = []
        for r in spec["ratios"]:
            ws = int(np.sqrt(stride * stride / r))
            hs = int(ws * r)
            for s in spec["scales"]:
                base.append([0, 0, ws * s, hs * s])
        base = np.array(base, np.float32)
        A = len(base)
        anchor = np.tile(base, S * S).reshape(-1, 4)
        ori = -(S // 2) * stride
        xx, yy = np.meshgrid(ori + stride * np.arange(S), ori + stride * np.arange(S))
        anchor[:, 0] = np.tile(xx.flatten(), (A, 1)).flatten()
        anchor[:, 1] = np.tile(yy.flatten(), (A, 1)).flatten()
        return anchor

    def init(self, img, box):
        sp = self.spec
        self.center = np.array([box[0] + (box[2] - 1) / 2, box[1] + (box[3] - 1) / 2])
        self.size = np.array([box[2], box[3]], np.float64)
        wz = self.size[0] + sp["context"] * self.size.sum()
        hz = self.size[1] + sp["context"] * self.size.sum()
        s_z = round(np.sqrt(wz * hz))
        self.avg = img.mean(axis=(0, 1))
        z = get_subwindow(img, self.center, sp["exemplar"], s_z, self.avg)
        self.kf = self._template_feats(z)

    def _head(self, sf):
        sp = self.spec
        c = self._corr(self.kf, sf)
        if sp["name"] == "dasiamrpn":
            t = time.perf_counter()
            reg = c["c_reg"][0]
            loc = (self.kern["w_adj"] @ reg.reshape(reg.shape[0], -1) + self.kern["b_adj"][:, None])
            out = {"cls": c["c_cls"], "loc": loc.reshape(1, *reg.shape)}
            self.t_corr += time.perf_counter() - t
            return out
        h = self._run("head", c)
        if sp["kind"] == "points":
            h = {"cls": h["cls"], "loc": np.exp(h["reg"])}
        return h

    @staticmethod
    def _softmax1(cls):
        """(2, ...) logits -> probability of class 1, flattened"""
        cls = cls.reshape(2, -1)
        e = np.exp(cls - cls.max(0))
        return e[1] / e.sum(0)

    def _decode(self, h):
        """head outputs -> score (N,), boxes (4, N) as cx, cy, w, h relative to the search centre"""
        score = self._softmax1(h["cls"][0])
        d = h["loc"][0].reshape(4, -1).astype(np.float64)
        if self.spec["kind"] == "points":
            p = self.points
            x1, y1, x2, y2 = p[:, 0] - d[0], p[:, 1] - d[1], p[:, 0] + d[2], p[:, 1] + d[3]
            return score, np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        a = self.anchors
        return score, np.stack([d[0] * a[:, 2] + a[:, 0], d[1] * a[:, 3] + a[:, 1],
                                np.exp(d[2]) * a[:, 2], np.exp(d[3]) * a[:, 3]])

    def update(self, img):
        sp = self.spec
        wz = self.size[0] + sp["context"] * self.size.sum()
        hz = self.size[1] + sp["context"] * self.size.sum()
        s_z = np.sqrt(wz * hz)
        scale_z = sp["exemplar"] / s_z
        s_x = s_z * sp["instance"] / sp["exemplar"]
        x = get_subwindow(img, self.center, sp["instance"], round(s_x), self.avg)
        sf = self._run("search", {"x": blob(x)})
        score, box = self._decode(self._head(sf))

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, hh):
            pad = (w + hh) * 0.5
            return np.sqrt((w + pad) * (hh + pad))

        s_c = change(sz(box[2], box[3]) / sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / self.size[1]) / (box[2] / box[3]))
        penalty = np.exp(-(r_c * s_c - 1) * sp["penalty_k"])
        pscore = penalty * score
        pscore = pscore * (1 - sp["window_influence"]) + self.window * sp["window_influence"]
        best = int(np.argmax(pscore))
        b = box[:, best] / scale_z
        lr = penalty[best] * score[best] * sp["lr"]
        cx, cy = b[0] + self.center[0], b[1] + self.center[1]
        w = self.size[0] * (1 - lr) + b[2] * lr
        hh = self.size[1] * (1 - lr) + b[3] * lr
        H, W = img.shape[:2]
        cx, cy = max(0, min(cx, W)), max(0, min(cy, H))
        w, hh = max(10, min(w, W)), max(10, min(hh, H))
        self.center, self.size = np.array([cx, cy]), np.array([w, hh])
        return [cx - w / 2, cy - hh / 2, w, hh], float(score[best])


class SiamAPN(RPNLike):
    """SiamAPN++ (vision4robotics; pysot/tracker/adsiamapn_tracker.py): the anchors come from the network's own
    'anchor' map each frame; score = mean of softmax(cls1)*w1, softmax(cls2)*w2 and raw cls3*w3."""

    def __init__(self, spec, backend, mdir):
        Siamese.__init__(self, spec, backend, mdir)
        S, st_ = spec["score_size"], spec["stride"]
        hann = np.hanning(S)
        self.window = np.outer(hann, hann).flatten()
        g = st_ * np.linspace(0, S - 1, S) - st_ * (S - 1) / 2
        self.gx = np.tile(g, S)                     # x fastest
        self.gy = np.repeat(g, S)

    def _decode(self, h):
        sp = self.spec
        shap = h["anchor"][0].astype(np.float64).reshape(4, -1) * sp["anchor_scale"]
        w, hh = shap[0] + shap[1], shap[2] + shap[3]
        ax, ay = self.gx - shap[0] + w / 2, self.gy - shap[2] + hh / 2
        w1, w2, w3 = sp["w"]
        score = (self._softmax1(h["cls1"][0]) * w1 + self._softmax1(h["cls2"][0]) * w2 +
                 h["cls3"][0].reshape(-1).astype(np.float64) * w3) / 3
        d = h["loc"][0].reshape(4, -1).astype(np.float64)
        return score, np.stack([d[0] * w + ax, d[1] * hh + ay, np.exp(d[2]) * w, np.exp(d[3]) * hh])


class SiamFC(Siamese):
    """siamfc-pytorch (huanglianghua) tracker: 3 scales, upsampled response, Hanning window. RGB input."""

    def __init__(self, spec, backend, mdir):
        super().__init__(spec, backend, mdir)
        sp = spec
        self.up = sp["response_up"] * sp["response_sz"]
        h = np.outer(np.hanning(self.up), np.hanning(self.up))
        self.hann = h / h.sum()
        n = sp["scale_num"]
        self.scales = sp["scale_step"] ** np.linspace(-(n // 2), n // 2, n)

    @staticmethod
    def crop(img, center, size, out, border):
        size = round(size)
        c0 = np.round(center - (size - 1) / 2)
        corners = np.round(np.concatenate([c0, c0 + size])).astype(int)
        # siamfc-pytorch pads the whole frame by the largest overshoot; padding only the crop gives the same pixels
        H, W = img.shape[:2]
        y0, x0, y1, x1 = max(corners[0], 0), max(corners[1], 0), min(corners[2], H), min(corners[3], W)
        pads = (y0 - corners[0], corners[2] - y1, x0 - corners[1], corners[3] - x1)  # top, bottom, left, right
        if y1 <= y0 or x1 <= x0:
            patch = np.empty((corners[2] - corners[0], corners[3] - corners[1], 3), np.uint8)
            patch[:] = np.array(border).round()  # copyMakeBorder rounds the fill value
        elif any(pads):
            patch = cv2.copyMakeBorder(img[y0:y1, x0:x1], *pads, cv2.BORDER_CONSTANT, value=border)
        else:
            patch = img[y0:y1, x0:x1]
        return cv2.resize(patch, (out, out), interpolation=cv2.INTER_LINEAR)

    def init(self, img, box):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # siamfc-pytorch works in (y, x, h, w) with 1-based centers
        self.center = np.array([box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 + (box[2] - 1) / 2])
        self.target = np.array([box[3], box[2]], np.float64)
        ctx = sp["context"] * self.target.sum()
        self.z_sz = np.sqrt(np.prod(self.target + ctx))
        self.x_sz = self.z_sz * sp["instance"] / sp["exemplar"]
        self.avg = tuple(float(v) for v in img.mean(axis=(0, 1)))
        z = self.crop(img, self.center, self.z_sz, sp["exemplar"], self.avg)
        self.kf = self._template_feats(z)

    def update(self, img):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resp = []
        for f in self.scales:
            x = self.crop(img, self.center, self.x_sz * f, sp["instance"], self.avg)
            sf = self._run("search", {"x": blob(x)})
            resp.append(self._corr(self.kf, sf)["c0"][0, 0] * sp["out_scale"])
        t = time.perf_counter()
        resp = np.stack([cv2.resize(r, (self.up, self.up), interpolation=cv2.INTER_CUBIC) for r in resp])
        resp[:sp["scale_num"] // 2] *= sp["scale_penalty"]
        resp[sp["scale_num"] // 2 + 1:] *= sp["scale_penalty"]
        sid = int(np.argmax(resp.max(axis=(1, 2))))
        r = resp[sid] - resp[sid].min()
        r /= r.sum() + 1e-16
        r = (1 - sp["window_influence"]) * r + sp["window_influence"] * self.hann
        loc = np.array(np.unravel_index(r.argmax(), r.shape), np.float64)
        disp = (loc - (self.up - 1) / 2) * sp["stride"] / sp["response_up"]
        disp = disp * self.x_sz * self.scales[sid] / sp["instance"]
        self.center += disp
        scale = (1 - sp["scale_lr"]) + sp["scale_lr"] * self.scales[sid]
        self.target *= scale
        self.z_sz *= scale
        self.x_sz *= scale
        self.t_corr += time.perf_counter() - t
        box = [self.center[1] + 1 - (self.target[1] - 1) / 2, self.center[0] + 1 - (self.target[0] - 1) / 2,
               self.target[1], self.target[0]]
        return box, float(resp[sid].max())


class LightTrack(Siamese):
    """researchmm/LightTrack lib/tracker/lighttrack.py (Ocean-style, anchor-free). RGB input. The search size is
    chosen at init: 288 (search_big/head_big) when the target covers < 0.4 % of the frame, else 256."""

    def init(self, img, box):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], np.float64)  # get_axis_aligned_bbox
        self.size = np.array([box[2], box[3]], np.float64)
        big = box[2] * box[3] / float(img.shape[0] * img.shape[1]) < sp["big_area_ratio"]
        self.inst = sp["instance_big"] if big else sp["instance"]
        self.parts = ("search_big", "head_big") if big else ("search", "head")
        S = int(round(self.inst / sp["stride"]))
        self.window = np.outer(np.hanning(S), np.hanning(S))
        g = (np.arange(S) - np.floor(float(S // 2))) * sp["stride"] + self.inst // 2
        self.gx, self.gy = np.meshgrid(g, g)
        wz = self.size[0] + sp["context"] * self.size.sum()
        hz = self.size[1] + sp["context"] * self.size.sum()
        self.avg = img.mean(axis=(0, 1))
        z = get_subwindow(img, self.center, sp["exemplar"], round(np.sqrt(wz * hz)), self.avg)
        self.kf = self._template_feats(z)

    def update(self, img):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wz = self.size[0] + sp["context"] * self.size.sum()
        hz = self.size[1] + sp["context"] * self.size.sum()
        s_z = np.sqrt(wz * hz)
        scale_z = sp["exemplar"] / s_z
        s_x = s_z + 2 * ((self.inst - sp["exemplar"]) / 2) / scale_z
        s_x = s_x + 0.5 if round(s_x + 1) - round(s_x) != 1 else round(s_x)  # python2round
        x = get_subwindow(img, self.center, self.inst, int(round(s_x)), self.avg)
        sf = self._run(self.parts[0], {"x": blob(x)})
        h = self._run(self.parts[1], self._corr(self.kf, sf))
        score = 1 / (1 + np.exp(-h["cls"][0, 0].astype(np.float64)))
        reg = h["reg"][0].astype(np.float64)
        x1, y1 = self.gx - reg[0], self.gy - reg[1]
        x2, y2 = self.gx + reg[2], self.gy + reg[3]
        t = self.size * scale_z

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, hh):
            pad = (w + hh) * 0.5
            return np.sqrt((w + pad) * (hh + pad))

        s_c = change(sz(x2 - x1, y2 - y1) / sz(t[0], t[1]))
        r_c = change((t[0] / t[1]) / ((x2 - x1) / (y2 - y1)))
        penalty = np.exp(-(r_c * s_c - 1) * sp["penalty_k"])
        pscore = penalty * score
        pscore = pscore * (1 - sp["window_influence"]) + self.window * sp["window_influence"]
        r, c = np.unravel_index(pscore.argmax(), pscore.shape)
        dx = ((x1[r, c] + x2[r, c]) / 2 - self.inst // 2) / scale_z
        dy = ((y1[r, c] + y2[r, c]) / 2 - self.inst // 2) / scale_z
        pw, ph = (x2[r, c] - x1[r, c]) / scale_z, (y2[r, c] - y1[r, c]) / scale_z
        lr = penalty[r, c] * score[r, c] * sp["lr"]
        res = np.array([pw * lr + (1 - lr) * self.size[0], ph * lr + (1 - lr) * self.size[1]])
        size = self.size * (1 - lr) + lr * res  # the original smooths twice
        H, W = img.shape[:2]
        self.center = np.array([max(0, min(W, self.center[0] + dx)), max(0, min(H, self.center[1] + dy))])
        self.size = np.array([max(10, min(W, size[0])), max(10, min(H, size[1]))])
        return [max(0., self.center[0] - self.size[0] / 2), max(0., self.center[1] - self.size[1] / 2),
                self.size[0], self.size[1]], float(score[r, c])


class SESiamFC(Siamese):
    """ISosnovik/SiamSE lib/tracker.py SESiamFCTracker (SiamDW-style SiamFC, 3 scales). RGB, ImageNet normalization
    (model script on the Hailo). The backbone's max projection over its internal scales is inside the HEF."""

    def __init__(self, spec, backend, mdir):
        super().__init__(spec, backend, mdir)
        up = spec["response_sz"] * spec["response_up"]
        w = np.outer(np.hanning(up), np.hanning(up))
        self.window = w / w.sum()
        n = spec["scale_num"]
        self.scales = spec["scale_step"] ** (np.arange(n) - np.ceil(n // 2))

    def init(self, img, box):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], np.float64)  # get_axis_aligned_bbox
        self.size = np.array([box[2], box[3]], np.float64)
        self.avg = img.mean(axis=(0, 1))
        wz = self.size[0] + sp["context"] * self.size.sum()
        hz = self.size[1] + sp["context"] * self.size.sum()
        s_z = round(np.sqrt(wz * hz))
        self.s_x = s_z + 2 * ((sp["instance"] - sp["exemplar"]) / 2) / (sp["exemplar"] / s_z)
        self.min_s_x, self.max_s_x = 0.2 * self.s_x, 5 * self.s_x
        self.kf = self._template_feats(get_subwindow(img, self.center, sp["exemplar"], s_z, self.avg))

    def update(self, img):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inst = self.s_x * self.scales
        resp = []
        for s in inst:
            x = get_subwindow(img, self.center, sp["instance"], int(round(s)), self.avg)
            sf = self._run("search", {"x": blob(x)})
            resp.append(self._corr(self.kf, sf)["c0"][0, 0] * sp["out_scale"])
        t = time.perf_counter()
        up = sp["response_sz"] * sp["response_up"]
        resp = np.stack([cv2.resize(r, (up, up), interpolation=cv2.INTER_CUBIC) for r in resp])
        n = sp["scale_num"]
        best = int(np.argmax(resp.max(axis=(1, 2)) * sp["scale_penalty"] ** np.abs(np.arange(n) - n // 2)))
        r = resp[best] - resp[best].min()
        r = r / r.sum()
        r = (1 - sp["window_influence"]) * r + sp["window_influence"] * self.window
        ry, rx = np.unravel_index(r.argmax(), r.shape)
        disp = (np.array([rx, ry]) - np.ceil(up / 2)) * sp["stride"] / sp["response_up"] * self.s_x / sp["instance"]
        lr = sp["scale_lr"]
        self.s_x = max(self.min_s_x, min(self.max_s_x, (1 - lr) * self.s_x + lr * inst[best]))
        size = (1 - lr) * self.size + lr * self.size * self.scales[best]
        H, W = img.shape[:2]
        c = self.center + disp
        self.center = np.array([max(0, min(W, c[0])), max(0, min(H, c[1]))])
        self.size = np.array([max(10, min(W, size[0])), max(10, min(H, size[1]))])
        self.t_corr += time.perf_counter() - t
        return [max(0., self.center[0] - self.size[0] / 2), max(0., self.center[1] - self.size[1] / 2),
                self.size[0], self.size[1]], float(resp[best].max())


class FEAR(Siamese):
    """PinataFarms/FEARTracker model_training/tracker/fear_tracker.py with the default siam_tracker.yaml (no
    smoothing: plain argmax of the sigmoid score). Non-square context crops (box extended by 20 % / 200 % per side,
    resized to 128 / 256), RGB, ImageNet normalization. The head takes the search encodings and the correlations."""

    def __init__(self, spec, backend, mdir):
        super().__init__(spec, backend, mdir)
        S = spec["score_size"]
        g = (np.arange(S) - np.floor(float(S // 2))) * spec["stride"] + spec["instance"] // 2
        self.gx, self.gy = np.meshgrid(g, g)

    @staticmethod
    def _clamp(b, shape, min_side=3):  # model_training/utils/utils.py clamp_bbox + ensure_bbox_boundaries
        x1, y1 = min(max(0, b[0]), shape[1]), min(max(0, b[1]), shape[0])
        x2, y2 = min(max(0, b[0] + b[2]), shape[1]), min(max(0, b[1] + b[3]), shape[0])
        x, y, w, h = np.array([x1, y1, x2 - x1, y2 - y1]).astype("int32")
        if w < min_side:
            w = min_side
            x -= max(0, x + w - shape[1])
        if h < min_side:
            h = min_side
            y -= max(0, y + h - shape[0])
        return np.array([x, y, w, h])

    @staticmethod
    def _crop(img, b, size, off, pad_value):
        """get_extended_crop: context = box extended by off*w / off*h per side (int32), padded, resized to size."""
        x, y, w, h = b
        c = np.array([x - w * off, y - h * off, w * (1.0 + 2 * off), h * (1.0 + 2 * off)]).astype("int32")
        pl, pt = max(-c[0], 0), max(-c[1], 0)
        pr, pb = max(c[0] + c[2] - img.shape[1], 0), max(c[1] + c[3] - img.shape[0], 0)
        crop = img[c[1] + pt:c[1] + c[3] - pb, c[0] + pl:c[0] + c[2] - pr]
        crop = cv2.copyMakeBorder(crop, int(pt), int(pb), int(pl), int(pr), cv2.BORDER_CONSTANT,
                                  value=[float(v) for v in pad_value])
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR), c

    def init(self, img, box):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.box = self._clamp(np.array(box, np.float64), img.shape)
        self.avg = img.mean(axis=(0, 1))
        z, _ = self._crop(img, self.box, sp["exemplar"], sp["template_offset"], self.avg)
        self.kf = self._template_feats(z)

    def update(self, img):
        sp = self.spec
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, c = self._crop(img, self.box, sp["instance"], sp["search_offset"], self.avg)
        sf = self._run("search", {"x": blob(x)})
        feeds = dict(sf)
        feeds.update(self._corr(self.kf, sf))
        h = self._run("head", feeds)
        score = 1 / (1 + np.exp(-h["cls"][0, 0].astype(np.float64)))
        reg = np.exp(h["reg"][0].astype(np.float64))
        r, cc = np.unravel_index(score.argmax(), score.shape)
        x1, y1 = self.gx[r, cc] - reg[0, r, cc], self.gy[r, cc] - reg[1, r, cc]
        x2, y2 = self.gx[r, cc] + reg[2, r, cc], self.gy[r, cc] + reg[3, r, cc]
        ws, hs = c[2] / sp["instance"], c[3] / sp["instance"]  # _rescale_bbox
        b = [round(x1 * ws + c[0]), round(y1 * hs + c[1]), max(3, round((x2 - x1) * ws)), max(3, round((y2 - y1) * hs))]
        self.box = self._clamp(np.array(b, np.float64), img.shape)
        return [float(v) for v in self.box], float(score[r, cc])


class VitTrack(Siamese):
    """OpenCV TrackerVit (modules/video/src/tracking/tracker_vit.cpp): one-stream ViT, ONE network per frame with the
    template crop and the search crop as inputs (no CPU correlation). BGR 0-255 in; the ImageNet normalization is
    inside models/vittrack/model.onnx (-> on-chip). Score x centered Hann window, argmax; below 0.2 keep the box."""

    def __init__(self, spec, backend, mdir):
        super().__init__(spec, backend, mdir)
        S = spec["score_size"]
        h = 0.5 * (1 - np.cos(2 * np.pi / (S + 1) * (np.arange(S) + 1)))
        self.window = np.outer(h, h)

    @staticmethod
    def crop(img, box, factor):
        x, y, w, h = [int(v) for v in box]
        sz = int(np.ceil(np.sqrt(w * h) * factor))
        x1, y1 = x + int((w - sz) / 2), y + int((h - sz) / 2)  # C++ integer division truncates toward zero
        x2, y2 = x1 + sz, y1 + sz
        x1p, y1p = max(0, -x1), max(0, -y1)
        x2p, y2p = max(x2 - img.shape[1] + 1, 0), max(y2 - img.shape[0] + 1, 0)
        roi = img[y1 + y1p:y2 - y2p, x1 + x1p:x2 - x2p]
        return cv2.copyMakeBorder(roi, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT, value=0), sz

    def init(self, img, box):
        sp = self.spec
        z, _ = self.crop(img, box, sp["template_factor"])
        self.z = blob(cv2.resize(z, (sp["exemplar"], sp["exemplar"])))
        self.rect = [int(v) for v in box]

    def update(self, img):
        sp = self.spec
        x, sz = self.crop(img, self.rect, sp["search_factor"])
        o = self._run("model", {"template": self.z, "search": blob(cv2.resize(x, (sp["instance"], sp["instance"])))})
        S = sp["score_size"]
        conf = o["output1"].reshape(S, S) * self.window
        size, off = o["output2"].reshape(2, S, S), o["output3"].reshape(2, S, S)
        my, mx = np.unravel_index(conf.argmax(), conf.shape)
        score = float(conf[my, mx])
        if score >= sp["score_threshold"]:
            cx, cy = (mx + off[0, my, mx]) / S, (my + off[1, my, mx]) / S
            w, h = size[0, my, mx], size[1, my, mx]
            r = self.rect
            x0, y0 = r[0] + int((r[2] - sz) / 2), r[1] + int((r[3] - sz) / 2)
            self.rect = [int(np.floor((cx - w / 2) * sz + x0)), int(np.floor((cy - h / 2) * sz + y0)),
                         int(np.floor(w * sz)), int(np.floor(h * sz))]
        return [float(v) for v in self.rect], score


class HiT(Siamese):
    """kangben258/HiT lib/test/tracker/HiT.py (STARK-style): template (x2, 128) and search (x4, 256) square crops
    with zero padding, RGB, ONE network per frame (normalization inside). The two 16x16 corner maps come from the
    Hailo; soft-argmax (softmax over 256 cells, expected coordinate) on the CPU; box mapped back, clip_box margin 10."""

    def __init__(self, spec, backend, mdir):
        super().__init__(spec, backend, mdir)
        fs, st_ = spec["feat_sz"], spec["stride"]
        g = (np.arange(fs) + spec.get("coord_offset", 0.0)) * st_
        self.cx, self.cy = np.tile(g, fs), np.repeat(g, fs)  # x fastest, as indice.repeat in the head
        self.img_sz = fs * st_

    @staticmethod
    def sample_target(img, bb, factor, out_sz):
        x, y, w, h = bb
        sz = int(np.ceil(np.sqrt(w * h) * factor))
        x1 = round(x + 0.5 * w - sz * 0.5)
        y1 = round(y + 0.5 * h - sz * 0.5)
        x2, y2 = x1 + sz, y1 + sz
        x1p, x2p = max(0, -x1), max(x2 - img.shape[1] + 1, 0)
        y1p, y2p = max(0, -y1), max(y2 - img.shape[0] + 1, 0)
        crop = img[y1 + y1p:y2 - y2p, x1 + x1p:x2 - x2p]
        crop = cv2.copyMakeBorder(crop, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT)
        return cv2.resize(crop, (out_sz, out_sz)), out_sz / sz

    def init(self, img, box):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        z, _ = self.sample_target(rgb, box, sp["template_factor"], sp["exemplar"])
        self.z = blob(z)
        self.state = [float(v) for v in box]

    def update(self, img):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, rf = self.sample_target(rgb, self.state, sp["search_factor"], sp["instance"])
        maps = self._run("model", {"search": blob(x), "template": self.z})["maps"][0].reshape(2, -1).astype(np.float64)
        e = np.exp(maps - maps.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        x1, y1, x2, y2 = (p[0] @ self.cx, p[0] @ self.cy, p[1] @ self.cx, p[1] @ self.cy)
        b = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]) / self.img_sz * sp["instance"] / rf
        cxp, cyp = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        half = 0.5 * sp["instance"] / rf
        cx, cy = b[0] + cxp - half, b[1] + cyp - half
        bx = [cx - 0.5 * b[2], cy - 0.5 * b[3], b[2], b[3]]
        H, W, m = img.shape[0], img.shape[1], 10  # clip_box(margin=10)
        x1, y1 = min(max(0, bx[0]), W - m), min(max(0, bx[1]), H - m)
        x2, y2 = min(max(m, bx[0] + bx[2]), W), min(max(m, bx[1] + bx[3]), H)
        self.state = [x1, y1, max(m, x2 - x1), max(m, y2 - y1)]
        return list(self.state), float(max(p[0].max(), p[1].max()))


class MixFormerV2(HiT):
    """MCG-NJU/MixFormerV2 lib/test/tracker/mixformer2_vit_online.py with one online template: STARK crops
    (template x2 -> 112, search x4.5 -> 224), RGB, ONE network (normalization inside) with template, online template,
    search and the 4 constant box tokens as inputs. Box = expected value of the 96-bin distributions (CPU). The best
    template crop with score > 0.5 becomes the online template every update_interval frames."""

    def __init__(self, spec, backend, mdir):
        Siamese.__init__(self, spec, backend, mdir)
        self.reg = np.load(f"{mdir}/reg_tokens.npy")
        self.indice = np.arange(spec["bins"]) * spec["bin_stride"]

    def init(self, img, box):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        z, _ = self.sample_target(rgb, box, sp["template_factor"], sp["exemplar"])
        self.t = self.ot = self.ot_max = blob(z)
        self.state = [float(v) for v in box]
        self.max_score, self.frame = -1.0, 0

    def update(self, img):
        sp = self.spec
        self.frame += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, rf = self.sample_target(rgb, self.state, sp["search_factor"], sp["instance"])
        o = self._run("model", {"template": self.t, "online_template": self.ot, "search": blob(x), "reg_tokens": self.reg})
        lg = o["box_logits"].reshape(4, -1).astype(np.float64)  # rows: l, r, t, b
        e = np.exp(lg - lg.max(1, keepdims=True))
        l, r, t, b = (e / e.sum(1, keepdims=True)) @ self.indice / sp["img_sz"]
        bx = np.array([(l + r) / 2, (t + b) / 2, r - l, b - t]) * sp["instance"] / rf
        cxp, cyp = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        half = 0.5 * sp["instance"] / rf
        cx, cy = bx[0] + cxp - half, bx[1] + cyp - half
        bb = [cx - 0.5 * bx[2], cy - 0.5 * bx[3], bx[2], bx[3]]
        H, W, m = img.shape[0], img.shape[1], 10  # clip_box(margin=10)
        x1, y1 = min(max(0, bb[0]), W - m), min(max(0, bb[1]), H - m)
        x2, y2 = min(max(m, bb[0] + bb[2]), W), min(max(m, bb[1] + bb[3]), H)
        self.state = [x1, y1, max(m, x2 - x1), max(m, y2 - y1)]
        score = float(1 / (1 + np.exp(-float(o["score"].reshape(-1)[0]))))
        if score > sp["score_threshold"] and score > self.max_score:  # max_score_decay = 1.0
            z, _ = self.sample_target(rgb, self.state, sp["template_factor"], sp["exemplar"])
            self.ot_max, self.max_score = blob(z), score
        if self.frame % sp["update_interval"] == 0:
            self.ot, self.max_score, self.ot_max = self.ot_max, -1.0, self.t
        return list(self.state), score


class AVTrack(HiT):
    """wuyou3474/AVTrack lib/test/tracker/avtrack.py (OSTrack CENTER head): STARK crops (x2 -> 128, x4 -> 256), RGB,
    ONE network (normalization inside, fixed block pattern, see avtrack_export.py). score x centered Hann window,
    argmax, box = ((ix + off_x) / fs, (iy + off_y) / fs, size_w, size_h); map back, clip_box margin 10."""

    def __init__(self, spec, backend, mdir):
        Siamese.__init__(self, spec, backend, mdir)
        fs = spec["feat_sz"]
        h = 0.5 * (1 - np.cos(2 * np.pi / (fs + 1) * (np.arange(fs) + 1)))
        self.window = np.outer(h, h)

    def init(self, img, box):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        z, _ = self.sample_target(rgb, box, sp["template_factor"], sp["exemplar"])
        self.z = blob(z)
        self.state = [float(v) for v in box]

    def update(self, img):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, rf = self.sample_target(rgb, self.state, sp["search_factor"], sp["instance"])
        o = self._run("model", {"template": self.z, "search": blob(x)})
        return self._center_box(o, rf, img, margin=10)

    def _center_box(self, o, rf, img, margin):
        """OSTrack CENTER head: Hann-windowed score argmax, offset + size at that cell, map back, clip_box."""
        sp = self.spec
        fs = sp["feat_sz"]
        score = o["score"].reshape(fs, fs) * self.window
        size, off = o["size"].reshape(2, fs, fs), o["offset"].reshape(2, fs, fs)
        iy, ix = np.unravel_index(score.argmax(), score.shape)
        b = np.array([(ix + off[0, iy, ix]) / fs, (iy + off[1, iy, ix]) / fs, size[0, iy, ix], size[1, iy, ix]])
        b = b * sp["instance"] / rf
        cxp, cyp = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        half = 0.5 * sp["instance"] / rf
        cx, cy = b[0] + cxp - half, b[1] + cyp - half
        bb = [cx - 0.5 * b[2], cy - 0.5 * b[3], b[2], b[3]]
        H, W, m = img.shape[0], img.shape[1], margin
        x1, y1 = min(max(0, bb[0]), W - m), min(max(0, bb[1]), H - m)
        x2, y2 = min(max(m, bb[0] + bb[2]), W), min(max(m, bb[1] + bb[3]), H)
        self.state = [x1, y1, max(m, x2 - x1), max(m, y2 - y1)]
        return list(self.state), float(score[iy, ix])


class LightFC(AVTrack):
    """LiYunfengLYF/LightFC lib/test/tracker/lightfc.py: STARK crops (x2 -> 128, x4 -> 256), RGB, split like NanoTrack:
    MobileNetV2 backbone on the template (once) and the search crop, pixel-wise correlation on the CPU, fusion + center
    head as one network. Same CENTER-head decode as OSTrack/AVTrack, clip_box margin 2."""

    def init(self, img, box):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        z, _ = self.sample_target(rgb, box, sp["template_factor"], sp["exemplar"])
        self.kf = self._template_feats(z)
        self.state = [float(v) for v in box]

    def update(self, img):
        sp = self.spec
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, rf = self.sample_target(rgb, self.state, sp["search_factor"], sp["instance"])
        sf = self._run("search", {"x": blob(x)})
        o = self._run("head", {"s0": sf["s0"], **self._corr(self.kf, sf)})
        return self._center_box(o, rf, img, margin=2)


def make_tracker(name, backend="onnx", models=f"{HERE}/models"):
    mdir = f"{models}/{name}"
    spec = json.load(open(f"{mdir}/spec.json"))
    be = (HailoBackend if backend == "hailo" else OnnxBackend)(mdir, spec)
    cls = {"siamfc": SiamFC, "lighttrack": LightTrack, "apn": SiamAPN, "sesiamfc": SESiamFC,
           "fear": FEAR, "vit": VitTrack, "hit": HiT, "mixformer": MixFormerV2,
           "avtrack": AVTrack, "lightfc": LightFC}.get(spec["kind"], RPNLike)
    return cls(spec, be, mdir)


# ------------------------------------------------------------------------------------------------ video runner
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    return iw * ih / (aw * ah + bw * bh - iw * ih + 1e-9)


def load_gt(video):
    base = os.path.splitext(video)[0]
    for f in (f"{base}_gt.jsonl", f"{base}_gt_partial.jsonl"):
        if os.path.exists(f):
            gt = {}
            for line in open(f):
                r = json.loads(line)
                if r["box"]:
                    b = r["box"]
                    gt[r["fn"]] = (b["x"], b["y"], b["w"], b["h"])
            return gt
    return {}


def run_video(path, tracker, args, out_dir):
    gt = load_gt(path)
    if not gt:
        print(f"skip {path}: no GT")
        return None
    start = min(gt)
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    log = open(f"{out_dir}/{name}_track.jsonl", "w")
    ious, t_upd = [], []
    tracker.t_net = tracker.t_corr = 0.0
    fn = -1
    while True:
        ok, img = cap.read()
        fn += 1
        if not ok or (args.max_frames and fn > start + args.max_frames):
            break
        # Pi's FFmpeg leaves garbage in the last row of 1080p H.264 frames (coded height 1088); results would
        # depend on memory layout. Replace it with the row above (harmless elsewhere).
        img[-1] = img[-2]
        if fn < start:
            continue
        if fn == start:
            t0 = time.perf_counter()
            tracker.init(img, gt[start])
            t_init = time.perf_counter() - t0
            tracker.t_net = tracker.t_corr = 0.0  # count per-frame work only
            box, score = gt[start], 1.0
        else:
            t0 = time.perf_counter()
            box, score = tracker.update(img)
            t_upd.append(time.perf_counter() - t0)
        rec = {"fn": fn, "box": [round(float(v), 1) for v in box], "score": round(score, 3)}
        if fn in gt and fn != start:
            rec["iou"] = round(iou(box, gt[fn]), 3)
            ious.append(rec["iou"])
        log.write(json.dumps(rec) + "\n")
    log.close()
    n = max(len(t_upd), 1)
    st = dict(frames=len(t_upd), gt_frames=len(ious), iou_sum=float(np.sum(ious)), succ=int(np.sum(np.array(ious) > 0.5)),
              ms=1000 * float(np.sum(t_upd)) / n, ms_net=1000 * tracker.t_net / n, ms_corr=1000 * tracker.t_corr / n,
              ms_init=1000 * t_init)
    print(f"{args.tracker:18s} {name:18s} frames {st['frames']:4d}  {st['ms']:5.1f} ms/frame (net {st['ms_net']:5.1f}, "
          f"corr {st['ms_corr']:4.1f})  mIoU {st['iou_sum'] / max(st['gt_frames'], 1):.3f}  "
          f"IoU>0.5 {st['succ'] / max(st['gt_frames'], 1):.2f}", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracker")
    ap.add_argument("input", help="video file or folder of videos (GT from <name>_gt*.jsonl)")
    ap.add_argument("--backend", choices=["onnx", "hailo"], default="onnx")
    ap.add_argument("--models", default=f"{HERE}/models")
    ap.add_argument("--out", default="track_out")
    ap.add_argument("--max-frames", type=int, default=0, help="quick test: frames after init per clip")
    args = ap.parse_args()

    tracker = make_tracker(args.tracker, args.backend, args.models)
    out_dir = f"{args.out}/{args.tracker}"
    os.makedirs(out_dir, exist_ok=True)
    vids = sorted(glob.glob(f"{args.input}/*.mp4")) if os.path.isdir(args.input) else [args.input]
    res = {}
    try:
        for v in vids:
            st = run_video(v, tracker, args, out_dir)
            if st:
                res[os.path.splitext(os.path.basename(v))[0]] = st
    finally:
        tracker.net.close()
    if res:
        f = sum(s["frames"] for s in res.values())
        g = sum(s["gt_frames"] for s in res.values())
        tot = dict(frames=f, gt_frames=g,
                   miou=sum(s["iou_sum"] for s in res.values()) / max(g, 1),
                   success=sum(s["succ"] for s in res.values()) / max(g, 1),
                   clip_mean_success=float(np.mean([s["succ"] / max(s["gt_frames"], 1) for s in res.values()])),
                   ms=sum(s["ms"] * s["frames"] for s in res.values()) / max(f, 1),
                   ms_net=sum(s["ms_net"] * s["frames"] for s in res.values()) / max(f, 1),
                   ms_corr=sum(s["ms_corr"] * s["frames"] for s in res.values()) / max(f, 1))
        tot["fps"] = 1000 / tot["ms"]
        print(f"{args.tracker:18s} {'TOTAL':18s} frames {f:4d}  {tot['ms']:5.1f} ms/frame ({tot['fps']:.0f} FPS, net "
              f"{tot['ms_net']:.1f}, corr {tot['ms_corr']:.1f})  mIoU {tot['miou']:.3f}  IoU>0.5 {tot['success']:.3f} "
              f"(clip mean {tot['clip_mean_success']:.3f})")
        json.dump({"tracker": args.tracker, "backend": args.backend, "clips": res, "total": tot},
                  open(f"{out_dir}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
