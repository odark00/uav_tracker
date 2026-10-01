#!/usr/bin/env python3
"""
uav_metrics.py - evaluation of detection/tracking output against ground truth.

Computes, from two CSV files (ground truth and tracker output) in the same format:

    frame,id,x1,y1,x2,y2[,...extra columns are ignored]

frame numbers are 1-based, boxes are in pixels. Frames without a row simply have no object.

Metrics
-------
Per-frame localisation : IoU, centre error (px and relative to target size)
Single-target curves   : success plot + AUC, precision plot + P@d
Tracking (MOT)         : MOTA, IDSW, IDF1 / IDP / IDR, HOTA / DetA / AssA
Task-specific          : loss intervals, reacquisition time, whether the ID was
                         kept after a loss, compliance with a maximum recovery time

Everything is implemented with numpy/pandas/scipy only (no tracking-eval dependency).
The MOT metrics follow the reference definitions (CLEAR-MOT, ID measures, TrackEval HOTA).

Usage
-----
    python uav_metrics.py --gt gt.csv --pred pred.csv --fps 25 --out results/ --plot
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REQUIRED_COLUMNS = ("frame", "id", "x1", "y1", "x2", "y2")
_BIG = 1e6  # cost of a forbidden assignment


# --------------------------------------------------------------------------- #
# Data loading and geometry
# --------------------------------------------------------------------------- #
def load_tracks(path: str | Path) -> pd.DataFrame:
    """Read a track CSV and validate its columns."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    df = df[list(REQUIRED_COLUMNS)].copy()
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    if df.duplicated(["frame", "id"]).any():
        raise ValueError(f"{path}: duplicate (frame, id) rows")
    return df.sort_values(["frame", "id"]).reset_index(drop=True)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (n,4) and b (m,4), both as x1,y1,x2,y2."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


@dataclass
class FrameData:
    """GT and tracker objects of one frame, with their pairwise IoU."""
    frame: int
    gt_ids: np.ndarray
    gt_boxes: np.ndarray
    tr_ids: np.ndarray
    tr_boxes: np.ndarray
    iou: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.iou = iou_matrix(self.gt_boxes, self.tr_boxes)


def build_frames(gt: pd.DataFrame, pred: pd.DataFrame) -> list[FrameData]:
    """Align GT and predictions frame by frame over the union of frame numbers."""
    g = {f: d for f, d in gt.groupby("frame")}
    p = {f: d for f, d in pred.groupby("frame")}
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    out = []
    for f in sorted(set(g) | set(p)):
        gd, pd_ = g.get(f, empty), p.get(f, empty)
        out.append(FrameData(
            f,
            gd["id"].to_numpy(int), gd[["x1", "y1", "x2", "y2"]].to_numpy(float),
            pd_["id"].to_numpy(int), pd_[["x1", "y1", "x2", "y2"]].to_numpy(float),
        ))
    return out


def _centres(boxes: np.ndarray) -> np.ndarray:
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]


# --------------------------------------------------------------------------- #
# Per-frame localisation, success / precision curves
# --------------------------------------------------------------------------- #
def per_gt_localisation(frames: list[FrameData]) -> pd.DataFrame:
    """
    For every GT object in every frame, take the tracker box with the highest IoU
    (ties, e.g. no overlap at all, are broken by the nearest centre) and report
    IoU, centre error in px and centre error normalised by sqrt(w*h) of the GT box.
    A frame without any tracker box yields IoU = 0 and an infinite error.
    """
    rows = []
    for fd in frames:
        gc, tc = _centres(fd.gt_boxes), _centres(fd.tr_boxes)
        for i, gid in enumerate(fd.gt_ids):
            w = fd.gt_boxes[i, 2] - fd.gt_boxes[i, 0]
            h = fd.gt_boxes[i, 3] - fd.gt_boxes[i, 1]
            if len(fd.tr_ids) == 0:
                rows.append((fd.frame, gid, -1, 0.0, np.inf, np.inf))
                continue
            dist = np.hypot(*(tc - gc[i]).T)
            j = min(range(len(dist)), key=lambda k: (-fd.iou[i, k], dist[k]))
            rows.append((fd.frame, gid, fd.tr_ids[j], fd.iou[i, j], dist[j], dist[j] / np.sqrt(w * h)))
    return pd.DataFrame(rows, columns=["frame", "gt_id", "tr_id", "iou", "centre_err_px", "centre_err_norm"])


def success_auc(iou: np.ndarray, n_thr: int = 101) -> tuple[np.ndarray, np.ndarray, float]:
    """Success plot S(t) = share of frames with IoU > t and its area (AUC)."""
    thr = np.linspace(0, 1, n_thr)
    s = np.array([(iou > t).mean() for t in thr])
    return thr, s, float(np.trapezoid(s, thr))


def precision_at(err_px: np.ndarray, d: float = 20.0) -> float:
    """Share of frames whose centre error is at most d pixels."""
    return float((err_px <= d).mean())


# --------------------------------------------------------------------------- #
# CLEAR-MOT: MOTA, IDSW, recall, precision
# --------------------------------------------------------------------------- #
def clear_mot(frames: list[FrameData], thr: float = 0.5) -> dict:
    """
    CLEAR-MOT (Bernardin & Stiefelhagen 2008) with an IoU >= thr match criterion.

    Per frame, correspondences from the previous match are kept first (continuity),
    the remaining objects are matched with the Hungarian algorithm.
    IDSW is counted when a GT object is matched to a different tracker ID than
    the one it was last matched to.
        MOTA = 1 - (FN + FP + IDSW) / num_gt
    """
    tp = fp = fn = idsw = n_gt = 0
    last: dict[int, int] = {}  # gt id -> tracker id of its most recent match
    for fd in frames:
        n_gt += len(fd.gt_ids)
        pairs: list[tuple[int, int]] = []
        used_g: set[int] = set()
        used_t: set[int] = set()
        t_index = {tid: j for j, tid in enumerate(fd.tr_ids)}
        for i, gid in enumerate(fd.gt_ids):
            j = t_index.get(last.get(gid, -1))
            if j is not None and fd.iou[i, j] >= thr:
                pairs.append((i, j)); used_g.add(i); used_t.add(j)
        rg = [i for i in range(len(fd.gt_ids)) if i not in used_g]
        rt = [j for j in range(len(fd.tr_ids)) if j not in used_t]
        if rg and rt:
            sub = fd.iou[np.ix_(rg, rt)]
            cost = np.where(sub >= thr, 1 - sub, _BIG)
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < _BIG:
                    pairs.append((rg[r], rt[c]))
        for i, j in pairs:
            gid, tid = int(fd.gt_ids[i]), int(fd.tr_ids[j])
            if gid in last and last[gid] != tid:
                idsw += 1
            last[gid] = tid
        tp += len(pairs)
        fn += len(fd.gt_ids) - len(pairs)
        fp += len(fd.tr_ids) - len(pairs)
    n_tr = tp + fp
    return {
        "num_gt": n_gt, "num_pred": n_tr, "TP": tp, "FP": fp, "FN": fn, "IDSW": idsw,
        "MOTA": 1 - (fn + fp + idsw) / n_gt if n_gt else float("nan"),
        "recall": tp / n_gt if n_gt else float("nan"),
        "precision": tp / n_tr if n_tr else float("nan"),
    }


# --------------------------------------------------------------------------- #
# ID measures: IDF1, IDP, IDR
# --------------------------------------------------------------------------- #
def id_measures(frames: list[FrameData], thr: float = 0.5) -> dict:
    """
    Identity measures (Ristani et al. 2016). Each GT trajectory is paired with at most
    one tracker trajectory for the whole video so that the number of frames where the
    pair overlaps with IoU >= thr (IDTP) is maximal.
        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
    """
    gids = sorted({int(i) for fd in frames for i in fd.gt_ids})
    tids = sorted({int(i) for fd in frames for i in fd.tr_ids})
    gi, ti = {g: k for k, g in enumerate(gids)}, {t: k for k, t in enumerate(tids)}
    overlap = np.zeros((len(gids), len(tids)))
    n_gt = n_tr = 0
    for fd in frames:
        n_gt += len(fd.gt_ids); n_tr += len(fd.tr_ids)
        for a, gid in enumerate(fd.gt_ids):
            for b, tid in enumerate(fd.tr_ids):
                if fd.iou[a, b] >= thr:
                    overlap[gi[int(gid)], ti[int(tid)]] += 1
    idtp = 0.0
    if overlap.size:
        r, c = linear_sum_assignment(-overlap)
        idtp = float(overlap[r, c].sum())
    idfn, idfp = n_gt - idtp, n_tr - idtp
    den = 2 * idtp + idfp + idfn
    return {
        "IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
        "IDF1": 2 * idtp / den if den else float("nan"),
        "IDP": idtp / n_tr if n_tr else float("nan"),
        "IDR": idtp / n_gt if n_gt else float("nan"),
    }


# --------------------------------------------------------------------------- #
# HOTA (Luiten et al. 2021), following the TrackEval reference implementation
# --------------------------------------------------------------------------- #
def hota(frames: list[FrameData], alphas: np.ndarray | None = None) -> dict:
    """HOTA = sqrt(DetA * AssA), averaged over IoU thresholds alpha = 0.05 ... 0.95."""
    alphas = np.arange(0.05, 0.951, 0.05) if alphas is None else np.asarray(alphas)
    gids = sorted({int(i) for fd in frames for i in fd.gt_ids})
    tids = sorted({int(i) for fd in frames for i in fd.tr_ids})
    gi, ti = {g: k for k, g in enumerate(gids)}, {t: k for k, t in enumerate(tids)}
    gcount, tcount = np.zeros(len(gids)), np.zeros(len(tids))
    potential = np.zeros((len(gids), len(tids)))
    for fd in frames:
        g_idx = [gi[int(i)] for i in fd.gt_ids]; t_idx = [ti[int(i)] for i in fd.tr_ids]
        for k in g_idx: gcount[k] += 1
        for k in t_idx: tcount[k] += 1
        if g_idx and t_idx:
            s = np.where(fd.iou >= 0.05 - 1e-10, fd.iou, 0.0)
            potential[np.ix_(g_idx, t_idx)] += s
    global_align = potential / np.maximum(gcount[:, None] + tcount[None, :] - potential, 1e-12)
    n_gt, n_tr = int(gcount.sum()), int(tcount.sum())

    hota_a, det_a, ass_a = [], [], []
    for a in alphas:
        matches = np.zeros_like(potential)
        tp = 0
        for fd in frames:
            if len(fd.gt_ids) == 0 or len(fd.tr_ids) == 0:
                continue
            g_idx = [gi[int(i)] for i in fd.gt_ids]; t_idx = [ti[int(i)] for i in fd.tr_ids]
            score = global_align[np.ix_(g_idx, t_idx)] * fd.iou
            for r, c in zip(*linear_sum_assignment(-score)):
                if fd.iou[r, c] >= a - 1e-10:
                    matches[g_idx[r], t_idx[c]] += 1; tp += 1
        fn, fp = n_gt - tp, n_tr - tp
        ass = matches / np.maximum(gcount[:, None] + tcount[None, :] - matches, 1e-12)
        A = float((matches * ass).sum() / max(tp, 1))
        D = tp / max(tp + fn + fp, 1)
        hota_a.append(np.sqrt(A * D)); det_a.append(D); ass_a.append(A)
    return {"HOTA": float(np.mean(hota_a)), "DetA": float(np.mean(det_a)), "AssA": float(np.mean(ass_a)),
            "HOTA_alphas": [round(float(a), 2) for a in alphas], "HOTA_per_alpha": [float(h) for h in hota_a]}


# --------------------------------------------------------------------------- #
# Task-specific: target loss and recovery
# --------------------------------------------------------------------------- #
def loss_intervals(loc: pd.DataFrame, fps: float, thr: float = 0.5, max_recovery_s: float = 1.5) -> pd.DataFrame:
    """
    A loss interval is a maximal run of consecutive GT frames where the tracker does not
    follow the target (IoU < thr). For each interval report its length, whether the target
    was re-acquired within the video, whether the tracker ID after re-acquisition equals
    the ID before the loss, and whether both conditions of the requirement hold
    (re-acquired within max_recovery_s AND same ID). Expects one GT object.
    """
    loc = loc.sort_values("frame").reset_index(drop=True)
    lost = (loc["iou"] < thr).to_numpy()
    rows, i, n = [], 0, len(loc)
    while i < n:
        if not lost[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and lost[j + 1]:
            j += 1
        before = loc["tr_id"].iloc[i - 1] if i > 0 else None
        after = loc["tr_id"].iloc[j + 1] if j + 1 < n else None
        frames_lost = j - i + 1
        seconds = frames_lost / fps
        recovered = after is not None
        same = recovered and before is not None and int(before) == int(after)
        rows.append({
            "start_frame": int(loc["frame"].iloc[i]), "end_frame": int(loc["frame"].iloc[j]),
            "frames_lost": frames_lost, "seconds_lost": round(seconds, 3),
            "reacquired": recovered, "id_before": before, "id_after": after, "id_kept": bool(same),
            "meets_requirement": bool(same and seconds <= max_recovery_s),
        })
        i = j + 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, fps: float = 25.0, iou_thr: float = 0.5,
             max_recovery_s: float = 1.5, precision_px: float = 20.0) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run every metric. Returns (summary dict, per-frame localisation, loss intervals)."""
    frames = build_frames(gt, pred)
    loc = per_gt_localisation(frames)
    finite = np.isfinite(loc["centre_err_px"])
    locked = loc["iou"] >= iou_thr
    _, _, auc = success_auc(loc["iou"].to_numpy())

    summary: dict = {
        "frames_gt": int(gt["frame"].nunique()), "frames_pred": int(pred["frame"].nunique()),
        "gt_ids": int(gt["id"].nunique()), "pred_ids": int(pred["id"].nunique()),
        "iou_threshold": iou_thr, "fps_assumed": fps,
    }
    summary.update(clear_mot(frames, iou_thr))
    summary.update(id_measures(frames, iou_thr))
    h = hota(frames)
    summary.update({k: h[k] for k in ("HOTA", "DetA", "AssA")})
    summary.update({
        "success_AUC": auc,
        f"precision@{precision_px:g}px": precision_at(loc["centre_err_px"].to_numpy(), precision_px),
        "mean_IoU_all_frames": float(loc["iou"].mean()),
        "mean_IoU_when_locked": float(loc.loc[locked, "iou"].mean()) if locked.any() else float("nan"),
        "median_centre_err_px_when_locked": float(loc.loc[locked & finite, "centre_err_px"].median()) if locked.any() else float("nan"),
    })
    intervals = loss_intervals(loc, fps, iou_thr, max_recovery_s) if gt["id"].nunique() == 1 else pd.DataFrame()
    if len(intervals):
        summary["loss_intervals"] = int(len(intervals))
        summary["losses_id_kept"] = int(intervals["id_kept"].sum())
        summary["losses_meeting_requirement"] = int(intervals["meets_requirement"].sum())
        summary["max_loss_seconds"] = float(intervals["seconds_lost"].max())
    return summary, loc, intervals


def plot_report(loc: pd.DataFrame, path: Path, precision_px: float = 20.0) -> None:
    """IoU timeline coloured by tracker ID, success plot and precision plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(12, 11))
    cmap = plt.get_cmap("tab10")
    for k, tid in enumerate(sorted(t for t in loc["tr_id"].unique() if t >= 0)):
        s = loc[loc["tr_id"] == tid]
        ax[0].scatter(s["frame"], s["iou"], s=6, color=cmap(k % 10), label=f"ID {tid}")
    ax[0].axhline(0.5, color="k", ls="--", lw=1)
    ax[0].set(xlabel="frame", ylabel="IoU with GT", title="IoU per frame (colour = tracker ID)", ylim=(-0.03, 1.03))
    ax[0].legend(ncol=8, fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.3))
    thr, s, auc = success_auc(loc["iou"].to_numpy())
    ax[1].plot(thr, s, lw=2); ax[1].fill_between(thr, s, alpha=0.15)
    ax[1].set(xlabel="IoU threshold", ylabel="success rate", title=f"Success plot, AUC = {auc:.3f}")
    d = np.arange(0, 101)
    err = loc["centre_err_px"].to_numpy()
    ax[2].plot(d, [(err <= x).mean() for x in d], lw=2); ax[2].axvline(precision_px, color="k", ls="--", lw=1)
    ax[2].set(xlabel="centre error threshold, px", ylabel="precision",
              title=f"Precision plot, P@{precision_px:g}px = {precision_at(err, precision_px):.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground-truth CSV")
    ap.add_argument("--pred", required=True, help="tracker output CSV")
    ap.add_argument("--fps", type=float, default=25.0, help="video frame rate, used to convert frames to seconds")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for a match")
    ap.add_argument("--max-recovery-s", type=float, default=1.5, help="requirement: max time to re-acquire the target")
    ap.add_argument("--precision-px", type=float, default=20.0, help="centre-error threshold for the precision metric")
    ap.add_argument("--out", default=None, help="directory for summary.json, per_frame.csv, loss_intervals.csv")
    ap.add_argument("--plot", action="store_true", help="also write report.png (needs matplotlib)")
    a = ap.parse_args()

    summary, loc, intervals = evaluate(load_tracks(a.gt), load_tracks(a.pred), a.fps, a.iou_thr,
                                       a.max_recovery_s, a.precision_px)
    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"{k:<{width}}  {v:.4f}" if isinstance(v, float) else f"{k:<{width}}  {v}")
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        loc.to_csv(out / "per_frame.csv", index=False)
        intervals.to_csv(out / "loss_intervals.csv", index=False)
        if a.plot:
            plot_report(loc, out / "report.png", a.precision_px)
        print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()