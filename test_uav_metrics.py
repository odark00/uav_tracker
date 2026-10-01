"""Unit tests for uav_metrics. Run: pytest -q test_uav_metrics.py"""
import numpy as np
import pandas as pd
import pytest

import uav_metrics as um


def make(rows):
    return pd.DataFrame(rows, columns=["frame", "id", "x1", "y1", "x2", "y2"])


def moving_gt(n=100, gid=1):
    return make([(f, gid, 10 + f, 20, 50 + f, 60) for f in range(1, n + 1)])


def test_iou_known_values():
    assert um.iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1.0)
    assert um.iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)  # 50 / 150
    assert um.iou_matrix([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0.0
    assert um.iou_matrix(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


def test_perfect_tracker():
    gt = moving_gt()
    s, _, iv = um.evaluate(gt, gt.copy())
    for k in ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "recall", "precision"):
        assert s[k] == pytest.approx(1.0), k
    assert s["IDSW"] == 0 and s["FP"] == 0 and s["FN"] == 0
    assert len(iv) == 0


def test_empty_tracker():
    gt = moving_gt()
    s, loc, _ = um.evaluate(gt, make([]))
    assert s["TP"] == 0 and s["FN"] == 100 and s["MOTA"] == 0.0 and s["IDF1"] == 0.0 and s["HOTA"] == 0.0
    assert (loc["iou"] == 0).all() and np.isinf(loc["centre_err_px"]).all()


def test_id_switch_is_counted_once():
    gt = moving_gt()
    pred = gt.copy()
    pred.loc[pred["frame"] > 50, "id"] = 2            # perfect boxes, ID changes at frame 51
    s, _, _ = um.evaluate(gt, pred)
    assert s["IDSW"] == 1
    assert s["MOTA"] == pytest.approx(1 - 1 / 100)     # only the switch is an error
    assert s["IDF1"] == pytest.approx(0.5)             # one ID covers half of the time
    assert s["AssA"] < 1.0 and s["DetA"] == pytest.approx(1.0)


def test_gap_with_same_id_is_recovered():
    gt = moving_gt()
    pred = gt[~gt["frame"].between(41, 60)]            # 20 missing frames, same ID afterwards
    s, _, iv = um.evaluate(gt, pred, fps=25)
    assert s["IDSW"] == 0 and s["FN"] == 20
    assert len(iv) == 1
    row = iv.iloc[0]
    assert row["frames_lost"] == 20 and row["seconds_lost"] == pytest.approx(0.8)
    assert row["id_kept"] and row["meets_requirement"]


def test_gap_with_new_id_fails_requirement():
    gt = moving_gt()
    pred = gt[~gt["frame"].between(41, 60)].copy()
    pred.loc[pred["frame"] > 60, "id"] = 9
    _, _, iv = um.evaluate(gt, pred, fps=25)
    assert not iv.iloc[0]["id_kept"] and not iv.iloc[0]["meets_requirement"]


def test_too_long_gap_fails_requirement_even_with_same_id():
    gt = moving_gt()
    pred = gt[~gt["frame"].between(11, 60)]            # 50 frames = 2 s at 25 fps
    _, _, iv = um.evaluate(gt, pred, fps=25, max_recovery_s=1.5)
    assert iv.iloc[0]["id_kept"] and not iv.iloc[0]["meets_requirement"]


def test_success_auc_equals_mean_iou():
    gt = moving_gt()
    pred = gt.copy()
    pred[["x1", "x2"]] += 3                            # constant shift -> constant IoU < 1
    _, loc, _ = um.evaluate(gt, pred)[0], um.evaluate(gt, pred)[1], None
    _, _, auc = um.success_auc(loc["iou"].to_numpy(), n_thr=1001)
    assert auc == pytest.approx(loc["iou"].mean(), abs=2e-3)


def test_multi_object_swap_detected():
    # two targets in separate lanes; the tracker swaps their IDs halfway through
    rows_gt, rows_pr = [], []
    for f in range(1, 41):
        a = (10 + f, 10, 30 + f, 30)
        b = (50 - f, 100, 70 - f, 120)
        rows_gt += [(f, 1, *a), (f, 2, *b)]
        swap = f > 20
        rows_pr += [(f, 2 if swap else 1, *a), (f, 1 if swap else 2, *b)]
    s, _, _ = um.evaluate(make(rows_gt), make(rows_pr))
    assert s["IDSW"] == 2
    assert s["IDF1"] == pytest.approx(0.5)


def test_load_tracks_validates_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("frame,id,x1\n1,1,0\n")
    with pytest.raises(ValueError):
        um.load_tracks(p)