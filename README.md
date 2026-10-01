# UAV Tracker Comparison

Benchmark of single-object trackers for tracking drones in video.


## Quick start

### Install

```bash
python -m pip install -r requirements.txt
```

Some trackers are implemented as third-party integrations under [third_party](third_party). Follow their README files there if a specific tracker is missing or needs extra setup.

### Run a tracker comparison

Compare several trackers side by side in one window:

```bash
python track.py --trackers sutrack_b focustrack bytetrack --source /path/to/video.mp4
```

Save the result instead of (or as well as) showing it. Outputs are written under `runs/track/<video>_track.mp4`:

```bash
python track.py --trackers sutrack_b --source /path/to/video.mp4 --save --no-show
```

Run every video in a folder:

```bash
python track.py --trackers sutrack_b --source test_videos
```

Use a webcam:

```bash
python track.py --trackers sutrack_b --source 0
```

## Trackers

| Tracker | What it is | Runtime |
|---|---|---|
| `sutrack` | SUTrack-T224, unified single-ViT tracker (Fast-iTPN tiny), AAAI 2025 | GPU |
| `sutrack_b` | SUTrack-B224, Fast-iTPN base encoder, 2 templates with online update | GPU |
| `mcitrack_b` | MCITrack-B224, Fast-iTPN base + Mamba neck that carries context between frames, AAAI 2025 | GPU |
| `focustrack` | FocusTrack, OSTrack ViT-B for anti-UAV; enlarges its search region when the target is lost, 2025 | GPU |
| `asymtrack` | AsymTrack-B, asymmetric Siamese EfficientMod, AAAI 2025 | GPU |
| `asymtrack_t` | AsymTrack-T, shallower backbone, 256 px search region | GPU |
| `hit_b` / `hit_s` | HiT-Base / HiT-Small, hierarchical LeViT transformer, ICCV 2023 | GPU |
| `ortrack_deit` | ORTrack-DeiT, occlusion-robust ViT for UAVs, CVPR 2025 | GPU |
| `tctrackpp` | TCTrack++, AlexNet Siamese with temporal contexts, TPAMI 2023 | GPU |
| `mast` | MaST-tiny, motion-aware sparse ViT, ECCV 2026 | CPU (ONNX) |
| `avtrack` | AVTrack-DeiT, adaptive ViT | CPU (ONNX) |
| `hcat` | HCAT, ResNet-18 Siamese + hierarchical cross-attention, ECCVW 2022 | CPU (ONNX) |
| `ecohc` | ECO-HC, correlation filter on FHOG + colour names | CPU |
| `autotrack` | AutoTrack, correlation filter with automatic spatio-temporal regularisation, CVPR 2020 | CPU |
| `kalman` | Initialised on the first GT box and then corrected with the detection nearest to its prediction. |
| `deepsort`, `bytetrack` and `sort` | Multi-object trackers that cannot be initialised on a box. The target is the first confirmed track that overlaps the GT with IoU ≥ 0.5. Only that ID is followed afterwards, so a lost or switched ID loses the target. |

## Results

| Tracker | Type | AUC | Success@0.5 | Prec@20px | Norm. prec | SA | MOTA | F1 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| sutrack_b | sot | **0.725** | 0.876 | **0.933** | **0.815** | **0.712** | 0.751 | 0.875 | 73 |
| mcitrack_b | sot | 0.716 | 0.868 | 0.917 | 0.769 | 0.706 | 0.743 | 0.871 | 42 |
| focustrack | sot | 0.699 | **0.893** | 0.926 | 0.790 | 0.686 | **0.794** | **0.896** | 88 |
| sutrack | sot | 0.687 | 0.832 | 0.899 | 0.778 | 0.675 | 0.663 | 0.831 | 139 |
| asymtrack | sot | 0.680 | 0.867 | 0.897 | 0.755 | 0.647 | 0.739 | 0.868 | 163 |
| ortrack_deit | sot | 0.662 | 0.793 | 0.808 | 0.718 | 0.644 | 0.573 | 0.785 | 191 |
| hit_b | sot | 0.622 | 0.766 | 0.829 | 0.716 | 0.585 | 0.584 | 0.785 | 143 |
| asymtrack_t | sot | 0.615 | 0.774 | 0.807 | 0.700 | 0.593 | 0.543 | 0.771 | 183 |
| mast | sot | 0.590 | 0.702 | 0.739 | 0.646 | 0.514 | 0.454 | 0.708 | 62 |
| avtrack | sot | 0.554 | 0.646 | 0.654 | 0.585 | 0.502 | 0.356 | 0.659 | 64 |
| hit_s | sot | 0.537 | 0.650 | 0.661 | 0.594 | 0.468 | 0.475 | 0.705 | 147 |
| hcat | sot | 0.484 | 0.602 | 0.608 | 0.554 | 0.420 | 0.301 | 0.617 | 55 |
| autotrack | sot | 0.413 | 0.515 | 0.504 | 0.451 | 0.368 | 0.157 | 0.534 | 49 |
| ecohc | sot | 0.411 | 0.521 | 0.506 | 0.437 | 0.381 | 0.051 | 0.508 | 50 |
| tctrackpp | sot | 0.399 | 0.507 | 0.494 | 0.433 | 0.322 | 0.421 | 0.621 | 190 |
| kalman | det | 0.275 | 0.361 | 0.353 | 0.309 | 0.122 | 0.303 | 0.479 | 229 |
| deepsort | det | 0.100 | 0.129 | 0.123 | 0.113 | -0.060 | 0.120 | 0.244 | 167 |
| sort | det | 0.098 | 0.124 | 0.120 | 0.110 | -0.061 | 0.108 | 0.238 | **235** |
| bytetrack | det | 0.097 | 0.122 | 0.119 | 0.110 | -0.060 | 0.105 | 0.235 | 228 |

FPS comes from a separate pass in which each tracker ran alone on an idle GPU over 3 of these videos (`runs/fps`). The accuracy runs were spread over several sessions with other jobs running, so their timings are not comparable with each other.

**Summary:**
- `sutrack_b` is the most accurate overall: best AUC, centre precision, normalised precision and SA. It runs at 73 FPS.
- `focustrack` gives the best target / no-target decisions (Success@0.5, MOTA, F1) at 88 FPS. It is the strongest option when the drone disappears and comes back.
- `mcitrack_b` is close to `sutrack_b` in accuracy but is the slowest deep tracker (42 FPS).
- The larger models are 1–4 AUC points ahead of `sutrack` (T224), which is still the best choice above 130 FPS.
- The correlation filters (`ecohc`, `autotrack`) and `tctrackpp` come last.

### Detector-driven trackers (`det`)

These trackers use the YOLOv8n drone detector (`YOLOv8n_HuggingFace_TomSmaildrone-yolo-v1.pt`) on every frame, scaled to 1280 px. Their FPS includes the detector.

`kalman`, `deepsort`, `bytetrack` and `sort` fall far behind the single-object trackers because the detector misses the drone on most frames of this footage. On one sampled video it found the target in only 27 of 93 frames, even with a 125×75 px drone. While detections continue, ByteTrack and SORT follow the right drone with the right ID (IoU ≈ 0.72). After a long detection gap they drop the track and never pick the target up again. Tracking-by-detection is therefore only as good as the detector on this data. A single-object tracker that uses YOLO only for re-detection is the more robust design.

## Metrics

- **AUC**: area under the success plot, i.e. the mean of P(IoU > t) for t from 0 to 1.
- **Success@0.5**: share of frames with IoU > 0.5.
- **Prec@20px**: share of frames where the centre error is at most 20 px.
- **Norm. prec**: LaSOT normalised precision, where the centre error is divided by the ground-truth box size.
- **SA**: Anti-UAV state accuracy. It also rewards correctly reporting "no target".
- **MOTA / F1**: count a reported box as a match when IoU ≥ 0.5.
- **FPS**: tracker time only on full-resolution frames, without video decoding.

For the full definitions, see [metrics.py](metrics.py).

## Evaluate

```bash
python eval_tracker.py --tracker sutrack sutrack_b ... --source $VAL --count 3 --out runs/fps

python metrics.py --dir runs/eval                              # runs/eval/metrics_summary.csv

python collect_results.py                                     # results/results.csv + the tables above
python show_csv.py results/results.csv                         # or see results.ipynb
```

