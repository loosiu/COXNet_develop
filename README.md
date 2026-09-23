# COXNet CLFM Replacements: TOPC, OEPC, and TRPC

**IEEE Transactions on Circuits and Systems for Video Technology**, Vol. 36, No. 1, January 2026

[![paper](https://img.shields.io/badge/IEEE%20TCSVT-2026-blue)](https://doi.org/10.1109/TCSVT.2025.3595147)

**Authors:** Peiran Peng, Tingfa Xu, Liqiang Song, Mengqi Zhu, Yuqiang Fang, Jianan Li

---

## Introduction

COXNet is an RGBT tiny object detection framework that jointly addresses cross-modal fusion, misalignment, and scale variation in drone-based multi-spectral imagery. The core innovations are: **(1) CLFM** (Cross-Layer Fusion Module), which leverages wavelet decomposition to align and fuse complementary RGB and thermal features across pyramid levels; **(2) DASR** (Dynamic Adaptive Scale Refinement), which recalibrates spatial correspondences and integrates multi-scale contextual cues for robust tiny object localization; and **(3) a GeoShape-based label assignment strategy** that better fits the irregular geometry of tiny aerial targets, improving recall under severe scale imbalance.

This repository studies complete replacements for COXNet's CLFM while keeping
the original AAM/HOFM, DSR/MSF, detector head, assignment, and training recipe.
The current controlled method is **TOPC (Thermal-Anchored Object Prototype
Calibration)**. RGB and Thermal FPNs both start at level 1 and produce
equal-stride P3/P4/P5/P6 features. TOPC acts only on P3: a Thermal objectness
head selects object anchors, a shared RGB/Thermal projection defines one
calibration space, and each Thermal prototype searches a radius-2 RGB
neighborhood. The projected Thermal-minus-RGB discrepancy calibrates RGB only;
the original Thermal feature is passed unchanged to AAM/HOFM.

TOPC contains no cross-stage pairing, transposed convolution, DWT/IDWT,
frequency fusion, learned router, EDL, FiLM, contrastive loss, or P4 semantic
branch. P4-P6 go directly to their same-stage AAM/HOFM blocks. Legacy OEPC and
TRPC implementations remain available for controlled comparisons and
reproducibility. See [the TOPC method note](docs/TOPC.md),
[docs/TRPC.md](docs/TRPC.md), and
[the same-stage TRPC diagnosis](docs/trpc_same_stage_analysis_ko.md).

### TOPC data flow

```text
Thermal P3 ── objectness ── candidate-specific 3x3 prototype ───────────┐
                         shared 1x1 projection space                    │
RGB P3 ───────────────── radius-2 local semantic search ────────────────┤
                                                                       ↓
                     W(Thermal prototype - matched RGB prototype)
                                  ↓ attention/confidence support
RGB P3 ─────────────────────────── + ─────────────────────── RGB_cal P3 ─┐
                                                                         ├─ AAM/HOFM
Thermal P3 (unchanged) ──────────────────────────────────────────────────┘
```

The candidate decision is Thermal-first and uses threshold plus top-k without
local-peak NMS, so adjacent P3 cells are not suppressed. Matching is a full
softmax over the local RGB window and does not warp either modality. Maximum
attention is the only reliability coefficient. Candidate residuals retain
their attention magnitude, while overlapping residuals are mass-normalized to
avoid unbounded amplification. RGB is exactly preserved outside the selected
support and whenever no candidate survives.

Thermal-coordinate GT centers supervise a normalized Gaussian objectness map.
The only added objective is `0.1 * loss_topc_objectness`; detection loss trains
the shared projection, matching, and discrepancy residual. The inherited
fused-to-Thermal feature loss is disabled.

---

## Main Results

### TOPC status

TOPC has passed config construction, same-stage shape checks, CPU
forward/backward, padding exclusion, Thermal-input preservation, adjacent
candidate, overlap-normalization, empty-candidate identity, and non-zero
gradient checks. Its three-seed result is not yet available; structural
validation must not be read as an AP improvement. Runtime, complete-model cost,
and FPS must still be measured on the target GPU.

### Initial cross-stage TRPC three-seed result

The completed legacy cross-stage seeds 0/1/2 are summarized in the COXNet
Table 1 column order. These results do not apply to the new same-stage config.
The best-checkpoint mean mAP50 is **45.77 ± 0.43**, compared with
**45.70 ± 0.49** for the paired COXNet reruns. See
[the full per-seed table and interpretation](docs/trpc_table1_ko.md).

### RGBTDronePerson

[[Dataset Link](https://nnnnerd.github.io/RGBTDronePerson/)]

| Method | mAP25 | mAP50 (all) | mAP50 (tiny) | mAP50 (tiny1) | mAP50 (tiny2) | mAP50 (tiny3) | mAP50 (small) | FLOPs (G) | FPS |
|--------|-------|-------------|--------------|---------------|---------------|---------------|---------------|-----------|-----|
| Cascade R-CNN | 42.47 | 31.55 | 31.99 | 0.00 | 29.43 | 37.77 | 33.61 | 76.23 | 11.8 |
| RetinaNet | 38.92 | 22.87 | 23.34 | 4.69 | 13.57 | 32.66 | 15.77 | 49.48 | 19.4 |
| FCOS | 45.21 | 29.89 | 30.71 | 9.40 | 22.73 | 34.87 | 26.00 | 78.32 | 19.2 |
| ATSS | 53.14 | 36.24 | 37.47 | 16.92 | 24.16 | 43.59 | 24.62 | 48.73 | 19.0 |
| GFL | 56.91 | 39.74 | 41.67 | 12.47 | 30.23 | 47.68 | 26.72 | 49.22 | 18.7 |
| FCOS w/ RFLA | 51.41 | 38.20 | 39.45 | 25.87 | 30.93 | 44.31 | 25.32 | 110.62 | 13.9 |
| QueryDet | 55.16 | 37.07 | 37.75 | 17.89 | 24.90 | 44.05 | 25.52 | 121.67 | 12.8 |
| TINet | 40.34 | 28.30 | 28.60 | 0.00 | 24.12 | 34.99 | 34.97 | 92.80 | 13.0 |
| CFT | 37.32 | 22.69 | 22.83 | 16.72 | 18.14 | 27.52 | 8.14 | 112.02 | 10.8 |
| HRFuser | 33.24 | 22.23 | 22.50 | 0.00 | 26.73 | 26.26 | 23.85 | 54.17 | 4.5 |
| QFDet | 57.34 | 42.08 | 44.04 | 20.27 | 30.09 | 50.36 | 26.78 | 81.43 | 14.2 |
| QFDet* | 61.62 | 46.72 | 48.75 | 22.15 | 37.91 | 53.71 | 28.41 | 242.82 | 5.7 |
| **COXNet (Ours)** | **59.01** | **45.57** | **47.18** | **27.37** | **35.55** | **52.56** | **29.74** | **51.27** | **17.6** |
| **COXNet* (Ours)** | **62.76** | **50.04** | **51.82** | **23.08** | **40.10** | **56.76** | **30.89** | **123.59** | **12.9** |

† indicates methods adapted for the RGBT baseline detector. * denotes models utilizing detection heads with P2-P6 feature maps.

### VTUAV-det

[[Dataset Link](https://nnnnerd.github.io/RGBTDronePerson/)]

| Method | mAP | mAP50 | mAP75 | mAPs | mAPm | mAPl | FPS |
|--------|-----|-------|-------|------|------|------|-----|
| ATSS | 21.4 | 52.7 | 13.9 | 5.9 | 20.8 | 45.2 | 25.1 |
| GFL | 29.8 | 67.8 | 22.2 | 10.3 | 27.9 | 55.7 | 23.6 |
| QueryDet | 29.5 | 68.9 | 20.2 | 7.8 | 29.9 | 53.5 | 14.6 |
| CFT | 8.7 | 29.3 | 2.4 | 3.9 | 8.5 | 23.4 | 8.3 |
| HRFuser | 25.9 | 55.9 | 20.1 | 2.7 | 27.9 | 51.9 | 5.6 |
| TINet | 26.8 | 59.4 | 20.1 | 1.2 | 29.0 | 53.7 | 14.5 |
| QFDet | 31.1 | 70.4 | 22.9 | 12.5 | 20.4 | 56.8 | 15.3 |
| QFDet* | 33.3 | 75.5 | 24.2 | 18.1 | 32.4 | 57.2 | 9.4 |
| **COXNet (Ours)** | **31.5** | **71.8** | **23.1** | **15.3** | **30.6** | **56.0** | **21.2** |
| **COXNet* (Ours)** | **33.5** | **76.1** | **25.1** | **18.6** | **32.6** | **56.8** | **15.0** |

### NII-CU

[[Dataset Link](https://www.okutama-segmentation.org/)]

| Method | mAP | mAP50 | mAP75 | FPS |
|--------|-----|-------|-------|-----|
| ATSS | 54.6 | 95.5 | 55.0 | 24.1 |
| GFL | 61.0 | 96.7 | 71.2 | 19.6 |
| CFT | 51.2 | 95.1 | 58.7 | 9.2 |
| QFDet | 58.3 | 96.7 | 65.3 | 17.3 |
| QFDet* | 63.7 | 97.6 | 76.4 | 10.3 |
| **COXNet (Ours)** | **61.4** | **98.2** | **70.5** | **17.9** |
| **COXNet* (Ours)** | **65.4** | **97.9** | **79.6** | **13.1** |

---

## Installation

**Requirements:** CUDA 11.3 · Python 3.9.18

**Step 1 — Clone the repository**

```bash
git clone git@github.com:loosiu/COXNet_develop.git
cd COXNet_develop
```

**Step 2 — Install PyTorch**

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3 — Install mmcv-full**

```bash
pip install mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

**Step 4 — Install remaining dependencies**

```bash
pip install -r requirements.txt
pip install setuptools==59.5.0 --force-reinstall
python setup.py develop
```

---

## Dataset Preparation

COXNet is evaluated on three RGBT benchmarks:

| Dataset | Description | Link |
|---------|-------------|------|
| **RGBTDronePerson** | Drone-based RGB-thermal person detection | [Project page](https://nnnnerd.github.io/RGBTDronePerson/) |
| **VTUAV-det** | Aerial vehicle and UAV detection | [Project page](https://nnnnerd.github.io/RGBTDronePerson/) |
| **NII-CU** | 6,000 RGBT image pairs with 19,000 annotated instances (pedestrian, vehicle, cyclist) | [Dataset](https://www.okutama-segmentation.org/) |

Datasets are intentionally not included in this repository. Organize them
locally under the git-ignored `data/` directory as follows:

```
data/
├── RGBTDronePerson/
│   ├── train/
│   │   ├── visible/
│   │   └── infrared/
│   └── val/
│       ├── visible/
│       └── infrared/
└── VTUAV/
    ├── train/
    └── val/
```

Update the `data_root` paths in the corresponding config files under `configs/_base_/datasets/` before training.

---

## Training

**TOPC complete CLFM replacement, single GPU (seed 0)**

```bash
python tools/train.py configs/coxnet/topc/TOPC.py \
    --seed 0 --deterministic
```

Use the same command with seeds 1 and 2 and separate `--work-dir` values for a
three-seed comparison. TOPC uses same-stage P3-P6 features, calibrates RGB P3
only, leaves Thermal unchanged, and then runs the original AAM/HOFM.

**Balanced OEPC complete CLFM replacement, single GPU (seed 0)**

```bash
python tools/train.py configs/coxnet/oepc/OEPC_balanced_core.py \
    --seed 0 --deterministic
```

This config uses RGB/Thermal FPN `start_level=1`, applies sparse OEPC at P3,
removes all CLFM/DeConv/DWT operations, and keeps the original AAM/HOFM and
detector. P4-P6 go directly to their same-stage AAM/HOFM blocks.

**Same-stage TRPC, single GPU (seed 0)**

```bash
python tools/train.py configs/coxnet/trpc/TRPC_same_stage.py --seed 0 --deterministic
```

**Same-stage control without TRPC**

```bash
python tools/train.py configs/coxnet/trpc/same_stage_no_trpc.py --seed 0 --deterministic
```

**TRPC, multi-GPU (e.g., 4 GPUs)**

```bash
bash tools/dist_train.sh configs/coxnet/trpc/TRPC_same_stage.py 4 --deterministic
```

**Original COXNet baseline**

```bash
python tools/train.py configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py --seed 0 --deterministic
```

Available configs:

```
configs/coxnet/
├── coxnet_r50_fpn_1x_rgbtdroneperson.py
├── coxnet_star_r50_fpn_1x_rgbtdroneperson.py
├── coxnet_r50_fpn_1x_vtuav.py
├── coxnet_star_r50_fpn_1x_vtuav.py
├── oepc/
│   ├── OEPC_balanced_core.py
│   └── OEPC_same_stage.py
├── topc/
│   └── TOPC.py
└── trpc/
    ├── TRPC.py
    ├── TRPC_same_stage.py
    └── same_stage_no_trpc.py
```

---

## Evaluation

```bash
python tools/test.py \
    configs/coxnet/topc/TOPC.py \
    /path/to/checkpoint.pth \
    --eval bbox
```

For a strict same-stage control without any RGB calibration, use
`configs/coxnet/trpc/same_stage_no_trpc.py`. It has the same RGB/Thermal FPN
levels and directly feeds both features into AAM/HOFM.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{peng2025coxnet,
  title={COXNet: Cross-layer fusion with adaptive alignment and scale integration for RGBT tiny object detection},
  author={Peng, Peiran and Xu, Tingfa and Zhu, Mengqi Zhu and Fang, Yuqiang and Li, Jianan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

---

## Acknowledgement

This work was supported by the Natural Science Foundation of Chongqing, China, under Grant cstc2021jcyj-msxmX1130.

This codebase is built upon [MMDetection](https://github.com/open-mmlab/mmdetection). We thank the OpenMMLab team for their excellent open-source framework.
