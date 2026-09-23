# TOPC: Thermal-Anchored Object Prototype Calibration

TOPC is a controlled, same-stage replacement for the complete COXNet CLFM.
It tests one narrow hypothesis: before AAM performs spatial alignment, a
Thermal object anchor can supply an object-conditioned semantic discrepancy
that improves the corresponding local RGB representation.

## Scope

- RGB and Thermal FPNs both use `start_level=1` (P3-P6).
- TOPC is applied only at P3; P4-P6 enter AAM/HOFM directly.
- TOPC modifies RGB only. The original Thermal feature enters AAM/HOFM
  unchanged.
- CLFM cross-stage pairing, DeConv, DWT/IDWT, and frequency fusion are absent.
- There is no P4 context branch, RGB proposal head, learned gate/router, EDL,
  FiLM, contrastive/cycle objective, or detector-utility branch.

## Four-stage method

### 1. Thermal object discovery

A depthwise `3x3`/GELU/`1x1` head predicts a P3 Thermal objectness heatmap.
Training targets are object-wise normalized Gaussians centered on Thermal GT
boxes. Candidate selection uses threshold `0.05` and top-100 without local
peak NMS so nearby tiny targets are not explicitly suppressed.

### 2. Object prototype in one shared calibration space

One shared `1x1` projection and one shared channel LayerNorm are reused for RGB
and Thermal:

```text
Z_R = g(F_R),  Z_T = g(F_T)
```

For each Thermal candidate, TOPC computes a heatmap-weighted `3x3` prototype
from `Z_T`. Padding cells are excluded and the weights are normalized by their
valid mass. This is a candidate-specific prototype rather than a global bank.

### 3. Thermal-to-RGB local semantic search

The Thermal prototype queries the radius-2 (`5x5`) neighborhood around the
same nominal RGB P3 coordinate. Cosine similarities in the shared space are
divided by temperature `0.2` and normalized with a full local softmax. No
feature is warped. The attention-weighted RGB prototype gives the semantic
discrepancy:

```text
d_i = P_T_i - P_R_i
```

Maximum attention is used as the only matching-confidence coefficient. It is
a deterministic statistic, not a learned router.

### 4. Sparse RGB residual before AAM

A small nonzero-initialized linear projection maps `d_i` to RGB channels. The
residual is written through the same local attention. For overlapping
candidates, TOPC accumulates weighted residuals, divides by total attention
mass, and multiplies by `clamp(mass, 0, 1)`. Thus a single candidate retains
its attention magnitude while overlaps do not amplify the correction without
bound.

```text
F_R_cal = F_R + delta
(F_R_cal, F_T) -> original AAM/HOFM
```

Pixels outside candidate support are exact identity. No candidate also gives
an exact identity operation.

## Objective and diagnostics

The detector objective is unchanged except for one weighted auxiliary loss:

```text
L = L_detection + 0.1 * L_topc_objectness
```

Training logs expose detached diagnostics named `topc_*`, including candidate
count/mean, matching confidence, support ratio, and delta ratio. These metrics
show whether the path is active; none by itself proves better localization or
cross-modal correspondence.

## Training

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/coxnet/topc/TOPC.py \
  --seed 0 --deterministic \
  --work-dir work_dir/coxmamba/rgbtdroneperson/topc/seed0
```

Run seeds 0, 1, and 2 sequentially with corresponding work directories. Data
is intentionally excluded from the repository.

## Evaluation status

The implementation has unit coverage for Gaussian targets, adjacent candidate
survival, valid-padding exclusion, shared projection reuse, prototype pooling,
local matching, residual overlap, empty-candidate identity, gradients, fusion
integration, and config construction. This validates the implementation
contract only. AP improvement over COXNet, same-stage no-calibration, OEPC, or
TRPC must be established from completed controlled runs.
