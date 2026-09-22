# Thermal-Referenced Prototype Calibration (TRPC)

The primary TRPC experiment now replaces COXNet's complete Cross-Layer Fusion
Module (CLFM). RGB and Thermal features with equal strides are paired directly;
there is no DWT/IDWT, frequency fusion, legacy DeConv, or interpolation fallback.
TRPC is intended to reduce semantic/domain discrepancy before the original
Adaptive Alignment Module (AAM), while leaving spatial alignment to AAM.

The first cross-stage TRPC implementation is retained in `TRPC.py` only for
checkpoint and result reproduction. Its learned residual was found to be
effectively inactive; see the [same-stage design and diagnosis](trpc_same_stage_analysis_ko.md).

## Primary architecture: Thermal-conditioned same-stage TRPC

```text
Thermal feature -> K learned scene prototypes -> key/value
RGB feature at each location -> query -> soft attention over Thermal prototypes
Thermal condition -> spatial FiLM scale/shift of the local RGB feature
calibrated RGB + original Thermal -> original AAM/HOFM and DSR/MSF
```

The main equations are:

```text
P_T       = Pool_K(F_T)
a(x)      = softmax(q(F_R(x)) k(P_T)^T / sqrt(d))
h_T(x)    = sum_k a_k(x) v(P_T^k)
[s(x),b(x)] = MLP(h_T(x))
F_R_cal(x)  = F_R(x) + epsilon [tanh(s(x)) LN(F_R(x)) + tanh(b(x))]
```

The prototypes summarize the Thermal scene; they are not claimed to be object
correspondences. RGB spatial structure remains at the original RGB location,
and AAM remains responsible for spatial alignment.

## Optimization contract

- Detector gradients flow through the complete conditioning path, including
  the Thermal prototype extractor. There is no `detach()`.
- The thermal objectness head is supervised using thermal-coordinate GT.
- There is no RGB objectness gate, hard prototype matching, matching confidence,
  or prototype-alignment loss.
- `epsilon=0.1` is fixed. The FiLM output has a small non-zero initialization
  (`std=1e-3`), so its learning path is active at the first step.
- Prototype diversity loss is disabled in the primary config; detection loss
  and weak Thermal targetness are the only objectives for the new path.

## Losses and diagnostics

The detector objective adds only:

- `loss_trpc_targetness`: balanced focal loss on thermal box support.

The log also reports:

- `trpc_conditioning_attention_entropy`
- `trpc_conditioning_prototype_usage`
- `trpc_conditioning_rms`, `trpc_film_raw_rms`
- `trpc_film_scale_abs_mean`, `trpc_film_shift_abs_mean`
- `trpc_modulation_ratio`: modulation before fixed epsilon
- `trpc_delta_ratio`: final feature change after fixed epsilon
- `trpc_realized_delta_ratio`: change retained after the residual addition

`delta_ratio` is the relative feature-change magnitude
`||F_R_cal - F_R||_2 / (||F_R||_2 + eps)`. Attention statistics diagnose token
use; they are not correspondence accuracy or evidence of domain calibration.

The legacy `TRPC.py` config still uses mutual matching, detached Thermal
references, RGB objectness reconstruction, and zero-initialized projection.
It also logs pre-projection residual RMS, post-projection RMS, pre-scale map
ratio, and final `delta_ratio` so the attenuation stage can be identified.

The same padding mask is supplied during training and `simple_test`, so padded
feature cells do not participate in prototype extraction or reconstruction.

## Files

```text
configs/coxnet/trpc/TRPC.py
configs/coxnet/trpc/TRPC_same_stage.py
configs/coxnet/trpc/same_stage_no_trpc.py
mmdet/models/utils/trpc.py
mmdet/models/utils/fusion_strategy.py
mmdet/models/detectors/fusionnet_xo.py
tests/test_models/test_utils/test_trpc.py
```

## Data

No datasets, checkpoints, or training outputs are versioned. Place the
RGBTDronePerson dataset under `data/RGBTDronePerson/`, or override `data_root`
in `configs/_base_/datasets/rgbtdroneperson_detection.py`.

## Training

```bash
python tools/train.py configs/coxnet/trpc/TRPC_same_stage.py --seed 0 --deterministic
```

The default output directory is:

```text
work_dir/coxmamba/rgbtdroneperson/trpc/TRPC_same_stage
```

For distributed training:

```bash
bash tools/dist_train.sh configs/coxnet/trpc/TRPC_same_stage.py 4 --deterministic
```

## Evaluation

```bash
python tools/test.py \
  configs/coxnet/trpc/TRPC_same_stage.py \
  /path/to/checkpoint.pth \
  --eval bbox
```
