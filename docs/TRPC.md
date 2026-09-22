# Thermal-Referenced Prototype Calibration (TRPC)

The primary TRPC experiment now replaces COXNet's complete Cross-Layer Fusion
Module (CLFM). RGB and Thermal features with equal strides are paired directly;
there is no DWT/IDWT, frequency fusion, legacy DeConv, or interpolation fallback.
TRPC is intended to reduce semantic/domain discrepancy before the original
Adaptive Alignment Module (AAM), while leaving spatial alignment to AAM.

The first cross-stage TRPC implementation is retained in `TRPC.py` only for
checkpoint and result reproduction. Its learned residual was found to be
effectively inactive; see the [same-stage design and diagnosis](trpc_same_stage_analysis_ko.md).

## Architecture

```text
RGB, Thermal
  -> task-learned RGB and objectness-supervised Thermal prototypes (P_R, P_T)
  -> cosine mutual top-1 matching
  -> detached Thermal reference guidance
  -> gated residual calibration of RGB prototypes only
  -> reconstruction with the RGB spatial assignment map
  -> original AAM/HOFM and DSR/MSF
```

The main equations are:

```text
P_T_ref = stopgrad(P_T)
G_T     = Match(P_R, P_T_ref) P_T_ref
alpha   = sigmoid(MLP[P_R, G_T, |P_R - G_T|])
P_R_cal = P_R + alpha * phi(G_T - P_R)
F_R_cal = F_R + lambda * A_R Delta_P_R
```

Thermal supplies **what** semantic correction is useful. The RGB assignment
map supplies **where** to reconstruct it. Thermal coordinates are never copied
directly into RGB before alignment.

## Optimization contract

- The calibration path cannot send detector gradients into the thermal
  prototypes (`P_T_ref = stopgrad(P_T)`).
- The thermal backbone still learns through the unchanged downstream
  AAM/HOFM detector path.
- The thermal objectness head is supervised using thermal-coordinate GT.
- RGB objectness is task-learned; thermal GT is not imposed at potentially
  displaced RGB coordinates.
- The final residual projection is zero-initialized, so iteration-zero TRPC is
  exactly the incoming same-stage RGB feature. In the legacy cross-stage config,
  it is the retained DeConv output.
- Prototype diversity is a weak embedding de-correlation loss, not a hard
  spatial orthogonality constraint.

## Losses and diagnostics

The detector objective adds:

- `loss_trpc_targetness`: balanced focal loss on thermal box support.
- `loss_trpc_diversity`: weak RGB/Thermal prototype collapse prevention.

The log also reports:

- `trpc_match_rate`, `trpc_match_confidence`
- `trpc_gate_mean`, `trpc_delta_ratio`
- `trpc_proto_rgb_cos_before`, `trpc_proto_rgb_cos_after`
- `trpc_attention_entropy_rgb`, `trpc_attention_entropy_thermal`
- `trpc_prototype_usage`, with modality-specific variants

`delta_ratio` is the relative feature-change magnitude
`||F_R_cal - F_R||_2 / (||F_R||_2 + eps)`. The prototype cosine values are
computed before the output projection, RGB assignment reconstruction, and
residual scaling. They are internal diagnostics and do not establish that the
actual AAM input was semantically calibrated. Likewise, `match_rate` only
reports how often mutual top-k pairing exists. It is not correspondence
accuracy, and the current matcher has no semantic-similarity rejection
threshold. Mutual top-1 is intentionally retained for the first experiment;
relax or reject matches only through a controlled ablation.

Claims about domain calibration require separate feature-level measurements,
cross-modal correspondence accuracy, AAM alignment error, and downstream
detection results. The diagnostics above are insufficient on their own.

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
