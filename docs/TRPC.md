# Thermal-Referenced Prototype Calibration (TRPC)

TRPC is a complete replacement for COXNet's Cross-Layer Fusion Module (CLFM).
It reduces the semantic/domain discrepancy before the original Adaptive
Alignment Module (AAM), while leaving spatial alignment to AAM.

## Architecture

```text
RGB, Thermal
  -> object-aware prototype extraction (P_R, P_T)
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
  exactly the retained DeConv path (`F_R_cal = F_R`).
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

`delta_ratio` is the relative calibration magnitude
`||F_R_cal - F_R||_2 / (||F_R||_2 + eps)`. A mechanism-consistent run should
eventually show matched-prototype cosine similarity increasing after
calibration. Mutual top-1 is intentionally retained for the first experiment;
relax it only if match rate remains too low throughout training.

## Files

```text
configs/coxnet/trpc/TRPC.py
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
python tools/train.py configs/coxnet/trpc/TRPC.py --seed 0 --deterministic
```

The default output directory is:

```text
work_dir/coxmamba/rgbtdroneperson/trpc/TRPC
```

For distributed training:

```bash
bash tools/dist_train.sh configs/coxnet/trpc/TRPC.py 4 --deterministic
```

## Evaluation

```bash
python tools/test.py \
  configs/coxnet/trpc/TRPC.py \
  /path/to/checkpoint.pth \
  --eval bbox
```
