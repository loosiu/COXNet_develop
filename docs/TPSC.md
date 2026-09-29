# TPSC: Tiny-aware Prototype Semantic Calibration

TPSC is the current controlled CLFM-replacement experiment in this repository.
It asks whether a small set of target-aware semantic prototypes can condition
RGB features before the unchanged COXNet AAM/HOFM path. No accuracy claim is
made until the matched control, core, and three canonical seeds have completed.

## Model path

Both RGB and Thermal FPNs use `start_level=1`, producing same-stride
P3/P4/P5/P6 tensors. TPSC consumes P3 and P4 together:

```text
RGB/T P3 + valid local detail + projected P4 semantic context
  -> modality-specific projections with shared slot queries
  -> K=8 RGB and Thermal semantic prototypes per level
  -> same-slot conditioning (core)
  -> optional one-block prototype relation mixer (canonical)
  -> bounded RGB channel scaling on P3/P4 only
  -> AAM/HOFM(calibrated RGB, original Thermal)
```

P5 and P6 bypass TPSC. Thermal feature values are never modified before
AAM/HOFM, but detection gradients may train the Thermal descriptor because its
prototypes condition RGB. Spatial alignment remains AAM's responsibility.

The P3 descriptor concatenates the original feature, a padding-aware local
detail residual, and upsampled P4 semantic context. Prototype extraction uses
shared slot queries but separate RGB/Thermal projections, so the slots provide
a common semantic basis without assuming instance-level coordinate matches.
There is no spatial threshold, candidate quota, top-k selection, local matching,
feature warp, DWT/FFT branch, EDL router, or teacher network.

## Core and relation variants

`TPSC_core.py` conditions each RGB slot with the same-index Thermal slot:

```text
Q_R = P_R + MLP([P_R, P_T, P_T - P_R])
```

`TPSC_relation.py` first concatenates RGB/Thermal P3/P4 slots into 32 nodes and
uses one pre-norm four-head self-attention block. It then applies the same
conditioner. This isolates whether cross-modal/cross-scale prototype relations
add value beyond shared-slot conditioning.

Each level flattens the conditioned RGB prototypes into a channel scale:

```text
scale = 0.1 * tanh(Linear(Flatten(Q_R)))
RGB_cal = RGB * (1 + scale)
```

The linear weight starts from a small non-zero normal distribution
(`std=1e-2`), and the fixed bound guarantees channel multipliers in `[0.9,1.1]`.
There is no learned output gate that can close the entire calibration path.

## Supervision and diagnostics

Thermal GT boxes produce max-merged Gaussian foreground maps at P3 and P4.
Padding and ignored regions receive no target or attention mass. The only TPSC
auxiliary losses are:

- `loss_tpsc_coverage = 0.05 * symmetric_KL(GT, mean Thermal slot attention)`
- `loss_tpsc_diversity = 0.01 * off-diagonal slot-attention cosine`

Empty-GT and all-padding terms return finite graph-connected zero. The original
`wf_loss=True`, `wf_loss_mode='kl_v2'`, and `wf_loss_weight=0.1` remain enabled
for all three matched configurations.

Detached monitors record attention entropy, prototype cosine/effective rank,
relation attention mass, channel-scale magnitude, modulation ratio, and cached
gradient norms. These are mechanism diagnostics, not extra objectives.

Three inference interventions are implemented on the TPSC module:

- `disable_modulation=True`: exact RGB identity
- `disable_relation=True`: relation mixer identity, leaving the core path
- `shuffle_thermal_prototypes=True`: roll Thermal prototypes across a batch;
  this is explicitly a no-op for batch size one

## Reproducible comparisons

Run the matched models at seed 0 before interpreting the canonical three-seed
result:

```bash
python tools/train.py configs/coxnet/tpsc/same_stage_control.py --seed 0 --deterministic
python tools/train.py configs/coxnet/tpsc/TPSC_core.py --seed 0 --deterministic
python tools/train.py configs/coxnet/tpsc/TPSC_relation.py --seed 0 --deterministic
```

Then run `TPSC_relation.py` with seeds 1 and 2 in separate work directories.
Compare best checkpoints selected by `bbox_mAP_50` on the same validation JSON.
Report AP25/AP50/AP75/tiny/tiny1/tiny2/tiny3/small and the mechanism monitors.

The fail-fast GPU-1 queue created for the local experiment also records the
exact Git commit, config, seed, and console log for every run. It holds
`/tmp/coxnet_oepc_gpu1.lock` and refuses to mix commits.

Legacy [PRLDFC](PRLDFC.md), [TOPC](TOPC.md), OEPC, and TRPC implementations are
retained for controlled reproduction; TPSC does not alter their code paths.
