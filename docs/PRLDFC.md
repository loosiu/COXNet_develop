# Prototype-Routed Local Dynamic Frequency Calibration

PRLDFC is an experimental complete replacement for COXNet's CLFM. It tests
whether a Thermal object seed can select useful low/mid/high-frequency evidence
for local RGB calibration without changing the Thermal stream or replacing the
existing AAM/HOFM alignment path.

## Scope

- Both FPNs use `start_level=1` and output equal-stride P3-P6 features.
- PRLDFC calibrates RGB P3/P4 in the canonical config.
- Thermal P3-P6 are passed to AAM/HOFM unchanged.
- P5/P6 go directly to the existing AAM/HOFM blocks.
- CLFM cross-stage pairing, DeConv, DWT/IDWT, LL fusion, and HF gates are absent.
- No hard top-k quota or local-maximum NMS is used for Thermal seeds.

## Data flow

```text
Thermal P3/P4
  -> dense seed logit + offset + scale
  -> detail-preserving prototype [center, detail, context]
  -> object-conditioned band weights and Thermal reliability

RGB/Thermal P3/P4
  -> shared learnable radial low/mid/high FFT bands
  -> band-wise Thermal-to-RGB local association
  -> dense band residuals
  -> normalized, epsilon-bounded RGB correction

(calibrated RGB, unchanged Thermal)
  -> original AAM/HOFM
```

The prototype is a router, not the transferred feature. It summarizes local
object evidence and predicts three band weights plus one Thermal reliability.
Actual correction content comes from dense frequency-band features. This avoids
reconstructing spatial detail from a small prototype set.

## Dense Thermal seed supervision

The seed head predicts one existence logit, a bounded two-dimensional offset,
and log width/height at every cell. Training assigns eligible GT to nearby valid
cells with deterministic one-to-one Hungarian matching. The assignment decision
uses detached predictions; the focal, offset, scale, and cardinality losses
remain differentiable.

P3 initially supervises objects with resized `sqrt(width * height)` in
`[0, 32)` pixels. P4 uses `[16, 64)`. The overlap is deliberate and only
affects PRLDFC seed supervision, not QLS assignment in the detector head.

Training uses a dense soft gate:

```text
gate = sigmoid((seed_logit - logit(seed_threshold)) / seed_temperature)
```

Optional sparse inference thresholds the probability but never forces a fixed
number of candidates.

## Frequency decomposition and routing

One shared frequency bank projects each modality and applies orthonormal
`rfft2`. Three normalized soft radial masks partition the spectrum. Their
widths are globally learned per PRLDFC level, constrained to sum to `0.5`, and
have a configured minimum width to prevent band collapse.

`dynamic` refers to the object-specific band weights and reliability. The FFT
boundaries are not predicted independently for every object.

For each band, the Thermal seed searches a radius-2 RGB neighborhood. Attention
chooses where to read information but its maximum is not multiplied into the
residual as a second confidence gate. When several seed supports overlap,
weighted residuals are mass-normalized before applying

```text
delta = epsilon * (1 - exp(-mass)) * tanh(weighted_average)
```

Therefore each output channel is bounded by `residual_epsilon` and invalid
padding receives exactly zero correction.

## Losses

The canonical experiment uses:

```text
L = L_detection
  + 0.10 * L_seed
  + 0.05 * L_offset
  + 0.02 * L_scale
  + 0.01 * L_cardinality
  + L_wf(kl_v2, weight=0.1)
```

`wf_loss` is retained only to match the COXNet training recipe. It is not a
claimed contribution of PRLDFC.

## Diagnostics

Training returns scalar monitors for:

- seed probability mass and threshold candidate count;
- eligible and unassigned GT counts;
- broadband/band attention entropy;
- low/mid/high routing weights and Thermal reliability;
- support ratio, overlap mass, raw residual RMS, and final delta ratio;
- residual-cap activation rate.

These values show whether the path is active. They do not prove semantic
correspondence or AP improvement.

## Controlled configs

Canonical P3/P4 treatment:

```bash
python tools/train.py configs/coxnet/prldfc/PRLDFC.py \
  --seed 0 --deterministic
```

P3-only level ablation:

```bash
python tools/train.py configs/coxnet/prldfc/PRLDFC_p3.py \
  --seed 0 --deterministic
```

Same-stage control without calibration:

```bash
python tools/train.py configs/coxnet/prldfc/same_stage_no_calibration.py \
  --seed 0 --deterministic
```

Run A/B/C comparisons at seed 0 before launching seeds 1 and 2. Structural
tests and a successful smoke run are not evidence of an AP gain.

## Implementation references

- [Design specification](superpowers/specs/2026-09-28-prldfc-design.md)
- [Implementation plan](superpowers/plans/2026-09-28-prldfc-implementation.md)
- Module: `mmdet/models/utils/prldfc.py`
- Integration: `mmdet/models/utils/fusion_strategy.py`
- Tests: `tests/test_models/test_utils/test_prldfc.py`
