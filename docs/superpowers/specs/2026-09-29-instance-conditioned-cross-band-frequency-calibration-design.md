# Instance-Conditioned Cross-Band Frequency Calibration for COXNet

## Objective

Replace only COXNet's CLFM computation with Instance-Conditioned Cross-Band
Frequency Calibration (ICBFC). The module uses Thermal-referenced object
instances to reason over all RGB/Thermal Haar-DWT sub-band pairs, reconstructs
an instance-specific RGB residual, and passes the calibrated RGB feature and
unchanged Thermal feature to the existing HOFM/AAM path.

The research question is:

> If different objects in one scene exhibit heterogeneous spectral responses,
> why should spectral interaction be shared across all instances?

ICBFC tests the answer by conditioning cross-modal frequency interaction on
individual object instances rather than global image statistics or a single
dense saliency map.

## Research Boundary

This is a new CLFM replacement, not a reproduction or extension stack of the
complete DyFCLT detector.

Included:

- fixed, orthonormal Haar DWT as a frequency-decomposition primitive;
- Thermal-guided class-agnostic instance separation;
- RGB and Thermal instance tokens for `LL`, `LH`, `HL`, and `HH`;
- all-to-all `4 x 4` cross-modal, cross-band relation modeling;
- instance-specific sparse-soft band routing;
- normalized instance-to-spatial reconstruction;
- one-way RGB calibration before the existing HOFM/AAM module.

Excluded from the first experiment:

- FFT and learned radial frequency boundaries;
- DyFCLT's SSE, IBS, FREF, and RT-DETR decoder;
- bidirectional RGB-to-Thermal calibration;
- prototype queues, contrastive learning, EDL, utility heads, and MoE;
- changes to the GFL head, QLSAssigner, NMS, dataset, or evaluation protocol.

Baseline invariants are RGB FPN `start_level=2`, Thermal FPN
`start_level=1`, four cross-stage pairs, learned RGB x2 DeConv, unchanged
Thermal features, HOFM/AAM, `wf_loss`, and the official detection head.

## Why the Instance Prior Uses Stride 4

The detector head begins at stride 8, but the known COXNet failure is strongest
for closely spaced objects. A one-channel stride-8 center map cannot reliably
represent two centers that quantize to the same cell. Therefore, ICBFC obtains
its instance prior from the Thermal backbone's stride-4 feature before the
Thermal FPN. This auxiliary feature is used only to locate and parameterize
instances; it is not added to the detection pyramid and does not change the
head stride.

The prior head predicts:

- a class-agnostic center heatmap;
- a two-dimensional sub-pixel center offset;
- a two-dimensional log-scale used to construct the instance support mask.

Center supervision uses Gaussian targets. Offset and scale losses are evaluated
only at GT center cells. Padding regions are excluded using `img_metas`.

## Candidate Semantics

### Training

GT boxes define the instance identities and continuous centers used for token
extraction. This prevents early random heatmaps from starving the calibration
branch. The predicted center, offset, and scale maps are trained against those
same instances through auxiliary losses.

The main detection loss still determines whether the frequency calibration is
useful. GT boxes supervise instance locations but never supervise a preferred
frequency band or relation matrix.

### Inference

Candidates are all stride-4 center-heatmap local maxima above a configured
score threshold. There is no fixed `top-100` or forced minimum candidate. The
implementation processes a variable number of candidates in chunks, so a
crowded image does not lose instances merely because it exceeded a global K.

Local peak suppression removes duplicate responses around one center. The
predicted sub-pixel offset preserves continuous coordinates, and the predicted
scale controls the support Gaussian. If no peak passes the threshold, ICBFC
returns the DeConv RGB identity feature unchanged.

Candidate recall, count, score distribution, and GT-to-candidate center recall
must be logged. The method cannot be called instance-aware if the candidate
stage merges nearby GT objects.

## Cross-Stage Data Flow

For each of four COXNet pyramid pairs, let `R` be the lower-resolution RGB FPN
feature and `T` the corresponding higher-resolution Thermal FPN feature.

1. Preserve the baseline learned cross-stage upsampling:

   `R_up = DeConv(R)`

2. Apply one-level orthonormal Haar DWT independently:

   `DWT(R_up) = {R_LL, R_LH, R_HL, R_HH}`

   `DWT(T)    = {T_LL, T_LH, T_HL, T_HH}`

3. Map each stride-4 Thermal instance center and scale continuously to the
   current DWT lattice. Extract Gaussian-weighted local tokens rather than a
   one-point sample:

   `r_i^b = Pool(R_b, M_i^b)`

   `t_i^b = Pool(T_b, M_i^b)`

   where `b` is one of `LL`, `LH`, `HL`, `HH`. Each instance therefore has
   eight frequency-modality tokens.

4. Model all-to-all Thermal-to-RGB band relations. RGB bands are queries and
   Thermal bands are keys and values:

   `A_i[b, a] = softmax_a(q(r_i^b) dot k(t_i^a) / sqrt(d) + E[b, a])`

   `c_i^b = sum_a A_i[b, a] v(t_i^a)`

   `A_i` has shape `4 x 4`. Its diagonal models same-band complementarity; its
   off-diagonal entries model relations such as Thermal `LL` to RGB `HH`.
   `E` is a learnable band-pair bias initialized with small non-zero noise so
   all routes are available without exact symmetry.

5. Produce one candidate correction for every RGB target band:

   `d_i^b = MLP([r_i^b, c_i^b, c_i^b - r_i^b])`

6. Keep relation and routing semantically separate. The relation matrix asks
   which Thermal band complements each RGB band. The instance router asks which
   resulting RGB-band corrections matter for this object:

   `g_i = sparsemax(router([R_i, C_i]))`

   `delta_i^b = g_i^b * d_i^b`

   Sparsemax supplies differentiable sparse-soft routing without a hard top-k
   band choice. No band is removed globally, and different instances may select
   different subsets.

7. Scatter the four correction tokens back to their DWT lattices using the
   instance Gaussian masks. At every spatial position, overlapping masks are
   normalized across instances:

   `Delta_b(x) = sum_i M_i^b(x) delta_i^b / (sum_i M_i^b(x) + eps)`

   Normalization prevents a dense group from receiving an arbitrarily larger
   residual solely because many supports overlap.

8. Reconstruct the spatial residual:

   `Delta_R = out_proj(IDWT(Delta_LL, Delta_LH, Delta_HL, Delta_HH))`

   `R_out = R_up + alpha * tanh(Delta_R)`

   The first experiment uses fixed `alpha=0.1`. `out_proj` uses Kaiming-normal
   initialization, not the previous `std=1e-3` initialization, so
   detection gradients can reach relation and routing parameters from the
   first update. The bounded residual protects the RGB identity path.

9. Pass `(R_out, T)` into the unchanged HOFM/AAM path. Thermal remains the
   spatial reference and is not calibrated by ICBFC.

## Losses

The first experiment adds only instance-prior supervision:

- `loss_icbfc_center`: CenterNet-style focal loss on the stride-4 Gaussian
  center heatmap;
- `loss_icbfc_offset`: masked L1 loss for sub-pixel center offsets;
- `loss_icbfc_scale`: masked smooth-L1 loss for log width and height.

The combined auxiliary loss weight is configurable and starts at `0.1`. No
band label, relation target, entropy target, or feature-alignment loss is added.
Relation, routing, and reconstruction must earn their use through the detector
loss. This keeps the causal comparison narrow.

Existing `wf_loss=True` and `kl_v2` remain unchanged for comparability with the
official COXNet baseline.

## Padding and Coordinate Contract

- GT boxes remain in resized-and-padded model-input coordinates.
- Continuous centers and sizes are divided by the actual stride of the raw
  Thermal prior feature, obtained from tensor and `batch_input_shape` sizes.
- Instance coordinates are scaled separately to every DWT lattice.
- Padding masks come from each sample's `img_shape` and `pad_shape` and are
  resized with nearest-neighbor semantics.
- Training and inference both pass `img_metas` through `extract_feat` so padded
  regions cannot create center candidates or enter instance pooling.
- A prediction outside the valid region is discarded before token extraction.

## Components and Interfaces

### `ThermalInstancePrior`

Input: raw stride-4 Thermal backbone feature.

Outputs:

- dense center logits, offsets, and log-scales;
- variable-length per-image instance descriptors containing continuous center,
  scale, score, and validity;
- auxiliary losses during training.

### `ICBFCLevel`

Input: one cross-stage `(Thermal, RGB)` FPN pair plus shared instances.

Responsibilities:

- RGB DeConv and shape validation;
- DWT of both modalities;
- per-instance, per-band token pooling;
- `4 x 4` relation attention;
- sparse-soft instance routing;
- normalized spatial reconstruction and IDWT;
- RGB residual calibration.

### `FusionLayer`

- Add a separate `use_clfm=['icbfc']` path.
- Own one shared `ThermalInstancePrior` and four `ICBFCLevel` modules.
- Preserve the baseline `v3` and experimental `dwt_dfca` paths unchanged.
- Return auxiliary ICBFC losses only during training.

### `FusionNetXO`

- Preserve the raw stride-4 Thermal backbone feature before applying `neck_t`.
- Pass it, GT boxes, and `img_metas` to `FusionLayer` for ICBFC.
- Override inference feature extraction so `img_metas` reaches the valid-mask
  and candidate path.
- Add ICBFC auxiliary losses to detector losses without treating diagnostics as
  optimized objectives.

### Configuration

Add `configs/coxnet/icbfc/ICBFC.py`, inheriting the official RGBTDronePerson
configuration and overriding only the CLFM method, ICBFC hyperparameters, and
work directory. The official baseline and DWT-DFCA configs remain unchanged.

## Diagnostics

Log the following values per level and in aggregate:

- `candidate_count` and `candidate_score_mean`;
- training GT candidate count and inference-style center recall;
- mean support ratio and overlap ratio;
- relation diagonal mass and off-diagonal mass;
- relation entropy;
- router entropy, active-band count, and per-instance router variance;
- per-band routing means, reported alongside variance rather than alone;
- residual-to-identity `delta_ratio`;
- gradient norms for relation Q/K/V, router, output projection, and DeConv in
  the GPU smoke test.

Activation requirements before training are:

- non-zero finite gradients in every new trainable path;
- a non-negligible initial `delta_ratio` without dominating `R_up`;
- at least two distinct instance routing vectors on a synthetic two-instance
  probe after a controlled optimization step;
- no exact, persistent uniform router caused solely by initialization;
- center candidates remain inside valid image regions.

## Verification Contract

Focused tests must prove:

1. Haar DWT/IDWT reconstructs even-sized CPU and CUDA tensors.
2. A stride-4 prior preserves two neighboring GT centers that are distinct at
   stride 4 but would collide at stride 8.
3. Candidate extraction returns a variable count, removes local duplicates,
   applies no fixed top-k truncation, and safely returns zero candidates.
4. Gaussian pooling produces eight tokens per instance with correct gradients.
5. Relation matrices have shape `N x 4 x 4`, rows are normalized, and
   off-diagonal paths affect output.
6. Sparsemax routing is finite, sums to one, can produce exact zeros, and
   remains differentiable on active routes.
7. Overlapping reconstruction masks are normalized instead of summed without
   bound.
8. Identity output is exact when there are no candidates.
9. Thermal features are unchanged and RGB output matches Thermal resolution.
10. Padding never generates candidates or contributes to token pooling.
11. Center, offset, scale, relation, router, output projection, DeConv, and
    input tensors receive finite gradients in the appropriate paths.
12. The config builds four ICBFC levels while preserving FPN start levels,
    HOFM/AAM, `wf_loss`, GFL, QLSAssigner, and test settings.
13. The official baseline and DWT-DFCA configs remain unchanged.

Verification proceeds through focused CPU tests, `compileall`, config/model
construction, and one real RGBTDronePerson GPU forward/backward smoke batch.
The existing active DWT-DFCA seed queue must not be stopped or changed by this
work.

## Controlled Experiment Plan

The minimum causal comparison is:

1. official COXNet baseline;
2. current image-level DWT-DFCA;
3. ICBFC without off-diagonal relations, retaining only same-band interaction;
4. full ICBFC with all-to-all `4 x 4` relations;
5. full ICBFC with a shared image-level router instead of instance routing.

Seed 0 is evaluated first. Seeds 1 and 2 are justified only after seed 0 shows
that candidates, relation, routing, and residual are active and the target
metrics do not materially regress. Besides AP, report tiny-scale AP, crowded
size-controlled localization error, center-candidate recall, and merged-object
failure rate.

## Success and Falsification Criteria

Implementation success means the focused verification contract passes and the
module is demonstrably active. It does not mean the research hypothesis is
validated.

The instance-conditioned hypothesis is weakened if:

- stride-4 candidate recall is poor for crowded GT objects;
- different instances converge to effectively identical relation/router
  distributions;
- off-diagonal interaction does not outperform same-band-only interaction;
- an image-level router matches full ICBFC;
- localization and merged-object metrics remain unchanged despite active
  calibration;
- gains disappear under matched seeds or are limited to one scale bucket while
  overall precision degrades.

The method is considered promising only if the instance path is active,
improves the targeted dense/tiny failure modes, and does not obtain apparent
gains from changed detector assignment, postprocessing, or evaluation settings.
