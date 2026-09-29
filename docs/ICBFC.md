# Instance-Conditioned Cross-Band Frequency Calibration

ICBFC is an experimental replacement for COXNet's Cross-Layer Fusion Module
(CLFM). It preserves COXNet's cross-stage FPN structure and RGB DeConv, changes
only the RGB feature before AAM/HOFM, and leaves the Thermal feature unchanged.

## 1. Motivation and scope

Global or dense spatial frequency fusion shares one spectral aggregation rule
across every object in an image. That is restrictive for RGBT drone scenes:
neighboring tiny people can differ in scale, local background, modality
visibility, and edge response. ICBFC instead formulates spectral aggregation
as an instance-conditioned routing problem:

> If different objects have heterogeneous spectral responses, their
> cross-modal frequency interactions should not be forced to share one route.

ICBFC does not claim that Haar DWT is novel. DWT supplies four fixed analysis
bands. The studied mechanism is the object-specific tokenization, all-to-all
cross-band relation, sparse routing, and normalized instance-to-spatial
reconstruction. ICBFC is not the complete DyFCLT architecture and does not use
its FFT transformer stack.

## 2. Preserved COXNet contract

The canonical config is [`configs/coxnet/icbfc/ICBFC.py`](../configs/coxnet/icbfc/ICBFC.py).
It inherits the official RGBTDronePerson baseline and preserves:

- RGB FPN `start_level=2` and Thermal FPN `start_level=1`;
- four cross-stage pairs and the original x2 RGB DeConv;
- unchanged Thermal values entering AAM/HOFM;
- AAM/HOFM, `wf_loss=True` with `kl_v2`, GFLQ, and QLSAssigner;
- output strides `[8, 16, 32, 64]`, score filtering, NMS IoU `0.3`, and
  `max_per_img=100`.

The raw Thermal backbone stride-4 feature is used only by the instance-prior
head. It is not added to the detector pyramid or prediction head.

## 3. Data flow

For cross-stage level `l`, let `R_{l+1}` be the lower-resolution RGB feature
and `T_l` the Thermal feature. The retained DeConv gives

```text
R_up = DeConv(R_{l+1}),       shape(R_up) = shape(T_l).
```

The complete route is:

```text
raw Thermal stride-4 -> center/offset/log-scale prior -> N instances

R_up, T_l -> Haar DWT -> four RGB + four Thermal band maps
           -> instance Gaussian pooling -> eight tokens per instance
           -> 4x4 Thermal-to-RGB relation
           -> sparse four-band router
           -> normalized Gaussian scatter -> IDWT
           -> bounded RGB residual -> unchanged AAM/HOFM
```

### 3.1 Thermal instance prior

A shared `3x3 Conv + GroupNorm + ReLU` stem predicts a class-agnostic center
heatmap, sub-cell offset, and log width/height at stride 4. During training,
continuous GT geometry supplies the routing instances so close GT centers that
would collide at stride 8 can remain distinct. Dense predictions still receive
the auxiliary prior losses.

At inference, the model uses valid 3x3 local maxima above a configurable score
threshold. There is no fixed top-k. Every valid maximum is retained and
processed in bounded-memory chunks. A zero-candidate image follows the exact
DeConv RGB identity path.

`img_shape`, `pad_shape`, and `batch_input_shape` define valid masks in both
training and inference. Padded cells cannot create candidates and are removed
from token pooling.

### 3.2 Instance band tokens

One-level orthonormal Haar DWT produces four bands for both modalities:

```text
B_R = {R_LL, R_LH, R_HL, R_HH}
B_T = {T_LL, T_LH, T_HL, T_HH}.
```

For instance `i`, a Gaussian support `M_i` is derived from its continuous
center and scale on the band lattice. Weighted pooling yields

```text
r_i^b = sum_x M_i(x) R_b(x) / sum_x M_i(x)
t_i^a = sum_x M_i(x) T_a(x) / sum_x M_i(x),
```

where `a,b` index the four Thermal and RGB bands. Each instance therefore has
eight modality-band tokens.

### 3.3 All-to-all cross-band relation

For every RGB target band `b` and Thermal source band `a`, ICBFC computes

```text
A_i[b,a] = softmax_a(q(r_i^b) k(t_i^a)^T / sqrt(d) + B[b,a]),
c_i^b    = sum_a A_i[b,a] v(t_i^a).
```

`A_i` is a normalized `4 x 4` matrix. It allows off-diagonal paths such as
Thermal LL to RGB high-frequency bands; it does not assume that equal-named
bands are always the correct complement.

The correction MLP consumes `[r_i^b, c_i^b, c_i^b-r_i^b]`. A separate
instance router predicts four logits and applies sparsemax:

```text
g_i = sparsemax(router([R_i, C_i])),       sum_b g_i^b = 1.
delta_i^b = g_i^b * MLP([r_i^b, c_i^b, c_i^b-r_i^b]).
```

The relation asks which Thermal band can complement each RGB target band. The
router asks which corrected RGB bands matter for this particular object.

### 3.4 Normalized spatial reconstruction

Instance corrections are projected back without overlap amplification:

```text
Delta_b(x) = sum_i M_i(x) delta_i^b / max(sum_i M_i(x), eps).
```

After inverse DWT and a non-zero Kaiming-initialized output projection,

```text
R_out = R_up + 0.1 * tanh(Conv1x1(IDWT(Delta))).
```

The fixed residual scale limits the initial intervention. `R_out` and the
original `T_l` then enter the existing AAM/HOFM.

## 4. Optimization

The prior is supervised with:

- CenterNet-style Gaussian center focal loss;
- masked L1 sub-cell offset loss;
- masked smooth-L1 log-scale loss.

The canonical `aux_loss_weight=0.1` multiplies each emitted prior term.
Relation, router, and reconstruction have no hand-authored band labels or
alignment loss. They learn from the detector objective. The existing COXNet
`wf_loss` remains enabled for baseline comparability.

Only keys beginning with `loss_icbfc_` are added to the optimized objective.
All `icbfc_*` diagnostics are detached logger values.

## 5. Diagnostics

The implementation reports:

- candidate count, candidate score mean, and center recall at the routing
  threshold;
- support and overlap ratios;
- relation diagonal/off-diagonal mean and entropy;
- router entropy, active-band count, per-instance variance, and LL/LH/HL/HH
  mean weights;
- residual-to-DeConv identity `delta_ratio`;
- per-level candidate/support/overlap/residual values.

[`tools/misc/smoke_icbfc.py`](../tools/misc/smoke_icbfc.py) additionally
requires finite, non-zero gradients for the prior center/offset/scale heads and
every level's relation Q/K/V, router, output projection, and DeConv.

## 6. Reproducibility

Focused tests:

```bash
python -m unittest \
  tests.test_models.test_utils.test_icbfc \
  tests.test_models.test_utils.test_icbfc_fusion \
  tests.test_models.test_detectors.test_icbfc_config \
  tests.test_tools.test_icbfc_launcher -v
```

Real-batch smoke:

```bash
export MMDET_DATASETS=/absolute/path/to/RGBTDronePerson/
CUDA_VISIBLE_DEVICES=0 python tools/misc/smoke_icbfc.py \
  --config configs/coxnet/icbfc/ICBFC.py
```

Fresh GPU-0 seeds:

```bash
bash tools/run_icbfc_seeds_gpu0.sh
```

The launcher uses `/tmp/coxnet_icbfc_gpu0.lock`, runs seeds `0,1,2` in order,
uses distinct work directories, and intentionally omits `--auto-resume`.

## 7. Falsification criteria and controls

The design is weakened if any of the following occurs:

- inference center recall remains low or valid candidates collapse to zero;
- all instances learn effectively identical relation matrices or router
  vectors;
- `delta_ratio` collapses to zero, showing that the detector ignores ICBFC;
- `delta_ratio` saturates near the residual cap or background false positives
  rise, showing over-transfer;
- off-diagonal relations can be removed without changing outputs or metrics;
- a DeConv-only cross-stage control matches or exceeds ICBFC;
- gains occur only for tiny1 while tiny2/tiny3/small or crowded localization
  degrade;
- the three-seed result does not improve on matched COXNet reruns.

Required evaluation therefore includes overall/tiny size splits, dense-object
localization, per-seed variance, DeConv-only ablation, diagonal-only relation,
shared image-level router, and GT-versus-predicted instance prior analysis.

## 8. Current limitations

- Training routing uses GT instances while inference routing depends on the
  learned Thermal prior; this creates a measurable train/inference gap.
- A weak Thermal object can still be missed by the Thermal-first prior.
- Gaussian pooling compresses local band maps to one token per band and can
  discard fine spatial structure.
- Sparse routing is learned indirectly through detection loss and is not proof
  of semantic band correspondence.
- Fixed Haar bands may not be optimal for every object or sensor pair.
- Passing mechanism tests proves implementation behavior, not detection
  accuracy. AP and efficiency claims remain pending completed checkpoints.
