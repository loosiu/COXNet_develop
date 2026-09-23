# OEPC-v2 Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace COXNet CLFM in the final experiment with a P3-only Thermal-guided Object-centric Calibration module that finds Thermal objects, associates them locally to RGB, injects a bounded direct RGB residual, and learns one route gate from candidate-local GFL utility.

**Architecture:** RGB and Thermal FPNs produce same-stage P3-P6 features. A new `ThermalGuidedObjectCalibration` consumes same-stage P3 plus P4 semantic context, generates only Thermal candidates, maps their detail-preserving descriptors into nearby RGB positions, and returns calibrated RGB P3 while leaving Thermal P3 unchanged for the existing AAM/HOFM. Legacy `oepc.py` remains untouched for balanced-experiment reproducibility.

**Tech Stack:** Python 3, PyTorch, MMCV/MMDetection 2.x, pytest, COXNet `FusionNetXO`, GFLQ head, `coxmamba` conda environment.

**Spec:** `docs/superpowers/specs/2026-09-23-oepc-v2-design.md`

## Global Constraints

- Work only in `/data/siwoo/COXNet-OEPC-v2` on `oepc-v2-main`; do not edit `/data/siwoo/COXNet-OEPC-balanced-core`.
- Preserve `mmdet/models/utils/oepc.py` and `configs/coxnet/oepc/OEPC_balanced_core.py` behavior.
- Final config: both FPNs `start_level=1`, `use_clfm=[]`, `use_trpc=False`, `use_oepc=False`, `use_oepc_v2=True`.
- Apply v2 only to P3. P4 provides descriptor context only; original Thermal P3 enters AAM/HOFM unchanged.
- Add no DWT, IDWT, DeConv, feature warp, EDL, global alignment, prototype diversity, or hard identity matching.
- Final auxiliary weights: candidate `0.10`, foreground `0.05`, local contrast `0.02`, utility `0.05`.
- Do not launch seeds 1/2 before seed 0 has completed and been evaluated.
- Push to `origin/main` by fast-forward only; never force-push.

## Review Focus

- Odd padded sizes: same-level modality shapes must match, P4 must be smaller than P3, and padding must not enter descriptors or support.
- Dense adjacent people: inference retains adjacent candidates and overlap aggregation cannot amplify residual magnitude.
- Empty scenes/no eligible candidates: losses stay finite, utility skips cleanly, and zero-support RGB stays unchanged.
- Local utility with zero positives: classification-only loss is finite and box/DFL/centerness contributions are zero.
- Mixed precision/near-zero norms: attention, aggregation, and RMS cap avoid NaN/Inf and respect the bound.

---

### Task 1: Add normalized Gaussian targets and descriptor primitives

**Files:**
- Create: `mmdet/models/utils/oepc_v2.py`
- Create: `tests/test_models/test_utils/test_oepc_v2.py`
- Modify: `mmdet/models/utils/__init__.py`

**Interfaces:**
- Produces `build_gaussian_center_targets(gt_boxes, padded_size, feat_size, device, valid_mask=None, max_radius=2) -> Tensor[N,1,H,W]`.
- Produces `ThermalGuidedObjectCalibration` and `_build_descriptor(feature, semantic, foreground_probability, valid, ...) -> dict`.

- [ ] **Step 1: Write failing Gaussian tests**

Test two nearby fractional-center boxes, per-object maximum `1`, max overlap rather than sum, zero padding, and empty GT.

```python
def test_gaussian_centers_are_normalized_and_padding_is_zero():
    boxes = [torch.tensor([[15., 11., 24., 20.], [24., 11., 33., 20.]])]
    valid = torch.ones(1, 1, 8, 10, dtype=torch.bool)
    valid[:, :, -1] = False
    target = build_gaussian_center_targets(
        boxes, (64, 80), (8, 10), torch.device('cpu'), valid)
    assert torch.count_nonzero(target == 1.0) >= 2
    assert target.min().item() >= 0.0
    assert target.max().item() <= 1.0
    assert torch.count_nonzero(target[:, :, -1]) == 0
```

- [ ] **Step 2: Verify the test fails**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py::test_gaussian_centers_are_normalized_and_padding_is_zero -q`

Expected: FAIL because `oepc_v2` does not exist.

- [ ] **Step 3: Implement normalized Gaussian targets**

Use float P3 centers, `radius=clamp(ceil(0.5*sqrt(wf*hf)),1,2)`, `sigma=(2r+1)/6`, normalize each object-local Gaussian window by its maximum, then merge with `torch.maximum`. Validate `valid_mask.shape == target.shape` before masking.

```python
gaussian = torch.exp(-((xx - cx).square() + (yy - cy).square()) /
                     (2.0 * sigma * sigma))
gaussian = gaussian / gaussian.max().clamp_min(1e-12)
target[b, 0, y0:y1, x0:x1] = torch.maximum(
    target[b, 0, y0:y1, x0:x1], gaussian)
```

- [ ] **Step 4: Write failing descriptor tests**

Test P4 influence, same-level validation, masked averaging, foreground-weight stop-gradient, and false context availability when foreground fills the ring.

```python
first = module._build_descriptor(p3, p4, foreground, valid, **rgb_ops)
second = module._build_descriptor(p3, p4 + 1.0, foreground, valid, **rgb_ops)
assert not torch.equal(first['descriptor'], second['descriptor'])
first['contrast'].sum().backward()
assert module.rgb_foreground_head[-1].weight.grad is None
```

- [ ] **Step 5: Verify descriptor tests fail**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'descriptor or context' -q`

Expected: FAIL because descriptor methods are missing.

- [ ] **Step 6: Implement detail/context/P4 descriptor**

Create separate RGB/Thermal P3 projections, P4 projections, foreground heads, descriptor fusions, and `ChannelLayerNorm`s. Use mask-normalized 3x3 averaging and 7x7 ring context.

```python
base = base_norm(base_projection(feature))
detail = base - self._valid_average(base, valid, kernel_size=3)
context, coverage = self._ring_context(
    base, 1.0 - foreground_probability.detach(), valid)
available = coverage >= self.min_context_coverage
contrast = (base - context) * available.to(base.dtype)
semantic_feature = F.interpolate(
    semantic_norm(semantic_projection(semantic)),
    size=feature.shape[-2:], mode='bilinear', align_corners=False)
descriptor = descriptor_norm(descriptor_fusion(torch.cat(
    [base, detail, contrast, semantic_feature], dim=1)))
```

Require equal RGB/Thermal P3 shapes, equal RGB/Thermal P4 shapes, and P4 smaller than P3 in both dimensions.

- [ ] **Step 7: Export and verify Task 1**

Add `ThermalGuidedObjectCalibration` to `mmdet/models/utils/__init__.py`.

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'gaussian or descriptor or context' -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mmdet/models/utils/oepc_v2.py mmdet/models/utils/__init__.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Add OEPC-v2 detail descriptors and Gaussian targets"
```

### Task 2: Implement differentiable support and Thermal-to-RGB association

**Files:**
- Modify: `mmdet/models/utils/oepc_v2.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- Produces `_candidate_support(logits, valid)`: soft during training, threshold/top-K during evaluation.
- Produces `_local_attention(thermal_descriptor, rgb_descriptor, valid)` and `_aggregate_to_rgb(...)`.

- [ ] **Step 1: Write failing support tests**

Test non-zero candidate-logit gradient in training, adjacent above-threshold cells surviving evaluation, top-K cap, padding, no eligible cells, no RGB candidate head, and candidate-logit invariance when only RGB changes.

```python
module.train()
support = module._candidate_support(logits, valid)
support.sum().backward()
assert torch.count_nonzero(logits.grad) > 0
module.eval()
support = module._candidate_support(two_adjacent_high_logits, valid)
assert support[0, 0, 2, 2] > 0 and support[0, 0, 2, 3] > 0
```

- [ ] **Step 2: Verify support tests fail**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k support -q`

Expected: FAIL because `_candidate_support` is missing.

- [ ] **Step 3: Implement Thermal-only candidate/support**

Instantiate only `thermal_candidate_head`, initialized to `logit(0.01)`. Defaults are threshold `0.10`, logit temperature `0.5`, top-K `100`. Training uses:

```python
threshold_logit = math.log(tau / (1.0 - tau))
support = torch.sigmoid(
    (candidate_logits - threshold_logit) / self.support_logit_temperature)
support = support * valid.to(support.dtype)
```

Evaluation masks values below threshold, applies per-image top-K directly, and never uses local-max pooling.

- [ ] **Step 4: Write failing association/overlap tests**

Test distance-prior center preference, Thermal-query/RGB-key direction, normalized overlap average, support cap `1`, padding, and finite extreme-value outputs.

```python
fields = module._aggregate_to_rgb(values, support, attention, valid)
assert torch.allclose(fields['condition'][:, :, 1, 1],
                      torch.full((1, 4), 3.0))
assert fields['support'][0, 0, 1, 1] == 1.0
```

- [ ] **Step 5: Implement local attention and normalized scatter**

Use normalized Thermal descriptor queries and unfolded RGB keys in radius `2`; divide dot products by `sqrt(embed_dim) * temperature`, subtract distance prior, and mask invalid RGB options. Fold weighted values and weights back to RGB coordinates:

```python
condition = folded_weighted_value / folded_weight.clamp_min(1e-6)
rgb_support = folded_weight.clamp(0.0, 1.0)
```

Map candidate probability, Thermal foreground, and context reliability with the same normalized weights. Return confidence and normalized entropy.

- [ ] **Step 6: Verify and commit Task 2**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'support or association or overlap or distance' -q
git add mmdet/models/utils/oepc_v2.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Add differentiable thermal association support"
```

### Task 3: Add direct residual, one router, and minimal auxiliary losses

**Files:**
- Modify: `mmdet/models/utils/oepc_v2.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- Produces `forward(rgb, thermal, rgb_semantic, thermal_semantic, valid_mask=None, center_target=None, foreground_target=None, return_aux=False)`.
- Aux loss keys: `candidate_loss`, `foreground_loss`, `contrastive_loss`; no EDL key.
- Utility payload keys: `batch_index`, `rgb_keep`, `rgb_trial`, `thermal`, `route_prediction`, `residual_penalty`, `local_mask`.

- [ ] **Step 1: Write failing residual/router tests**

Assert no FiLM/evidence modules, small non-zero final residual weights, zero router bias, exact identity at zero support, `0.2` RMS cap, unchanged Thermal tensor, and non-zero finite detection gradient to the candidate head.

```python
assert not hasattr(module, 'film_out')
assert not any('evidence' in name for name, _ in module.named_modules())
assert torch.count_nonzero(module.residual_out.weight) > 0
assert torch.count_nonzero(module.router[-1].bias) == 0
```

- [ ] **Step 2: Verify residual tests fail**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'residual or router or thermal_unchanged' -q`

Expected: FAIL because forward calibration is incomplete.

- [ ] **Step 3: Implement direct residual and one gate**

```python
interaction = torch.cat([
    rgb_descriptor, thermal_condition,
    thermal_condition - rgb_descriptor,
    thermal_condition * rgb_descriptor], dim=1)
raw_delta = self.residual_out(F.gelu(self.residual_in(interaction)))
router_input = torch.cat([
    rgb_descriptor, thermal_condition,
    (thermal_condition - rgb_descriptor).abs(),
    mapped_probability, rgb_foreground, mapped_thermal_foreground,
    match_confidence, match_entropy, context_reliability], dim=1)
route = torch.sigmoid(self.router(router_input))
delta_zero = rgb_support * route * raw_delta
delta = self._cap_residual(delta_zero, rgb)
output = rgb + delta.to(rgb.dtype)
```

Use `Conv1x1(4*embed_dim -> hidden)`, GELU, `Conv3x3(hidden -> channels)`. Initialize the last weight `Normal(0,1e-2)`, bias 0. Router input is `3*embed_dim+6`, output one channel, final weight/bias zero for initial route `0.5`.

- [ ] **Step 4: Implement minimal losses and diagnostics**

Use candidate focal, two foreground focal terms, and only local object-vs-safe-background contrast. Another foreground person is never a negative. Return candidate probability/max/pass count, soft support mass/ratio, hard count, center recall, context coverage/availability, attention confidence/entropy, route mean/on-support, raw residual RMS, delta ratio/cap ratio, and utility sampling metrics. Empty valid sets return finite zeros.

For local utility, detach the Thermal probability used for discrete sampling, restrict eligibility to valid cells above `candidate_threshold`, and sample one cell proportional to its score. Split the differentiable Thermal support into `sample_support = support * sample_mask` and `other_support = support * (1 - sample_mask)`, then recompute sample-only and other-candidate association fields. Build `rgb_keep` from all non-sampled corrections and `rgb_trial` by adding the sampled candidate's bounded residual with route forced to `1`. Use the sample association support as `local_mask`, the support-weighted mean route as `route_prediction`, and support-weighted raw residual RMS as `residual_penalty`. If there is no eligible candidate, return `utility_payload=None` and `utility_sampled=0`.

```python
eligible_score = (probability.detach() * valid) * (
    probability.detach() >= self.candidate_threshold)
sample_mask, sample_batch = self._sample_one_candidate(eligible_score)
sample_support = thermal_support * sample_mask
other_support = thermal_support * (1.0 - sample_mask)
sample_fields = self._fields_for_support(sample_support, descriptors, attention)
other_fields = self._fields_for_support(other_support, descriptors, attention)
local_mask = sample_fields['support'] > 0
utility_payload = dict(
    batch_index=sample_batch,
    rgb_keep=(rgb + other_delta)[sample_batch:sample_batch + 1],
    rgb_trial=(rgb + self._cap_residual(
        other_delta + sample_raw_delta * sample_fields['support'], rgb)
    )[sample_batch:sample_batch + 1],
    thermal=thermal[sample_batch:sample_batch + 1],
    route_prediction=_masked_mean(
        route[sample_batch:sample_batch + 1],
        local_mask[sample_batch:sample_batch + 1]),
    residual_penalty=_masked_mean(
        sample_raw_delta[sample_batch:sample_batch + 1].square().mean(1, True),
        local_mask[sample_batch:sample_batch + 1]),
    local_mask=local_mask[sample_batch:sample_batch + 1])
```

- [ ] **Step 5: Run full module forward/backward tests**

Backpropagate output detection surrogate plus all three auxiliary losses. Require non-zero finite gradients for P3/P4 projections, candidate and foreground heads, descriptor fusions, attention, direct residual, router, and RGB/Thermal inputs. Separately prove the context weighting does not update the foreground head.

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mmdet/models/utils/oepc_v2.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Implement bounded OEPC-v2 RGB calibration"
```

### Task 4: Add candidate-local fixed-assignment GFL loss

**Files:**
- Modify: `mmdet/models/dense_heads/gflq_head.py`
- Modify: `tests/test_models/test_dense_heads/test_gfl_head.py`

**Interfaces:**
- Produces `sampled_local_detector_loss(cls_score, bbox_pred, centerness, target_cache, level, batch_index, local_mask) -> scalar Tensor`.
- Existing `sampled_detector_loss` remains unchanged for legacy OEPC.

- [ ] **Step 1: Write failing local-mask tests**

Build a small GFL head and target cache. Changing outputs outside the mask must leave local loss unchanged; changing an inside logit must change it. Also test a wrong mask shape and a mask with zero local positives.

```python
base = head.sampled_local_detector_loss(
    cls, bbox, center, cache, 0, 0, mask)
outside = cls.clone()
outside[:, :, 0, 0] += 50.0
assert torch.allclose(base, head.sampled_local_detector_loss(
    outside, bbox, center, cache, 0, 0, mask))
inside = cls.clone()
inside[:, :, 4, 4] += 50.0
assert not torch.allclose(base, head.sampled_local_detector_loss(
    inside, bbox, center, cache, 0, 0, mask))
```

- [ ] **Step 2: Verify local tests fail**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_dense_heads/test_gfl_head.py -k local -q`

Expected: FAIL because the method is absent.

- [ ] **Step 3: Implement local target filtering**

Select one image/level from the existing cache. Validate `local_mask` as `[1,1,H,W]`. Flatten row-major and repeat by `num_base_priors`. Outside the mask set `label_weights=0` and labels to `num_classes`, excluding those anchors from both classification and positive regression sets.

```python
local = local_mask.to(dtype=torch.bool, device=labels.device).flatten(1)
if local.shape[1] * self.num_base_priors != labels.shape[1]:
    raise ValueError('local_mask does not match the sampled GFL level')
local = local.repeat_interleave(self.num_base_priors, dim=1)
local_labels = labels.clone()
local_weights = label_weights.clone()
local_labels[~local] = self.num_classes
local_weights[~local] = 0
positive = local & (local_labels >= 0) & (local_labels < self.num_classes)
num_samples = max(float(positive.sum().item()), 1.0)
```

Call `loss_single` with cached anchors/bbox targets and locally filtered labels/weights. Use the existing bbox/DFL normalization. With no local positives, return finite classification-only loss.

- [ ] **Step 4: Run the full GFL test file**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_dense_heads/test_gfl_head.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmdet/models/dense_heads/gflq_head.py tests/test_models/test_dense_heads/test_gfl_head.py
git commit -m "Add candidate-local GFL utility loss"
```

### Task 5: Integrate OEPC-v2 Final into FusionLayer and FusionNetXO

**Files:**
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- Adds `use_oepc_v2=False, oepc_v2_cfg=None` to `FusionLayer` and `FusionNetXO`.
- Exactly one of TRPC, legacy OEPC, or v2 may be active; an active replacement requires `use_clfm=[]`.
- P3 v2 call receives `rgb_semantic=v_feats[1]` and `thermal_semantic=t_feats[1]`.
- V2 utility payload adds `utility_kind='candidate_local'` and `local_mask`.

- [ ] **Step 1: Write failing fusion tests**

Test mutual exclusion, P3/P4 flow, original Thermal entering HOFM, P4 affecting the descriptor but not directly replacing P3, no DWT/DeConv, legacy OEPC construction, v2 loss weights, and padding masks.

```python
layer = FusionLayer(
    in_channels=16, num_layers=2, fs_type='fusionnet-xo',
    use_clfm=[], use_trpc=False, use_oepc=False, use_oepc_v2=True,
    oepc_v2_cfg=dict(apply_levels=(0,), embed_dim=8,
                     candidate_loss_weight=0.1,
                     foreground_loss_weight=0.05,
                     contrastive_loss_weight=0.02,
                     utility_loss_weight=0.05), usepoolup=[])
assert set(layer.oepc_v2_layers) == {'0'}
assert not any('dwt' in n.lower() or 'deconv' in n.lower()
               for n, _ in layer.oepc_v2_layers.named_modules())
```

- [ ] **Step 2: Verify integration tests fail**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'fusion or legacy' -q`

Expected: FAIL because no v2 integration exists.

- [ ] **Step 3: Add v2 construction and P3/P4 forwarding**

Import the v2 module and target builder separately. Validate replacement flags, instantiate configured levels without altering downstream HOFM RNG state, require level 1 context for P3, and build Gaussian/box targets from the same image geometry.

```python
v_feat, aux_i = self.oepc_v2_layers[str(i)](
    v_feat, t_feat, v_feats[i + 1], t_feats[i + 1],
    valid_mask=valid_mask, center_target=center_target,
    foreground_target=foreground_target, return_aux=True)
```

Call HOFM with calibrated `v_feat` and the untouched `t_feat`. Build keep/trial HOFM features from the utility payload and carry `local_mask` and `utility_kind` forward.

- [ ] **Step 4: Add candidate-local detector dispatch**

Include v2 in the metadata-aware `extract_feat` branch. When `utility_kind == 'candidate_local'`, run both v2 keep/trial head forwards and local losses under `torch.no_grad()` with the same target cache and local mask. Use their detached difference and penalty to build the utility target; add only weighted router BCE, not `loss_oepc_trial_det`.

```python
keep_detection = self.bbox_head.sampled_local_detector_loss(
    *keep_outputs, target_cache, level, batch_index, local_mask)
trial_detection = self.bbox_head.sampled_local_detector_loss(
    *trial_outputs, target_cache, level, batch_index, local_mask)
utility_target = torch.sigmoid(
    (keep_detection.detach() - trial_detection.detach()
     - penalty_weight * penalty.detach()) / utility_temperature).detach()
losses['loss_oepc_v2_utility'] = utility_weight * F.binary_cross_entropy(
    route_prediction.float().clamp(1e-6, 1.0 - 1e-6), utility_target)
```

Keep the legacy OEPC global utility and trial-detection behavior unchanged.

- [ ] **Step 5: Add backward/simple-test regression tests**

Assert local keep/trial values are finite, utility BCE updates the router, no v2 trial-loss key exists, `simple_test` forwards `img_metas`, and padded rows produce zero support.

- [ ] **Step 6: Run v2 and legacy suites**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py tests/test_models/test_utils/test_oepc.py tests/test_models/test_utils/test_trpc.py tests/test_models/test_dense_heads/test_gfl_head.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mmdet/models/utils/fusion_strategy.py mmdet/models/detectors/fusionnet_xo.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Integrate OEPC-v2 with local detector utility"
```

### Task 6: Add the final controlled configuration

**Files:**
- Create: `configs/coxnet/oepc/OEPC_v2_final.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- Canonical config: `configs/coxnet/oepc/OEPC_v2_final.py`.
- Canonical seed-0 directory: `work_dir/coxmamba/rgbtdroneperson/oepc_v2_final/seed0`.

- [ ] **Step 1: Write a failing config-contract test**

```python
cfg = mmcv.Config.fromfile('configs/coxnet/oepc/OEPC_v2_final.py')
assert cfg.model.neck.start_level == 1
assert cfg.model.neck_t.start_level == 1
assert cfg.model.use_clfm == []
assert cfg.model.use_trpc is False and cfg.model.use_oepc is False
assert cfg.model.use_oepc_v2 is True
assert cfg.model.oepc_v2_cfg.apply_levels == (0,)
assert 'edl_loss_weight' not in cfg.model.oepc_v2_cfg
assert cfg.model.oepc_v2_cfg.utility_loss_weight == 0.05
```

- [ ] **Step 2: Verify the config test fails**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k config_contract -q`

Expected: FAIL because the config is absent.

- [ ] **Step 3: Create the fixed final config**

```python
model = dict(
    neck=dict(start_level=1), neck_t=dict(start_level=1),
    use_clfm=[], use_trpc=False, use_oepc=False, use_oepc_v2=True,
    wf_loss=False,
    oepc_v2_cfg=dict(
        apply_levels=(0,), embed_dim=64, context_kernel=7,
        min_context_coverage=0.25, gaussian_max_radius=2,
        search_radius=2, search_temperature=0.2,
        distance_prior_weight=0.1, candidate_prior=0.01,
        candidate_threshold=0.10, support_logit_temperature=0.5,
        max_candidates=100, residual_scale=0.2,
        residual_init_std=1e-2, focal_gamma=2.0,
        candidate_loss_weight=0.10, foreground_loss_weight=0.05,
        contrastive_loss_weight=0.02, utility_loss_weight=0.05,
        utility_temperature=0.01, utility_penalty_weight=0.01))
work_dir = 'work_dir/coxmamba/rgbtdroneperson/oepc_v2_final/seed0'
```

- [ ] **Step 4: Build the config/model and run the contract test**

```bash
conda run -n coxmamba python -c "from mmcv import Config; from mmdet.models import build_detector; c=Config.fromfile('configs/coxnet/oepc/OEPC_v2_final.py'); m=build_detector(c.model, train_cfg=c.get('train_cfg'), test_cfg=c.get('test_cfg')); print(type(m.fuse_layer.oepc_v2_layers['0']).__name__); assert not hasattr(m.fuse_layer, 'idwt_layers')"
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k config_contract -q
```

Expected: `ThermalGuidedObjectCalibration`, then PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/coxnet/oepc/OEPC_v2_final.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Add final OEPC-v2 experiment config"
```

### Task 7: Update repository documentation

**Files:**
- Modify: `README.md`
- Create: `docs/OEPC_V2_FINAL.md`

**Interfaces:**
- Documents the canonical config, architecture, losses, diagnostics, command, and legacy-method distinction.

- [ ] **Step 1: Update the README primary method**

Make OEPC-v2 Final/TOC the primary method and include:

```text
Thermal P3/P4 -> Thermal candidates -> local RGB association
-> bounded direct RGB residual -> original AAM/HOFM/GFL
```

State P3-only calibration, P4 descriptor context only, Thermal unchanged, no CLFM/DWT/IDWT/DeConv/EDL, legacy OEPC/TRPC retention, and no AP claim before controlled training.

- [ ] **Step 2: Add detailed final-method documentation**

Document equations/defaults, auxiliary weights, local GFL utility mask behavior, diagnostics, output directory, and seed-0 command:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/coxnet/oepc/OEPC_v2_final.py --seed 0 --deterministic
```

State that seeds 1/2 are not launched until seed 0 evaluation.

- [ ] **Step 3: Cross-check docs with config/code**

```bash
rg -n "OEPC-v2|ThermalGuidedObjectCalibration|OEPC_v2_final|EDL|DWT|DeConv|seed 0" README.md docs/OEPC_V2_FINAL.md configs/coxnet/oepc/OEPC_v2_final.py
git diff --check
```

Expected: paths/defaults agree; EDL only appears in statements that it is absent.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/OEPC_V2_FINAL.md
git commit -m "Document final thermal-guided OEPC-v2"
```

### Task 8: Full verification, smoke training, review, and main push

**Files:**
- Verify every file changed by Tasks 1-7.
- Do not change source for a verification failure until a failing regression test exists.

**Interfaces:**
- Produces a verified clean commit series and fast-forwards `origin/main`.

- [ ] **Step 1: Run focused tests**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py tests/test_models/test_utils/test_oepc.py tests/test_models/test_utils/test_trpc.py tests/test_models/test_dense_heads/test_gfl_head.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run static/config verification**

```bash
git diff --check
git status --short
conda run -n coxmamba python -m compileall -q mmdet/models/utils/oepc_v2.py mmdet/models/utils/fusion_strategy.py mmdet/models/detectors/fusionnet_xo.py mmdet/models/dense_heads/gflq_head.py configs/coxnet/oepc/OEPC_v2_final.py
```

Expected: no whitespace/compile errors and no uncommitted source changes.

- [ ] **Step 3: Run one GPU-1 smoke iteration**

```bash
oepc_smoke_dir=$(mktemp -d /tmp/oepc-v2-smoke-XXXXXX)
CUDA_VISIBLE_DEVICES=1 conda run -n coxmamba python tools/train.py configs/coxnet/oepc/OEPC_v2_final.py --seed 0 --deterministic --cfg-options runner.type=IterBasedRunner runner.max_iters=1 data.samples_per_gpu=1 data.workers_per_gpu=0 checkpoint_config.interval=1 evaluation.interval=999999 work_dir="$oepc_smoke_dir"
```

Expected: one forward/backward/optimizer iteration completes with finite detection, candidate, foreground, contrastive, and local utility losses. Record but do not delete the temporary log before reporting.

- [ ] **Step 4: Inspect smoke evidence**

Confirm no `loss_oepc_edl` or v2 `loss_oepc_trial_det`; candidate/support/context/route/delta/local-utility metrics are finite; `nvidia-smi` shows the smoke process exited.

- [ ] **Step 5: Review the whole branch**

Review `git diff 69d9037...HEAD` against the spec, focusing on padding, train/eval branching, detached context weights, overlap denominators, local utility mask order, legacy OEPC behavior, and exact loss weights. Any fix first gets a failing test and dedicated commit.

- [ ] **Step 6: Verify remote ancestry**

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git status --short --branch
git log --oneline --decorate origin/main..HEAD
```

Expected: ancestry exits 0, worktree is clean, and only intended commits are listed. If ancestry fails, stop and inspect remote changes; never force-push.

- [ ] **Step 7: Push over remote main**

```bash
git push origin HEAD:main
```

Expected: fast-forward succeeds.

- [ ] **Step 8: Verify remote and report**

```bash
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

Expected: hashes match. Report commit, test counts, smoke log, config, architecture, and that full AP remains unmeasured. Do not start full seed-0 training unless separately requested after implementation review.
