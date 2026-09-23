# TF-OPC Implementation and Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, verify, document, push, and sequentially train seed 0/1/2 of Thermal-First Object-Prototype Calibration on GPU 1.

**Architecture:** A P3-only module uses Thermal P3/P4 to discover candidates and form candidate-specific object/context/detail prototypes. Each prototype searches a local RGB P3/P4 descriptor with top-M attention, receives cycle-consistency supervision, and injects one gated, overlap-normalized, RMS-capped residual into RGB before the unchanged AAM/HOFM. Legacy OEPC/TRPC remain reproducible.

**Tech Stack:** Python, PyTorch, MMCV/MMDetection, pytest, COXNet FusionNetXO/GFLQ, conda `coxmamba`, Git, GPU 1.

**Spec:** `docs/superpowers/specs/2026-09-23-tf-opc-design.md`

## Global Constraints

- Worktree `/data/siwoo/COXNet-OEPC-v2`, branch `oepc-v2-main`; preserve other worktrees/jobs.
- New module `oepc_v2.py`; do not change legacy `oepc.py` behavior.
- Same-stage P3-P6, P3 calibration only, P4 descriptor context only, Thermal P3 unchanged to AAM.
- No CLFM/DWT/IDWT/DeConv/EDL/foreground auxiliary/detector utility/FiLM/global K prototypes.
- Loss: `L_det + 0.10 L_heatmap + 0.05 L_contrast + 0.02 L_cycle`.
- Training selection: ST thresholded top-K, no local peak. Inference: configurable local peak then thresholded top-K.
- Push verified code to `origin/main` without force before launching full training.
- Run GPU 1 seeds strictly sequentially: 0 exits successfully, then 1, then 2. Use separate work dirs and logs.

## Review Focus

- ST selection must be hard/sparse forward and have non-zero heatmap-logit backward gradient.
- Candidate windows at boundaries/padding must normalize only valid cells and remain finite with missing context.
- Top-M association/scatter ordering must map correct offsets and normalized overlap must not amplify residuals.
- Empty/no-candidate batches must return identity RGB and finite zero auxiliary losses.
- P4 may affect descriptors but may not replace/resize the P3 feature entering AAM.

---

### Task 1: Gaussian target and P3/P4 descriptors

**Files:**
- Create: `mmdet/models/utils/oepc_v2.py`
- Create: `tests/test_models/test_utils/test_oepc_v2.py`
- Modify: `mmdet/models/utils/__init__.py`

**Interfaces:**
- `build_gaussian_center_targets(gt_boxes, padded_size, feat_size, device, valid_mask=None, max_radius=2)`.
- `ThermalFirstObjectPrototypeCalibration._encode(rgb_p3, thermal_p3, rgb_p4, thermal_p4, valid)` returns modality `base`, `detail`, and `descriptor`.

- [ ] **Step 1: RED — write Gaussian behavior tests**

Test fractional centers, two adjacent boxes, object-wise normalized maxima, max overlap, empty GT, and masked padding.

- [ ] **Step 2: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k gaussian -q`

Expected: import/missing-symbol failure.

- [ ] **Step 3: GREEN — implement normalized Gaussian builder**

Use the spec radius/sigma formula, an object-local meshgrid, per-object `gaussian /= gaussian.max().clamp_min(1e-12)`, `torch.maximum` merge, and exact valid-mask shape validation.

- [ ] **Step 4: RED — write descriptor tests**

Test equal-modality shape validation, P4-smaller validation, P4 changing descriptor output, constant P3 producing zero detail away from boundaries, and padded values not changing valid descriptors.

- [ ] **Step 5: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k descriptor -q`

Expected: missing class/method failure.

- [ ] **Step 6: GREEN — implement descriptors**

Create independent RGB/Thermal P3/P4 projections and descriptor fusions. Use mask-normalized 3x3 average and only `bilinear, align_corners=False` P4-to-P3 resize.

```python
base = base_norm(p3_projection(p3))
detail = base - valid_average(base, valid, 3)
semantic = F.interpolate(semantic_norm(p4_projection(p4)),
                         size=p3.shape[-2:], mode='bilinear',
                         align_corners=False)
descriptor = descriptor_norm(fusion(torch.cat(
    [base, detail, semantic], dim=1)))
```

- [ ] **Step 7: Verify and commit**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'gaussian or descriptor' -q
git add mmdet/models/utils/oepc_v2.py mmdet/models/utils/__init__.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Add TF-OPC targets and descriptors"
```

### Task 2: ST candidates and candidate-specific prototypes

**Files:**
- Modify: `mmdet/models/utils/oepc_v2.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- `_select_candidates(logits, valid) -> dict(indices, weights, hard_mask, soft_mask, active)`.
- `_build_prototypes(thermal_descriptor, thermal_detail, selection, valid) -> dict(object, context, detail, prototype, context_valid)`.

- [ ] **Step 1: RED — selection tests**

Test training hard forward values `{0,1}`, no local-peak suppression of adjacent cells, top-K cap, non-zero logit gradient through ST weights, evaluation local-peak behavior, RGB-invariant heatmap, and no-candidate behavior.

- [ ] **Step 2: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'selection or heatmap' -q`

Expected: missing method failure.

- [ ] **Step 3: GREEN — implement Thermal heatmap and ST selection**

Instantiate only a Thermal heatmap head with prior `0.01`. Training top-K uses no max-pool. Gather ST weights from:

```python
soft = torch.sigmoid((logits - threshold_logit) / temperature)
hard = thresholded_topk_mask(probability, valid, max_candidates)
straight_through = soft + (hard - soft).detach()
weights = straight_through.flatten(2).gather(2, indices)
```

Evaluation applies odd `inference_peak_kernel` max-pool equality before threshold/top-K.

- [ ] **Step 4: RED — prototype tests**

Use hand-filled 7x7 features to assert 3x3 object pooling, 7x7-minus-3x3 context, detail pooling, per-candidate output shape, boundary validity, zero context fallback, and local contrast not treating another object as a negative.

- [ ] **Step 5: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k prototype -q`

Expected: missing prototype method failure.

- [ ] **Step 6: GREEN — implement per-candidate prototypes**

Use `F.unfold` and gathered candidate indices. Normalize Gaussian object weights and ring context weights by valid mass. Produce `prototype_mlp([P_obj, P_obj-P_ctx, P_detail])`, plus margin cosine contrast only for active candidates with valid context.

- [ ] **Step 7: Verify and commit**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'selection or heatmap or prototype' -q
git add mmdet/models/utils/oepc_v2.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Add sparse candidate-specific thermal prototypes"
```

### Task 3: Top-M association, cycle loss, gate, and residual

**Files:**
- Modify: `mmdet/models/utils/oepc_v2.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- `_associate(thermal_prototype, rgb_descriptor, indices, valid) -> dict(attention, offsets, rgb_prototype, similarity, confidence, entropy)`.
- `_cycle_loss(rgb_prototype, thermal_descriptor, forward_fields, indices, active, valid)`.
- `forward(rgb, thermal, rgb_semantic, thermal_semantic, valid_mask=None, center_target=None, return_aux=False)`.

- [ ] **Step 1: RED — forward association tests**

Test Thermal-query direction, radius-2 bounds, exactly top-M nonzero weights, hand-derived RGB prototype, distance-prior center preference, and invalid option masking.

- [ ] **Step 2: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k association -q`

Expected: missing association method failure.

- [ ] **Step 3: GREEN — implement top-M association**

Unfold RGB descriptor search windows, gather by Thermal indices, cosine compare projected prototype/query with normalized RGB keys, subtract distance prior, top-M mask, and softmax only selected valid options. Return zero finite fields when a candidate has no valid option.

- [ ] **Step 4: RED — cycle tests**

Create synthetic descriptors with known zero displacement and one-cell displacement. Assert zero/positive normalized cycle loss, inactive candidates ignored, and no feature tensor is warped.

- [ ] **Step 5: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k cycle -q`

Expected: missing cycle method failure.

- [ ] **Step 6: GREEN — implement reverse local association and cycle**

Compute forward expected RGB coordinate, round/detach it only for reverse window extraction, query Thermal descriptors with `P_R`, compute reverse top-M attention, and apply `smooth_l1((y_back-y_origin)/max(radius,1), 0)` over active valid candidates.

- [ ] **Step 7: RED — calibration safety/backward tests**

Test one gate, no EDL/FiLM/utility symbols in the v2 module, small nonzero residual init, exact identity outside support, normalized overlap for two candidates, RMS cap, original Thermal equality, empty batch identity, and non-zero finite gradients for heatmap/prototype/association/gate/residual/P3/P4 inputs.

- [ ] **Step 8: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k 'calibration or backward or empty' -q`

Expected: incomplete forward failure.

- [ ] **Step 9: GREEN — implement gate, residual, scatter, losses, metrics**

Use candidate MLP inputs from the spec, one sigmoid gate, top-M attention scatter, normalized overlap average, support cap, and per-location RMS cap. Return only `heatmap_loss`, `contrast_loss`, `cycle_loss` plus documented detached metrics. No active candidate returns finite zeros and identity RGB.

- [ ] **Step 10: Verify and commit**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -q
git add mmdet/models/utils/oepc_v2.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Implement TF-OPC matching and RGB calibration"
```

### Task 4: Fusion integration and canonical config

**Files:**
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Create: `configs/coxnet/oepc/TF_OPC.py`
- Modify: `tests/test_models/test_utils/test_oepc_v2.py`

**Interfaces:**
- `use_oepc_v2=False`, `oepc_v2_cfg=None` in detector/fusion constructors.
- Aux loss keys exposed as `loss_tfopc_heatmap`, `loss_tfopc_contrast`, `loss_tfopc_cycle`.
- Canonical work dir is seed-overridden from `work_dir/coxmamba/rgbtdroneperson/tf_opc/seed0`.

- [ ] **Step 1: RED — FusionLayer integration tests**

Test replacement mutual exclusion, P3 receives same-modality P4, Thermal input reaches HOFM unchanged, P4 affects TF-OPC output but not direct HOFM Thermal input, no utility payload, exact weighted loss keys, padding, and legacy OEPC/TRPC construction.

- [ ] **Step 2: Run RED**

Run: `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py -k fusion -q`

Expected: constructor/path failure.

- [ ] **Step 3: GREEN — integrate the new flag/path**

Instantiate `ModuleDict` only for configured P3. Preserve HOFM RNG initialization, build valid/Gaussian targets from `img_metas`, call v2 with `v_feats[1]/t_feats[1]`, aggregate exactly three weighted losses and metrics, and include `use_oepc_v2` in metadata-aware train extraction. Do not change the GFL utility API.

- [ ] **Step 4: RED — config contract test**

Assert both FPN start levels 1, CLFM/TRPC/legacy OEPC off, v2 P3-only, exact architecture defaults/loss weights, no EDL/utility options, and correct work dir.

- [ ] **Step 5: GREEN — create `TF_OPC.py`**

Set embed 64, object 3, context 7, search radius 2, top-M 4, threshold 0.10, prior 0.01, K 100, ST temperature 0.5, inference peak 3, residual cap 0.2, contrast margin 0.2, heatmap/contrast/cycle weights 0.10/0.05/0.02, `wf_loss=False`.

- [ ] **Step 6: Verify config and regressions**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py tests/test_models/test_utils/test_oepc.py tests/test_models/test_utils/test_trpc.py -q
conda run -n coxmamba python -c "from mmcv import Config; from mmdet.models import build_detector; c=Config.fromfile('configs/coxnet/oepc/TF_OPC.py'); m=build_detector(c.model, train_cfg=c.get('train_cfg'), test_cfg=c.get('test_cfg')); print(type(m.fuse_layer.oepc_v2_layers['0']).__name__)"
```

Expected: tests pass and class prints `ThermalFirstObjectPrototypeCalibration`.

- [ ] **Step 7: Commit**

```bash
git add mmdet/models/utils/fusion_strategy.py mmdet/models/detectors/fusionnet_xo.py configs/coxnet/oepc/TF_OPC.py tests/test_models/test_utils/test_oepc_v2.py
git commit -m "Integrate TF-OPC into COXNet"
```

### Task 5: Documentation, full verification, and push

**Files:**
- Modify: `README.md`
- Create: `docs/TF_OPC.md`

**Interfaces:**
- Documents actual architecture/config/commands and preserves prior results as baselines.

- [ ] **Step 1: Update documentation**

Describe TF-OPC as the primary current method, its exact data flow/loss/defaults, differences from CLFM and legacy OEPC, diagnostics, seed commands, and absence of a performance claim before training.

- [ ] **Step 2: Commit docs**

```bash
git add README.md docs/TF_OPC.md
git commit -m "Document TF-OPC training workflow"
```

- [ ] **Step 3: Run focused and repository tests**

```bash
conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_oepc_v2.py tests/test_models/test_utils/test_oepc.py tests/test_models/test_utils/test_trpc.py tests/test_models/test_dense_heads/test_gfl_head.py -q
conda run -n coxmamba python -m pytest -q
```

Record any unrelated repository failure by exact test name; do not conceal it.

- [ ] **Step 4: Run compile/config/static checks**

```bash
git diff --check
conda run -n coxmamba python -m compileall -q mmdet/models/utils/oepc_v2.py mmdet/models/utils/fusion_strategy.py mmdet/models/detectors/fusionnet_xo.py configs/coxnet/oepc/TF_OPC.py
```

- [ ] **Step 5: Run one GPU-1 smoke iteration**

Use `workers_per_gpu=0`, a unique `/tmp/tf-opc-smoke-*` work dir, seed 0, deterministic mode, and an IterBasedRunner `max_iters=1`. Confirm finite detector and three TF-OPC losses, then ensure the process exits.

- [ ] **Step 6: Whole-branch self-review**

Generate the executing-plans review package from base `69d9037` to HEAD. Because subagent delegation is disabled for this session, perform the prescribed read-only review yourself, ledger all rulings/minors, and fix Critical/Important findings with RED→GREEN tests in one pass.

- [ ] **Step 7: Verify and push remote main**

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git status --short --branch
git push origin HEAD:main
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

Require clean status, fast-forward ancestry, successful push, and equal remote/local hashes.

### Task 6: Launch sequential seed 0/1/2 training on GPU 1

**Files:**
- Create runtime logs/outputs only under `work_dir/coxmamba/rgbtdroneperson/tf_opc/`; never Git-add them.
- Create queue control/log files under a Git-ignored runtime directory.

**Interfaces:**
- Queue runs seed 0, then 1, then 2 only after prior exit code 0.
- Work dirs: `.../tf_opc/seed0`, `.../tf_opc/seed1`, `.../tf_opc/seed2`.

- [ ] **Step 1: Re-query GPU 1 and active training state**

Run `nvidia-smi`, `pgrep -af 'tools/train.py|dist_train'`, inspect existing tmux sessions/locks, and do not displace another experiment. Wait on the existing safe queue mechanism if GPU 1 is busy.

- [ ] **Step 2: Create a fail-fast sequential launcher**

Use the repository's existing queue/lock convention if present. Otherwise create a Git-ignored shell launcher whose three explicit commands use:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n coxmamba python tools/train.py configs/coxnet/oepc/TF_OPC.py --seed SEED --deterministic --work-dir work_dir/coxmamba/rgbtdroneperson/tf_opc/seedSEED
```

Use `set -euo pipefail` so seed 1/2 do not start after a failed seed. Redirect an outer queue log and write PID/status markers.

- [ ] **Step 3: Launch detached and verify seed 0 is real**

Start through the established tmux/nohup queue convention, then verify process command line, PID, GPU 1 allocation, newly growing seed-0 log, config dump, and first iteration. A shell PID without a Python child is not sufficient.

- [ ] **Step 4: Report live queue evidence**

Report pushed commit, launcher/session/PID, seed work dirs/log paths, current seed/iteration, and that seeds 1/2 are queued sequentially rather than running concurrently. Do not claim completed metrics until logs show evaluation completion.
