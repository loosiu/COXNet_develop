# TPSC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, verify, push, and train the TPSC same-stage CLFM replacement that uses target-aware P3/P4 prototypes to modulate RGB channels before the unchanged AAM/HOFM path.

**Architecture:** One `TinyAwarePrototypeSemanticCalibration` module consumes same-stage RGB/Thermal P3 and P4 together. Detail-preserving descriptors feed shared semantic slots, a level-wise RGB–Thermal conditioner and optional prototype-set relation block, then fixed-bound channel scaling modifies only RGB P3/P4 while the original Thermal features enter AAM/HOFM unchanged.

**Tech Stack:** Python 3.8, PyTorch/MMDetection, MMCV config/runner, pytest, CUDA through the `coxmamba` environment, Bash/tmux/flock for sequential GPU-1 training.

**Spec:** `docs/superpowers/specs/2026-09-29-tpsc-design.md`

## Global Constraints

- Work from the `oepc-v2-main` history containing design commit `9a34ccb` and this plan in `/data/siwoo/COXNet-OEPC-v2`; create an isolated implementation worktree/branch at execution time.
- Implement independently; do not copy AGPL-3.0 ProtoHGF source into this MIT repository.
- RGB and Thermal FPNs both use `start_level=1`; TPSC receives equal-shape modality pairs at P3 and P4.
- TPSC is mutually exclusive with CLFM, TRPC, OEPC, TOPC, and PRLDFC.
- Preserve AAM/HOFM, DSR, GFLQHead, QLSAssigner, NMS, and all legacy replacement implementations.
- Pass original Thermal feature values to AAM/HOFM; permit detection gradients through the Thermal prototype descriptor.
- Use P3/P4 only, `K=8`, prototype dimension 64, four attention heads, one relation block, fixed modulation bound 0.1, and non-zero modulation initialization `std=1e-2`.
- Keep `wf_loss=True`, `wf_loss_mode='kl_v2'`, and `wf_loss_weight=0.1` in all matched TPSC/control configs.
- Add only coverage loss weight 0.05 and diversity loss weight 0.01.
- Do not add spatial candidates, top-k quotas, local cross-modal search, FFT, EDL, teacher models, feature warp, or an additional learned output gate.
- Push only verified source/config/docs/tests to `origin/main` without force; never commit datasets, checkpoints, logs, or work directories.
- Run GPU 1 jobs sequentially and refuse mixed-commit training through an exact commit check and `/tmp/coxnet_oepc_gpu1.lock`.

## Review Focus

- An all-padding or partly padded feature map must produce finite zero-mass attention outside valid pixels and never NaN.
- Empty-GT images must have finite zero coverage loss while retaining valid detection gradients through the calibration path.
- P3 and P4 can have different spatial sizes, but each RGB/Thermal pair must match exactly and invalid level lists must fail clearly.
- Batch-size-one evaluation must make Thermal-prototype shuffle a documented no-op rather than silently indexing another image.
- The original Thermal tensor passed to AAM/HOFM must remain value-identical even though its descriptor receives gradients.

---

### Task 1: Gaussian coverage targets and valid detail descriptors

**Files:**
- Create: `mmdet/models/utils/tpsc.py`
- Create: `tests/test_models/test_utils/test_tpsc.py`

**Interfaces:**
- Produces: `build_tpsc_gaussian_targets(gt_bboxes, padded_size, feat_size, device, valid_mask=None) -> torch.Tensor` with shape `B x 1 x H x W`.
- Produces: `valid_average_pool(feature, valid_mask, kernel_size=3) -> torch.Tensor`.
- Produces: `TPSCDescriptor(channels, prototype_dim)` with `forward(p3, p4, valid_p3, valid_p4) -> Tuple[torch.Tensor, torch.Tensor]`.

- [ ] **Step 1: Write failing target and descriptor tests**

Add tests named:

- `test_tpsc_gaussian_targets_cover_fractional_centers_and_mask_padding`
- `test_tpsc_gaussian_targets_merge_by_max_and_handle_empty_gt`
- `test_tpsc_gaussian_targets_reject_mismatched_valid_mask`
- `test_tpsc_valid_average_pool_excludes_padding_and_handles_all_invalid`
- `test_tpsc_descriptor_combines_p3_detail_and_p4_context_without_mutating_inputs`

Assert minimum-radius Gaussian peaks, max merging, padding zeros, finite empty targets, exact output shapes, unchanged input tensors, and a changed P3 descriptor when only P4 changes.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python -m pytest -o addopts="" tests/test_models/test_utils/test_tpsc.py -q
```

Expected: collection/import failure because `mmdet.models.utils.tpsc` does not exist.

- [ ] **Step 3: Implement the target builder and descriptor primitives**

Implement the exact Task 1 interfaces. Gaussian radius is at least one feature cell, objects merge by pixel maximum, valid masks remove padding, and all-invalid normalized pooling returns finite zeros. `TPSCDescriptor` creates `P3`, `P3-ValidAvgPool3(P3)`, and upsampled projected `P4` branches without modifying source features.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run the Step 2 command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add mmdet/models/utils/tpsc.py tests/test_models/test_utils/test_tpsc.py
git commit -m "Add TPSC targets and detail descriptors"
```

### Task 2: Shared-slot extraction and auxiliary losses

**Files:**
- Modify: `mmdet/models/utils/tpsc.py`
- Modify: `tests/test_models/test_utils/test_tpsc.py`

**Interfaces:**
- Consumes: Task 1 descriptor tensors and valid masks.
- Produces: `SharedSlotPrototypeExtractor(prototype_dim=64, num_slots=8)` with `forward(rgb, thermal, valid_mask) -> dict`.
- Output keys: `rgb_prototypes`, `thermal_prototypes`, `rgb_attention`, `thermal_attention`.
- Produces: `prototype_coverage_loss(thermal_attention, target, valid_mask) -> torch.Tensor`.
- Produces: `prototype_diversity_loss(thermal_attention, valid_mask) -> torch.Tensor`.

- [ ] **Step 1: Write failing slot and loss tests**

Add tests named:

- `test_tpsc_shared_slots_normalize_each_modality_over_valid_space`
- `test_tpsc_slots_exclude_padding_and_return_zero_for_all_invalid_sample`
- `test_tpsc_uses_shared_queries_but_modality_specific_projections`
- `test_tpsc_coverage_prefers_attention_on_gaussian_foreground`
- `test_tpsc_coverage_is_zero_for_empty_target_and_finite_for_all_padding`
- `test_tpsc_diversity_penalizes_collapsed_slots_more_than_separated_slots`

Assert slot attention sums to one for samples with valid pixels, all-invalid samples are zero, RGB/Thermal share one query parameter, and losses are finite scalars.

- [ ] **Step 2: Run slot/loss tests and verify RED**

Run the Task 1 pytest command with `-k "slot or coverage or diversity"`. Expected: missing interface failures.

- [ ] **Step 3: Implement shared-slot extraction and losses**

Use modality-specific `1x1 Conv + channel LayerNorm` projections and the same learned `K x d` query tensor. Implement a masked spatial softmax that returns zeros when no location is valid. Normalize the Gaussian distribution only for non-empty targets; compute symmetric KL there and return graph-connected finite zero otherwise. Diversity is mean off-diagonal cosine between flattened Thermal attention maps.

- [ ] **Step 4: Run all TPSC tests and verify GREEN**

Run the Task 1 command. Expected: Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add mmdet/models/utils/tpsc.py tests/test_models/test_utils/test_tpsc.py
git commit -m "Implement TPSC shared semantic slots"
```

### Task 3: Cross-modal conditioning, relation mixing, and RGB modulation

**Files:**
- Modify: `mmdet/models/utils/tpsc.py`
- Modify: `tests/test_models/test_utils/test_tpsc.py`

**Interfaces:**
- Consumes: two RGB/Thermal descriptor pairs and Task 2 slot dictionaries.
- Produces: `PrototypeRelationBlock(dim=64, num_heads=4)` returning `(nodes, attention_weights)`.
- Produces: `TinyAwarePrototypeSemanticCalibration(channels, prototype_dim=64, num_slots=8, num_heads=4, relation_depth=1, modulation_bound=0.1, init_std=1e-2, use_relation=True)`.
- `forward(rgb_feats, thermal_feats, valid_masks=None, coverage_targets=None, return_aux=False, disable_modulation=False, shuffle_thermal_prototypes=False, disable_relation=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]]` where calibrated RGB features preserve the two input shapes.
- `gradient_diagnostics() -> Dict[str, float]` returns the latest backward norms for RGB projection, Thermal projection, relation, conditioner, and modulation parameters.
- Auxiliary loss keys: `coverage_loss`, `diversity_loss`.
- Auxiliary monitor keys: `attention_entropy_rgb`, `attention_entropy_thermal`, `prototype_pairwise_cosine`, `prototype_effective_rank`, `proto_cos_before`, `proto_cos_after`, `cross_modal_attention_mass`, `cross_scale_attention_mass`, `channel_scale_abs_mean`, `channel_scale_abs_max`, `modulation_ratio` and cached gradient norms.

- [ ] **Step 1: Write failing calibration tests**

Add tests named:

- `test_tpsc_core_uses_same_slot_thermal_conditioning_without_relation`
- `test_tpsc_relation_connects_modalities_and_scales_with_expected_shapes`
- `test_tpsc_modifies_only_rgb_and_respects_fixed_ten_percent_bound`
- `test_tpsc_nonzero_initialization_sends_detection_gradient_to_both_modalities`
- `test_tpsc_disable_modulation_is_exact_rgb_identity`
- `test_tpsc_disable_relation_matches_core_path`
- `test_tpsc_shuffle_thermal_prototypes_changes_batch_two_and_is_noop_for_batch_one`
- `test_tpsc_forward_handles_empty_gt_partial_padding_and_all_padding_without_nan`
- `test_tpsc_rejects_wrong_level_count_and_modality_shape_mismatch`

Assert original Thermal inputs are value-identical before/after, RGB scale lies in `[0.9, 1.1]`, the initial delta is non-zero, representative gradient norms become non-zero after backward, and intervention flags alter only their documented path.

- [ ] **Step 2: Run calibration tests and verify RED**

Run the Task 1 command with `-k "core or relation or modulation or shuffle or forward"`. Expected: missing class/method failures.

- [ ] **Step 3: Implement the core and relation paths**

Build one module that accepts exactly two levels. Per level, condition RGB slots with `MLP([P_R, P_T, P_T-P_R])`. With relation enabled, concatenate `[R3,T3,R4,T4]`, apply one pre-norm `nn.MultiheadAttention(batch_first=True)` plus MLP residual block, split nodes, then apply the same conditioner. Flatten conditioned RGB slots into level-specific linear scale heads initialized with `Normal(0,1e-2)` and use only `F_R * (1 + 0.1*tanh(gamma))`.

- [ ] **Step 4: Implement interventions, diagnostics, and gradient caching**

Implement batch roll for Thermal prototype shuffle, exact identity for `disable_modulation`, identity relation for `disable_relation`, detached attention/modulation monitors, and representative parameter hooks whose most recent backward norm is returned on the next forward or through `gradient_diagnostics()`.

- [ ] **Step 5: Run all TPSC tests and verify GREEN**

Run the Task 1 command. Expected: all TPSC module tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add mmdet/models/utils/tpsc.py tests/test_models/test_utils/test_tpsc.py
git commit -m "Implement TPSC prototype calibration"
```

### Task 4: Integrate TPSC into COXNet and add controlled configs

**Files:**
- Modify: `mmdet/models/utils/__init__.py`
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Create: `configs/coxnet/tpsc/same_stage_control.py`
- Create: `configs/coxnet/tpsc/TPSC_core.py`
- Create: `configs/coxnet/tpsc/TPSC_relation.py`
- Modify: `tests/test_models/test_utils/test_tpsc.py`

**Interfaces:**
- Consumes: `TinyAwarePrototypeSemanticCalibration` and `build_tpsc_gaussian_targets`.
- Produces: detector flags `use_tpsc: bool=False`, `tpsc_cfg: Optional[dict]=None`.
- `FusionLayer.forward` returns weighted `loss_tpsc_coverage`, `loss_tpsc_diversity`, plus detached `tpsc_*` monitors during training.
- Produces: `FusionLayer.last_tpsc_aux` for inference diagnostics.

- [ ] **Step 1: Write failing integration/config tests**

Add tests named:

- `test_fusion_layer_rejects_tpsc_with_any_other_clfm_replacement`
- `test_fusion_layer_tpsc_calibrates_p3_p4_before_hofm_and_preserves_thermal_values`
- `test_fusion_layer_tpsc_builds_level_targets_and_weighted_losses`
- `test_fusion_layer_tpsc_inference_receives_padding_masks_and_sets_last_aux`
- `test_fusionnet_xo_forwards_tpsc_flags_and_enters_aux_train_path`
- `test_tpsc_configs_define_matched_control_core_and_relation_contracts`

The config test asserts both FPN start levels 1, CLFM and legacy replacements off, exact TPSC defaults/loss weights, matched `wf_loss`, P3/P4 levels, and distinct work directories.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python -m pytest -o addopts="" tests/test_models/test_utils/test_tpsc.py -q
```

Expected: constructor/config failures for missing `use_tpsc` plumbing.

- [ ] **Step 3: Integrate one cross-level TPSC module before the FusionLayer level loop**

Add TPSC to replacement mutual exclusion. Save/restore the CPU RNG state around module construction so downstream HOFM initialization matches the control. Before the existing per-level loop, construct P3/P4 valid masks and training targets, call TPSC once, replace only the selected RGB feature list entries, and then run unchanged level-wise HOFM/AAM with original Thermal entries.

- [ ] **Step 4: Add detector plumbing and the three configs**

Pass flags/config through `FusionNetXO`, include TPSC in its auxiliary training condition, export the class in utils, and create control/core/relation configs with the spec defaults. Preserve `simple_test(img_metas=...)` mask propagation.

- [ ] **Step 5: Run TPSC plus legacy focused regressions**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python -m pytest -o addopts="" tests/test_models/test_utils/test_tpsc.py tests/test_models/test_utils/test_topc.py tests/test_models/test_utils/test_oepc.py tests/test_models/test_utils/test_trpc.py -q
```

Expected: all selected tests pass; unrelated repository collection failures are not counted as TPSC failures.

- [ ] **Step 6: Commit Task 4**

```bash
git add mmdet/models/utils/__init__.py mmdet/models/utils/fusion_strategy.py mmdet/models/detectors/fusionnet_xo.py configs/coxnet/tpsc tests/test_models/test_utils/test_tpsc.py
git commit -m "Integrate TPSC as a CLFM replacement"
```

### Task 5: Document, validate, and prepare reproducible training

**Files:**
- Create: `docs/TPSC.md`
- Modify: `README.md`
- Create outside Git tracking: `work_dir/coxmamba/rgbtdroneperson/tpsc/run_tpsc_gpu1.sh`

**Interfaces:**
- Produces: exact method/loss/diagnostic documentation without a pre-training performance claim.
- Produces: a fail-fast queue for `same_stage_control` seed 0, `TPSC_core` seed 0, then `TPSC_relation` seeds 0, 1, and 2.

- [ ] **Step 1: Add method and training documentation**

Describe TPSC as the current experiment, its P3/P4 flow, shared slots, core/relation distinction, RGB-only bound, losses, interventions, exact configs, and absence of a performance claim before completed evaluation. Preserve links and reproducibility notes for PRLDFC/TOPC/OEPC/TRPC.

- [ ] **Step 2: Add documentation/config contract checks and verify them**

Extend `test_tpsc.py` to assert README/config paths exist and that no forbidden candidate/frequency/teacher option enters canonical config. Run all TPSC tests. Expected: PASS.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/TPSC.md tests/test_models/test_utils/test_tpsc.py
git commit -m "Document TPSC experiment and controls"
```

- [ ] **Step 4: Run static and CPU verification**

Run `compileall` over new/modified Python files, the Task 4 focused suite, config construction for all three configs, and `git diff --check`. Expected: no syntax errors, all focused tests pass, all configs build, clean diff check.

- [ ] **Step 5: Run a GPU-1 one-batch forward/backward smoke**

After confirming GPU 1 is free, run a temporary one-iteration config with `workers_per_gpu=0`. Assert finite detector/TPSC/wf losses, non-zero TPSC modulation, and non-zero cached extractor/conditioner/relation/modulation gradients. Remove only the explicit temporary smoke directory after recording results.

- [ ] **Step 6: Commit any verification-only fixes**

If verification required source/test changes, commit them as `Fix TPSC verification issues`; otherwise make no empty commit.

### Task 6: Final review, push, and GPU-1 training queue

**Files:**
- Inspect: all TPSC commits and the design/plan documents.
- Create outside Git tracking: the Task 5 launcher, manifests, logs, and work directories.

**Interfaces:**
- Produces: a verified commit on `origin/main`.
- Produces: separate outputs under `work_dir/coxmamba/rgbtdroneperson/tpsc/{control_seed0,core_seed0,seed0,seed1,seed2}`.

- [ ] **Step 1: Run final whole-change review and verification**

Use a fresh reviewer if available, inspect the entire branch diff against the spec, rerun focused tests and GPU smoke, confirm no data/checkpoint/log is staged, and record exact commands/results.

- [ ] **Step 2: Fast-forward push verified code**

Fetch `origin`, verify `origin/main` is an ancestor of the implementation HEAD, then push `HEAD:main` without force. Verify the remote SHA with `git ls-remote origin refs/heads/main`.

- [ ] **Step 3: Create the exact-commit sequential launcher**

Use `/home/viplab/anaconda3/envs/coxmamba/bin/python`, physical GPU 1, deterministic seeds, `/tmp/coxnet_oepc_gpu1.lock`, an exact expected commit, one manifest and console log per run, and `set -euo pipefail`. Run control seed 0, core seed 0, relation seed 0, relation seed 1, relation seed 2; start each only after the previous command exits 0.

- [ ] **Step 4: Launch detached and verify live state**

Start tmux session `tpsc_gpu1`. Verify the real `tools/train.py` child, `CUDA_VISIBLE_DEVICES=1`, GPU allocation, exact config/commit in the manifest, a growing seed-specific log, and the first finite training iteration. Report later runs as queued, not running.

- [ ] **Step 5: Audit seed-0 mechanism before continuing identical bad code**

After the relation seed-0 first epoch, inspect coverage, attention entropy, modulation ratio, and cached gradient norms. If the path is dead, NaN, or all-zero, stop the queue, diagnose/fix/reverify, and restart all relation seeds from clean new work directories. Do not tune on validation AP mid-run.

- [ ] **Step 6: Report completed results only when artifacts exist**

For every completed run, parse best epoch by `bbox_mAP_50` and report AP25/AP50/AP75/tiny/tiny1/tiny2/tiny3/small, checkpoint path, commit, and mechanism diagnostics. Compare control/core/relation seed 0, then relation seed 0/1/2 mean and standard deviation against the unchanged COXNet reference using the same validation annotations.
