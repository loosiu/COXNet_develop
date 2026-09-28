# PRLDFC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Prototype-Routed Local Dynamic Frequency Calibration as a complete same-stage CLFM replacement and publish the verified commit to both requested repositories.

**Architecture:** A dense Thermal seed field supplies differentiable object support. Detail-preserving prototypes route globally learnable FFT bands and Thermal reliability, while dense band features produce a bounded RGB-only residual before the unchanged AAM/HOFM path.

**Tech Stack:** Python 3.8/3.9, PyTorch 1.13, MMDetection/MMCV 1.x, SciPy Hungarian assignment, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-prldfc-design.md`

## Global Constraints

- RGB and Thermal FPNs both use `start_level=1`; PRLDFC sees equal-shape P3/P4 features.
- PRLDFC is mutually exclusive with CLFM, TRPC, OEPC, and TOPC.
- Thermal features are never mutated; only calibrated RGB enters the existing AAM/HOFM.
- No DeConv, DWT/IDWT, hard top-k, peak NMS, EDL, contrastive, utility router, spatial warp, or patch FFT is added.
- Canonical treatment applies PRLDFC to levels `(0, 1)` and keeps `wf_loss=True`, `kl_v2`, weight `0.1`.
- Dense training uses soft support; optional sparse inference uses a threshold with no fixed candidate quota.
- Frequency boundaries are level-global; object-conditioned band weights and reliability are dynamic.
- Existing TOPC/TRPC/OEPC code, configs, checkpoints, and tests remain reproducible.

## Review Focus

- Zero-GT images must produce finite assignment/loss values and must not create false positive seeds.
- Same-cell crowded GT must receive distinct cells when candidates exist, otherwise report unassigned GT without duplicate supervision.
- Odd feature sizes, padding, and ignored regions must not contaminate FFT features, attention, targets, or residual output.
- Dense overlapping supports must remain bounded rather than increasing linearly with the number of seeds.
- Mixed-precision callers must not send unsupported half-precision tensors into FFT and must receive the original dtype back.

---

### Task 1: One-to-one Thermal seed supervision

**Files:**
- Create: `mmdet/models/utils/prldfc.py`
- Create: `tests/test_models/test_utils/test_prldfc.py`

**Interfaces:**
- Produces: `build_prldfc_seed_targets(seed_logits, seed_offsets, seed_log_scales, gt_bboxes, padded_size, valid_mask, level_scale_range, matching_radius=2) -> dict`.
- Produces: `PRLDFCSeedHead(channels, seed_prior)` returning `(logits, offsets, log_scales)`.
- Target dict keys: `seed_target`, `loss_valid`, `positive_mask`, `offset_target`, `scale_target`, `eligible_gt_count`, `unassigned_gt_count`.

- [ ] **Step 1: Write failing seed tests**

Add tests proving literal output shapes/ranges, scale filtering, zero-GT behavior, padding exclusion, and two GT with the same integer center receiving distinct positive cells.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=/tmp/prldfc-test-deps conda run -n bf python -m pytest -o addopts="" tests/test_models/test_utils/test_prldfc.py -q`

Expected: collection fails because `mmdet.models.utils.prldfc` does not exist.

- [ ] **Step 3: Implement seed head and deterministic Hungarian targets**

Use feature cell centers `(index + 0.5)`, detach predictions only for the matching decision, use radius-2 candidate unions, and keep differentiable losses outside the assignment. Invalid/ignored cells are excluded through `valid_mask`.

- [ ] **Step 4: Run the seed tests and verify GREEN**

Run the Task 1 command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

Commit message: `Implement one-to-one PRLDFC seed supervision`.

---

### Task 2: Dynamic frequency bank and RGB calibration core

**Files:**
- Modify: `mmdet/models/utils/prldfc.py`
- Modify: `tests/test_models/test_utils/test_prldfc.py`

**Interfaces:**
- Produces: `DynamicFrequencyBank(channels, frequency_dim, num_bands, temperature, min_band_width)` with `forward(feature, valid_mask) -> (bands, diagnostics)`.
- Produces: `PrototypeRoutedLocalDynamicFrequencyCalibration.forward(rgb, thermal, valid_mask=None, gt_bboxes=None, padded_size=None, level_scale_range=(0, 32), return_aux=False) -> rgb_cal` or `(rgb_cal, aux)`.
- Auxiliary losses: `seed_loss`, `offset_loss`, `scale_loss`, `cardinality_loss`.
- Auxiliary diagnostics include boundaries, band weights, reliability, attention entropy, support ratio, delta ratio, eligible/unassigned GT, and seed mass/count.

- [ ] **Step 1: Write failing frequency-bank tests**

Test monotonic boundaries with minimum width, normalized masks, reconstruction on odd/even shapes, padding masking, original dtype restoration, and non-zero gradients to band parameters and projection.

- [ ] **Step 2: Run the frequency tests and verify RED**

Run the Task 1 command. Expected: failures name the missing `DynamicFrequencyBank` behavior.

- [ ] **Step 3: Implement the frequency bank**

Perform `rfft2/irfft2` in float32 with orthonormal normalization, normalize the three radial masks to a partition, and cast reconstructed bands back to the caller dtype.

- [ ] **Step 4: Run frequency tests and verify GREEN**

Run the Task 1 command. Expected: seed and frequency tests pass.

- [ ] **Step 5: Write failing calibration tests**

Test equal-shape validation, RGB-only output, bitwise unchanged Thermal input, background/padding behavior, channel-wise epsilon bound under overlapping support, no required RGB objectness gate, diagnostic shapes, and non-zero detection gradients to seed, router, frequency, and residual paths.

- [ ] **Step 6: Run calibration tests and verify RED**

Run the Task 1 command. Expected: failures name the missing calibration class/behavior.

- [ ] **Step 7: Implement detail prototypes, local band association, router, and normalized residual broadcast**

Use valid normalized 3x3/7x7 pooling, dense local `unfold` attention, `fold` aggregation, one application of the soft seed gate, softmax band routing, sigmoid reliability, and `epsilon * support * tanh(avg)`.

- [ ] **Step 8: Run all PRLDFC tests and verify GREEN**

Run the Task 1 command. Expected: all PRLDFC tests pass.

- [ ] **Step 9: Commit**

Commit message: `Add prototype-routed frequency calibration core`.

---

### Task 3: COXNet integration and controlled configs

**Files:**
- Modify: `mmdet/models/utils/__init__.py`
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Modify: `tests/test_models/test_utils/test_prldfc.py`
- Create: `configs/coxnet/prldfc/PRLDFC.py`
- Create: `configs/coxnet/prldfc/PRLDFC_p3.py`
- Create: `configs/coxnet/prldfc/same_stage_no_calibration.py`

**Interfaces:**
- `FusionLayer.__init__` accepts `use_prldfc=False`, `prldfc_cfg=None`.
- `FusionNetXO.__init__` forwards the same arguments.
- `FusionLayer.forward` returns weighted `loss_prldfc_seed`, `loss_prldfc_offset`, `loss_prldfc_scale`, `loss_prldfc_cardinality` plus detached monitors during training.
- Inference stores detached level diagnostics in `last_prldfc_aux`.

- [ ] **Step 1: Write failing integration tests**

Test replacement mutual exclusion, equal-stage validation, P3/P4 application, unchanged Thermal inputs to HOFM, weighted auxiliary losses, padding propagation, absence of active `idwt_layers`, config construction, and canonical config values.

- [ ] **Step 2: Run integration tests and verify RED**

Run the Task 1 command. Expected: failures name missing `use_prldfc` plumbing/configs.

- [ ] **Step 3: Integrate PRLDFC and add configs**

Preserve RNG state around module construction as existing same-stage replacements do. Build valid masks from `img_metas`, pass GT only during training, and leave P5/P6 uncalibrated.

- [ ] **Step 4: Run PRLDFC and legacy focused tests**

Run: `PYTHONPATH=/tmp/prldfc-test-deps conda run -n bf python -m pytest -o addopts="" tests/test_models/test_utils/test_prldfc.py tests/test_models/test_utils/test_topc.py tests/test_models/test_utils/test_oepc.py tests/test_models/test_utils/test_trpc.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

Commit message: `Integrate PRLDFC as a same-stage CLFM replacement`.

---

### Task 4: Method documentation and reproducible entry points

**Files:**
- Create: `docs/PRLDFC.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-28-prldfc-design.md` only if an implementation ruling must be documented.

**Interfaces:**
- Produces documented seed-0 commands for canonical, P3-only, and no-calibration control.
- States that validation is structural and does not claim AP improvement or completed training.

- [ ] **Step 1: Document data flow, losses, diagnostics, controls, and commands**

Replace TOPC as the current experimental method in the README without deleting its reproducibility note. Link the design and implementation-plan documents.

- [ ] **Step 2: Verify documentation and config imports**

Run: `PYTHONPATH=/tmp/prldfc-test-deps conda run -n bf python -m pytest -o addopts="" tests/test_models/test_utils/test_prldfc.py -q`

Expected: all PRLDFC contract/config tests pass.

- [ ] **Step 3: Commit**

Commit message: `Document PRLDFC experiment and controls`.

---

### Task 5: Whole-branch verification and publication

**Files:**
- No production files unless a failing verification first receives a regression test.

**Interfaces:**
- Consumes all previous task outputs.
- Produces one reviewed commit range suitable for both requested remotes.

- [ ] **Step 1: Run syntax and focused CPU verification**

Run `compileall`, PRLDFC tests, and TOPC/OEPC/TRPC focused regression tests. Then run bare pytest and record every unrelated collection incompatibility rather than hiding it.

- [ ] **Step 2: Run model construction and GPU smoke**

Construct the canonical config and run one forward/backward/optimizer step with `CUDA_VISIBLE_DEVICES=1` in `coxmamba`. If GPU access is unavailable, report that boundary and do not call GPU verification complete.

- [ ] **Step 3: Request fresh whole-branch code review**

Use the review package from the implementation base through HEAD. Fix Critical/Important findings with RED-to-GREEN tests; ledger Minor findings.

- [ ] **Step 4: Verify remote histories and push**

Fetch both requested repositories, refuse non-fast-forward history silently, and push the verified feature branch/commit to `VIPLAB-Gachon/TRPC-Thermal-Referenced-Prototype-Calibration` and `loosiu/COXNet_develop`. Verify both remote branch SHAs with `git ls-remote`.
