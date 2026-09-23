# TOPC Implementation and Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, verify, push, and sequentially train seed 0/1/2 of Thermal-Anchored Object Prototype Calibration on GPU 1.

**Architecture:** Thermal P3 objectness selects candidate-specific heatmap-weighted prototypes. Each prototype softly searches a radius-2 RGB P3 neighborhood, and its Thermal-minus-RGB discrepancy is projected back only through that attention support with normalized cosine confidence. Calibrated RGB P3 and untouched Thermal P3 enter the original AAM/HOFM.

**Tech Stack:** Python, PyTorch, MMCV/MMDetection, pytest, COXNet FusionNetXO, conda `coxmamba`, Git, GPU 1.

**Spec:** `docs/superpowers/specs/2026-09-23-topc-design.md`

## Global Constraints

- Work only in `/data/siwoo/COXNet-OEPC-v2`; preserve active worktrees/jobs.
- Create `topc.py`; preserve legacy `oepc.py` and TRPC behavior.
- P3 only; no P4, CLFM, DWT/IDWT, DeConv, EDL, utility, gate, FiLM, contrast, cycle.
- Only `loss_topc_objectness = 0.1 * focal_heatmap_loss` is added to detection loss.
- Push verified code to `origin/main` without force, then run seeds 0→1→2 sequentially on GPU 1.

## Review Focus

- Boundary/padding normalization for 3x3 Thermal prototypes and 5x5 RGB search.
- Adjacent candidates must survive without local-peak NMS.
- Cosine confidence must be bounded `[0,1]` and not become a second learned router.
- One candidate must retain attention magnitude; overlapping candidates must not amplify residual unboundedly.
- Empty/no-candidate batches must be finite identity operations.

---

### Task 1: Gaussian objectness and candidate selection

**Files:**
- Create: `mmdet/models/utils/topc.py`
- Create: `tests/test_models/test_utils/test_topc.py`
- Modify: `mmdet/models/utils/__init__.py`

**Interfaces:**
- `build_topc_gaussian_targets(gt_boxes, padded_size, feat_size, device, valid_mask=None, max_radius=2)`.
- `ThermalAnchoredObjectPrototypeCalibration._select_candidates(logits, valid)` returns fixed-size indices, scores, active mask, and dense candidate mask.

- [ ] Write Gaussian tests for fractional/adjacent centers, per-object maximum 1, max overlap, empty GT, and padding.
- [ ] Run `conda run -n coxmamba python -m pytest tests/test_models/test_utils/test_topc.py -k gaussian -q`; expect missing import failure.
- [ ] Implement the normalized Gaussian target exactly as the spec.
- [ ] Write candidate tests for Thermal-only logits, threshold/top-100, adjacent cell survival, padding, and empty selection.
- [ ] Run the candidate selection tests; expect missing method failure.
- [ ] Implement a Thermal-only depthwise-3x3/GELU/1x1 objectness head with prior 0.01 and no local peak NMS.
- [ ] Run the Task 1 tests and commit `Add TOPC thermal object discovery`.

### Task 2: Candidate prototypes and local RGB matching

**Files:**
- Modify: `mmdet/models/utils/topc.py`
- Modify: `tests/test_models/test_utils/test_topc.py`

**Interfaces:**
- `_thermal_prototypes(thermal, probability, selection, valid)` returns `[N,K,C]` prototypes, prototype mass, and active mask.
- `_local_rgb_match(rgb, prototypes, indices, active, valid)` returns RGB prototypes, attention, similarity, confidence, offsets, and valid options.

- [ ] Write hand-derived prototype tests for heatmap-weighted 3x3 pooling, boundaries, padding, zero mass, and one prototype per candidate.
- [ ] Run prototype tests; expect missing method failure.
- [ ] Implement unfold/gather weighted pooling with normalized valid heatmap mass.
- [ ] Write local matching tests for radius-2 search, Thermal-query direction, softmax sum 1, known weighted RGB prototype, invalid masking, and confidence bounds.
- [ ] Run matching tests; expect missing method failure.
- [ ] Implement RGB `1x1` semantic projection, cosine search, temperature 0.2 softmax, and normalized max-cosine confidence.
- [ ] Run Task 2 tests and commit `Add TOPC object prototypes and RGB matching`.

### Task 3: Discrepancy residual and full module

**Files:**
- Modify: `mmdet/models/utils/topc.py`
- Modify: `tests/test_models/test_utils/test_topc.py`

**Interfaces:**
- `_scatter_residual(residual, confidence, attention, indices, active, output_size)`.
- `forward(rgb, thermal, valid_mask=None, center_target=None, return_aux=False)` returns calibrated RGB and objectness loss/metrics.

- [ ] Write scatter tests proving single-candidate attention magnitude, normalized overlap, exact outside-support zero, and correct boundary offsets.
- [ ] Run scatter tests; expect missing method failure.
- [ ] Implement `W(P_T-P_R)` with small nonzero init and fold/scatter aggregation from the spec.
- [ ] Write forward/backward tests for unchanged Thermal, identity without candidates, only objectness auxiliary loss, absence of EDL/utility/gate/FiLM, and finite nonzero gradients to objectness, RGB projection, residual projection, RGB, and Thermal.
- [ ] Run forward tests; expect incomplete forward failure.
- [ ] Implement full forward, balanced focal heatmap loss, and documented detached diagnostics.
- [ ] Run `test_topc.py` and commit `Implement TOPC RGB semantic calibration`.

### Task 4: COXNet integration and canonical config

**Files:**
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Create: `configs/coxnet/topc/TOPC.py`
- Modify: `tests/test_models/test_utils/test_topc.py`

**Interfaces:**
- Adds `use_topc=False`, `topc_cfg=None`.
- Adds `loss_topc_objectness`; never returns a utility payload.
- Canonical work dir base: `work_dir/coxmamba/rgbtdroneperson/topc/seed0`.

- [ ] Write FusionLayer tests for replacement exclusion, P3-only call, untouched Thermal into HOFM, padding mask, exact loss weighting, and legacy construction.
- [ ] Run fusion tests; expect constructor/path failure.
- [ ] Integrate TOPC while preserving HOFM RNG initialization and existing OEPC/TRPC paths.
- [ ] Write config contract test for both `start_level=1`, CLFM/TRPC/OEPC off, TOPC on, P3-only defaults, only objectness weight 0.1, and no P4/utility/EDL keys.
- [ ] Create `configs/coxnet/topc/TOPC.py` with channels 256, object kernel 3, search radius 2, threshold 0.05, K 100, temperature 0.2, init std 1e-2, `wf_loss=False`.
- [ ] Run TOPC plus OEPC/TRPC regression tests and config build.
- [ ] Commit `Integrate TOPC into COXNet`.

### Task 5: Documentation, verification, push, and sequential training

**Files:**
- Modify: `README.md`
- Create: `docs/TOPC.md`

**Interfaces:**
- Canonical training config and three work dirs under `work_dir/coxmamba/rgbtdroneperson/topc/seed{0,1,2}`.

- [ ] Document the exact four-stage method, one auxiliary loss, command, diagnostics, distinction from legacy methods, and unmeasured performance status; commit `Document TOPC workflow`.
- [ ] Run focused TOPC/OEPC/TRPC/GFL tests, then repository `pytest -q`; record exact unrelated failures.
- [ ] Run `git diff --check`, compileall, config/model build, and a one-iteration GPU-1 smoke with workers 0 and a unique `/tmp/topc-smoke-*` work dir.
- [ ] Generate the review package from `69d9037` to HEAD; because subagent delegation is disabled, perform the required read-only self-review and fix Critical/Important findings test-first.
- [ ] `git fetch origin`; require `origin/main` ancestor of HEAD and clean status; push `HEAD:main`; verify remote/local hashes match.
- [ ] Re-query GPU 1, processes, tmux, and existing locks. Do not displace an active job.
- [ ] Create a Git-ignored fail-fast sequential launcher using the repository queue convention. Each command uses GPU 1, `coxmamba`, deterministic mode, config `TOPC.py`, and its seed-specific work dir. Seed N+1 starts only after seed N exit code 0.
- [ ] Launch detached and verify the real seed-0 Python child, GPU allocation, growing log, config dump, and first iteration. Report commit, queue/session/PID, log/work paths, and seeds 1/2 as queued rather than running.
