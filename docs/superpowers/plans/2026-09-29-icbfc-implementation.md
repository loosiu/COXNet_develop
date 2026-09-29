# ICBFC Implementation and GPU-0 Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Instance-Conditioned Cross-Band Frequency Calibration as a cross-stage COXNet CLFM replacement, document it on `COXNet_develop/main`, push without rewriting remote history, and launch fresh seeds 0, 1, and 2 sequentially on physical GPU 0.

**Architecture:** A stride-4 Thermal instance-prior head supplies variable-count object centers and support scales. Each COXNet cross-stage pair retains RGB DeConv, applies Haar DWT to RGB and Thermal, pools eight modality-band tokens per instance, models a Thermal-to-RGB `4 x 4` relation, sparse-routes RGB target bands, and reconstructs a normalized RGB residual before the unchanged AAM/HOFM. The implementation is transplanted onto the latest `COXNet_develop/main` while preserving its legacy TPSC, PRLDFC, TOPC, OEPC, and TRPC code.

**Tech Stack:** Python 3.9, PyTorch 1.10, MMDetection/MMCV, pytest/unittest, Bash/tmux, Git/GitHub SSH.

**Spec:** `docs/superpowers/specs/2026-09-29-instance-conditioned-cross-band-frequency-calibration-design.md`

## Global Constraints

- Start the implementation worktree from `COXNet_develop` remote `main`, observed at `3da7f5e`, and re-fetch before work begins.
- Use the `dwt-dfca-oneway` implementation only as the cross-stage DeConv/Haar reference; do not replace the latest `main` tree or delete legacy methods.
- Preserve RGB FPN `start_level=2`, Thermal FPN `start_level=1`, four cross-stage pairs, RGB x2 DeConv, unchanged Thermal values, AAM/HOFM, `wf_loss=True`, GFL, QLSAssigner, NMS, dataset, and evaluation settings.
- Add ICBFC as a mutually exclusive CLFM replacement with `use_clfm=[]` and `use_icbfc=True`.
- Use the raw Thermal backbone stride-4 feature only for instance priors; do not add stride 4 to the detector head.
- Do not use a fixed candidate top-k. Process every valid local maximum above threshold in chunks; zero candidates must yield the exact DeConv RGB identity.
- Use GT instances for training-time token extraction and predicted variable-count instances for inference. Supervise center, offset, and log-scale only; do not add band labels or alignment losses.
- Initialize the RGB residual projection with Kaiming normal and use fixed `alpha=0.1`; do not restore the DWT-DFCA `std=1e-3` output bottleneck.
- Preserve the active GPU-1 DWT-DFCA seed queue and its worktree.
- GPU-0 training must start fresh with no `--auto-resume`, use separate seed work directories, a GPU-0 lock, and sequential seeds `0 1 2`.
- Push with a normal fast-forward `git push origin HEAD:main`; never force-push.
- README style may follow SafeDrive's centered title, badges, News, Framework, Results, and Quick Start sections, but must not copy its prose or invent an acceptance, checkpoint, result, or citation.

## Review Focus

- Two GT centers that collide at stride 8 but differ at stride 4 must remain two training instances; Task 2 pins this.
- Padded image rows and columns must neither create predicted instances nor enter Gaussian pooling; Tasks 2 and 4 pin this.
- A crowded image with more than 100 peaks must retain every thresholded local maximum and process them in chunks; Tasks 2 and 5 pin this.
- Adjacent/overlapping Gaussian supports must not scale the RGB residual with object count; Task 3 pins normalized reconstruction.
- Empty GT during training and zero predicted peaks during inference must remain finite and return a valid identity path; Tasks 2–4 pin this.

---

### Task 1: Create the isolated target-main workspace and carry the approved documents

**Files:**
- Create in target worktree: `docs/superpowers/specs/2026-09-29-instance-conditioned-cross-band-frequency-calibration-design.md`
- Create in target worktree: `docs/superpowers/plans/2026-09-29-icbfc-implementation.md`

**Interfaces:**
- Consumes: `origin/main` from `/data/siwoo/COXNet_develop` and approved documents from `/data/siwoo/COXNet-DWT-DFCA`.
- Produces: isolated `/data/siwoo/COXNet-ICBFC` worktree on branch `icbfc-main`, with the spec and this plan available to every later task.

- [ ] **Step 1: Re-fetch and verify the remote main base**

Run: `git fetch origin main && git rev-parse origin/main && git status --short` in `/data/siwoo/COXNet_develop`.

Expected: remote fetch succeeds; the source checkout has no user changes; record the actual `origin/main` SHA if it moved from `3da7f5e`.

- [ ] **Step 2: Create the isolated worktree**

Run: `git worktree add /data/siwoo/COXNet-ICBFC -b icbfc-main origin/main` from `/data/siwoo/COXNet_develop`.

Expected: a clean worktree on `icbfc-main`; no files under `/data/siwoo/COXNet-DWT-DFCA` change.

- [ ] **Step 3: Add the approved spec and plan using `apply_patch`**

Copy the exact reviewed Markdown contents into the target worktree without editing production code.

- [ ] **Step 4: Verify and commit the documentation baseline**

Run: `git diff --check && git status --short`.

Expected: only the two approved documents are new and whitespace checks pass.

Commit: `docs: specify ICBFC implementation`

---

### Task 2: Implement frequency, sparse-routing, target, and candidate primitives

**Files:**
- Create: `mmdet/models/utils/icbfc.py`
- Create: `tests/test_models/test_utils/test_icbfc.py`

**Interfaces:**
- Consumes: Torch tensors and `img_metas` geometry.
- Produces: `haar_dwt(tensor)`, `haar_idwt(ll, lh, hl, hh)`, `sparsemax(logits, dim=-1)`, `build_instance_targets(gt_bboxes, img_metas, output_size, device)`, and `extract_instance_candidates(center_logits, offsets, log_scales, valid_mask, score_threshold, chunk_size)`.

- [ ] **Step 1: Write failing primitive tests**

Add tests named:

- `test_haar_round_trip_preserves_values_dtype_and_device`
- `test_sparsemax_is_normalized_sparse_and_differentiable`
- `test_stride4_targets_keep_centers_that_stride8_would_merge`
- `test_candidate_extraction_keeps_more_than_100_valid_peaks`
- `test_candidate_extraction_removes_local_duplicates_and_padding`
- `test_candidate_extraction_allows_zero_candidates`

Use literal tensors and hand-derived center coordinates. The `>100` case must assert the exact untruncated count, not source text.

- [ ] **Step 2: Run the tests and observe RED**

Run: `conda run -n coxmamba python -m pytest -o addopts="" tests/test_models/test_utils/test_icbfc.py -q`

Expected: FAIL because `mmdet.models.utils.icbfc` does not exist.

- [ ] **Step 3: Implement the minimal primitives**

Haar functions must be device-agnostic and orthonormal. `sparsemax` must follow the projection-to-simplex algorithm without a new dependency. Target generation must return center heatmap, offset, log-scale, regression mask, and valid mask. Candidate extraction must use local peak suppression plus threshold, preserve continuous offsets/scales, reject invalid padding, return a per-image variable-length list, and use chunk metadata rather than truncation.

- [ ] **Step 4: Run the focused tests and observe GREEN**

Run the Task 2 pytest command.

Expected: all Task 2 tests pass.

- [ ] **Step 5: Commit**

Commit: `feat: add ICBFC instance and frequency primitives`

---

### Task 3: Implement the stride-4 Thermal instance prior

**Files:**
- Modify: `mmdet/models/utils/icbfc.py`
- Modify: `tests/test_models/test_utils/test_icbfc.py`

**Interfaces:**
- Consumes: `thermal_s4: Tensor[B,C,H,W]`, `gt_bboxes`, and `img_metas`.
- Produces: `ThermalInstancePrior.forward(thermal_s4, gt_bboxes=None, img_metas=None, return_loss=False) -> (instances, aux)` where each instance dictionary contains `centers`, `scales`, `scores`, and `batch_index`; `aux` contains dense predictions, finite diagnostics, and optional `loss_icbfc_center`, `loss_icbfc_offset`, `loss_icbfc_scale`.

- [ ] **Step 1: Write failing prior-head tests**

Add tests named:

- `test_training_prior_uses_every_gt_instance_and_returns_three_losses`
- `test_empty_gt_prior_losses_are_finite`
- `test_prior_predictions_exclude_padding`
- `test_center_offset_and_scale_parameters_receive_nonzero_gradients`
- `test_inference_prior_returns_variable_candidate_counts`

The first test must include two GT centers that differ at stride 4 but share a stride-8 cell.

- [ ] **Step 2: Run the prior tests and observe RED**

Run the Task 2 pytest command.

Expected: FAIL because `ThermalInstancePrior` is missing.

- [ ] **Step 3: Implement `ThermalInstancePrior`**

Use a shared `3x3 Conv + GroupNorm + ReLU` stem and separate `1x1` center, offset, and log-scale heads. Initialize center bias from prior probability `0.01`; training routing instances come from continuous GT geometry, while dense predictions receive CenterNet focal, masked L1 offset, and masked smooth-L1 log-scale losses. Inference calls Task 2 candidate extraction and does not force a candidate when none pass.

- [ ] **Step 4: Run the prior tests and observe GREEN**

Run the Task 2 pytest command.

Expected: all primitive and prior tests pass.

- [ ] **Step 5: Commit**

Commit: `feat: add thermal instance prior for ICBFC`

---

### Task 4: Implement instance cross-band relation, routing, and reconstruction

**Files:**
- Modify: `mmdet/models/utils/icbfc.py`
- Modify: `tests/test_models/test_utils/test_icbfc.py`

**Interfaces:**
- Consumes: `thermal`, lower-resolution `visible`, and Task 3 instance dictionaries.
- Produces: `ICBFCLevel.forward(thermal, visible, instances, return_aux=False) -> visible_out` or `(visible_out, aux)`; `aux` exposes relation/router/support/residual diagnostics and tensors required by losses only when requested.

- [ ] **Step 1: Write failing level tests**

Add tests named:

- `test_level_extracts_eight_tokens_per_instance`
- `test_relation_is_four_by_four_and_rows_sum_to_one`
- `test_off_diagonal_relation_changes_rgb_output`
- `test_sparse_router_can_choose_different_bands_per_instance`
- `test_overlapping_supports_are_normalized_not_summed`
- `test_no_instances_return_exact_deconv_identity`
- `test_level_preserves_thermal_and_matches_thermal_shape`
- `test_first_backward_reaches_qkv_router_output_and_deconv`
- `test_initial_delta_ratio_is_finite_nonzero_and_below_point_two`

Use a synthetic two-instance fixture with distinct band patterns. The no-instance assertion must compare output directly with `module.deconv(visible)`.

- [ ] **Step 2: Run the level tests and observe RED**

Run the Task 2 pytest command.

Expected: FAIL because `ICBFCLevel` is missing.

- [ ] **Step 3: Implement `ICBFCLevel`**

Own `TransBasicConv2d` for x2 RGB DeConv. Apply DWT to both modalities, map stride-4 continuous instance geometry to each half-resolution band lattice, and Gaussian-pool four RGB plus four Thermal tokens. Use RGB target-band Q and Thermal source-band K/V with a learnable noisy `4 x 4` bias. Produce four correction tokens, route them with sparsemax, scatter them with overlap-normalized Gaussian masks, IDWT, Kaiming-normal `out_proj`, `tanh`, and fixed `alpha=0.1`.

Process candidates in `candidate_chunk_size` chunks without altering the mathematical result. Store detached diagnostics including candidate count, support/overlap, relation diagonal/off-diagonal mass and entropy, router entropy/active bands/per-instance variance/per-band means, and `delta_ratio`.

- [ ] **Step 4: Run the level tests and observe GREEN**

Run the Task 2 pytest command.

Expected: all ICBFC unit tests pass.

- [ ] **Step 5: Commit**

Commit: `feat: add instance-conditioned cross-band calibration`

---

### Task 5: Integrate ICBFC into current COXNet main and add its config

**Files:**
- Modify: `mmdet/models/utils/__init__.py`
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Create: `configs/coxnet/icbfc/ICBFC.py`
- Create: `tests/test_models/test_utils/test_icbfc_fusion.py`
- Create: `tests/test_models/test_detectors/test_icbfc_config.py`

**Interfaces:**
- Consumes: Task 3 `ThermalInstancePrior`, Task 4 `ICBFCLevel`, latest-main FusionLayer auxiliary-loss conventions, and raw dual-backbone outputs.
- Produces: `use_icbfc: bool`, `icbfc_cfg: dict`, four cross-stage ICBFC levels, training auxiliary losses/diagnostics, and inference with `img_metas` padding control.

- [ ] **Step 1: Write failing FusionLayer tests**

Add tests named:

- `test_fusion_layer_builds_one_prior_and_four_levels`
- `test_fusion_layer_returns_four_thermal_resolution_features`
- `test_training_returns_center_losses_and_nonoptimized_diagnostics`
- `test_fusion_rejects_icbfc_with_other_clfm_replacements`
- `test_more_than_100_instances_complete_in_chunks_without_truncation`

- [ ] **Step 2: Write failing detector/config tests**

Add tests named:

- `test_icbfc_config_preserves_cross_stage_baseline_contract`
- `test_icbfc_config_builds_detector`
- `test_detector_passes_raw_stride4_thermal_and_img_metas`
- `test_simple_test_excludes_padding_candidates`
- `test_parse_losses_does_not_optimize_icbfc_diagnostics`
- `test_baseline_and_legacy_configs_still_build`

The config-contract test must assert RGB/Thermal FPN start levels `2/1`, head strides `[8,16,32,64]`, `wf_loss=True`, QLSAssigner, NMS IoU `0.3`, `use_clfm=[]`, and `use_icbfc=True`.

- [ ] **Step 3: Run integration tests and observe RED**

Run: `conda run -n coxmamba python -m pytest -o addopts="" tests/test_models/test_utils/test_icbfc_fusion.py tests/test_models/test_detectors/test_icbfc_config.py -q`

Expected: FAIL because ICBFC is not registered or accepted by FusionLayer/FusionNetXO.

- [ ] **Step 4: Integrate the prior and four levels**

Extend current-main exclusivity checks without deleting TPSC/PRLDFC/TOPC/OEPC/TRPC. Preserve raw Thermal backbone stage 0 before `neck_t`, pass it only to ICBFC, and preserve all legacy call signatures through optional arguments. Follow the current auxiliary-loss return convention. Aggregate diagnostics over non-empty levels and prefix them `icbfc_`; keys beginning `loss_icbfc_` are the only new optimized terms.

- [ ] **Step 5: Add the canonical config**

Inherit `../coxnet_r50_fpn_1x_rgbtdroneperson.py`; set `use_clfm=[]`, `use_icbfc=True`, explicit ICBFC hyperparameters, and `work_dir='work_dir/coxmamba/rgbtdroneperson/icbfc'`. Change no detector, assignment, postprocessing, or dataset setting.

- [ ] **Step 6: Run all ICBFC tests and observe GREEN**

Run: `conda run -n coxmamba python -m pytest -o addopts="" tests/test_models/test_utils/test_icbfc.py tests/test_models/test_utils/test_icbfc_fusion.py tests/test_models/test_detectors/test_icbfc_config.py -q`

Expected: all ICBFC tests pass.

- [ ] **Step 7: Commit**

Commit: `feat: integrate ICBFC into COXNet`

---

### Task 6: Add GPU smoke validation and the fresh sequential GPU-0 launcher

**Files:**
- Create: `tools/misc/smoke_icbfc.py`
- Create: `tools/run_icbfc_seeds_gpu0.sh`
- Create: `tests/test_tools/test_icbfc_launcher.py`

**Interfaces:**
- Consumes: canonical Task 5 config and local RGBTDronePerson data.
- Produces: a real-batch GPU smoke report and a lock-protected seed `0 -> 1 -> 2` launcher.

- [ ] **Step 1: Write failing launcher behavior tests**

Run the script with `ICBFC_DRY_RUN=1` and assert observable output contains exactly three ordered seed commands, physical GPU `0`, distinct work directories, the canonical config, `--deterministic`, and no `--auto-resume`. Assert the script uses `/tmp/coxnet_icbfc_gpu0.lock` when not dry-running.

- [ ] **Step 2: Run the launcher test and observe RED**

Run: `conda run -n coxmamba python -m pytest -o addopts="" tests/test_tools/test_icbfc_launcher.py -q`

Expected: FAIL because the launcher is absent.

- [ ] **Step 3: Implement the launcher and smoke script**

The launcher uses `/home/viplab/anaconda3/envs/coxmamba/bin/python`, `CUDA_VISIBLE_DEVICES=0`, seeds `0 1 2`, separate `work_dir/coxmamba/rgbtdroneperson/icbfc/seed{seed}` paths, `flock`, `tee`, and refuses a non-empty seed directory unless an explicit resume override is supplied. It must not use `--auto-resume` in the requested fresh run.

The smoke script builds the canonical model, loads one actual training batch, performs forward/backward, and fails unless all three prior heads, relation Q/K/V, router, output projection, and DeConv have finite non-zero gradients; it prints center count/recall, router variance, and `delta_ratio`.

- [ ] **Step 4: Run launcher test and dry run GREEN**

Run the Task 6 pytest command and `ICBFC_DRY_RUN=1 bash tools/run_icbfc_seeds_gpu0.sh`.

Expected: test passes; dry run prints three commands in seed order without starting training.

- [ ] **Step 5: Commit**

Commit: `chore: add reproducible ICBFC GPU-0 workflow`

---

### Task 7: Replace README presentation and add the method note

**Files:**
- Modify: `README.md`
- Create: `docs/ICBFC.md`

**Interfaces:**
- Consumes: verified commands, canonical config, method diagnostics, and honest experiment state from Tasks 2–6.
- Produces: a polished GitHub landing page and detailed reproducibility note.

- [ ] **Step 1: Write the method note**

Document the problem statement, module data flow, equations for `4 x 4` relation/router/reconstruction, training-vs-inference instance semantics, losses, diagnostics, falsification criteria, configs, commands, and limitations. Explicitly state that DWT is a primitive and that ICBFC is not the complete DyFCLT architecture.

- [ ] **Step 2: Rewrite README in the requested presentation style**

Use a centered project title/subtitle, truthful badges, concise News, a Mermaid or text Framework, method highlights, an experiment-status results table, Quick Start, dataset layout, training/evaluation commands, repository map, citation status, license, and acknowledgements. Keep the official COXNet citation and distinguish it from the new research method. Mark ICBFC accuracy as pending until checkpoints exist. Preserve links to legacy method notes without making them the landing-page focus.

- [ ] **Step 3: Validate documentation references**

Run a local link/path scan that checks every repository-relative path and command target named by README and `docs/ICBFC.md` exists.

Expected: no missing local files and no fabricated result/checkpoint claims.

- [ ] **Step 4: Commit**

Commit: `docs: present ICBFC method and workflow`

---

### Task 8: Verify the complete branch on CPU and GPU

**Files:**
- Modify only if a failure is reproduced by a new RED test: files owned by Tasks 2–6.

**Interfaces:**
- Consumes: complete ICBFC branch.
- Produces: fresh verification evidence and any TDD-backed fixes.

- [ ] **Step 1: Run focused CPU tests**

Run: `conda run -n coxmamba python -m pytest -o addopts="" tests/test_models/test_utils/test_icbfc.py tests/test_models/test_utils/test_icbfc_fusion.py tests/test_models/test_detectors/test_icbfc_config.py tests/test_tools/test_icbfc_launcher.py -q`

Expected: zero failures.

- [ ] **Step 2: Compile and build**

Run: `conda run -n coxmamba python -m compileall -q mmdet/models/utils/icbfc.py mmdet/models/utils/fusion_strategy.py mmdet/models/detectors/fusionnet_xo.py tools/misc/smoke_icbfc.py` and build the canonical detector from `Config.fromfile`.

Expected: exit 0 and four ICBFC levels.

- [ ] **Step 3: Run the broader repository suite and record inherited failures**

Run: `conda run -n coxmamba python -m pytest -o addopts="" tests -q` with full output saved outside the repo.

Expected: report every failure. Unrelated upstream collection/dependency failures are not hidden; any new ICBFC or previously passing legacy failure blocks completion.

- [ ] **Step 4: Re-check physical GPU 0 and run real-batch smoke**

Verify GPU index 0 is not occupied by another user job, then run `CUDA_VISIBLE_DEVICES=0 /home/viplab/anaconda3/envs/coxmamba/bin/python tools/misc/smoke_icbfc.py --config configs/coxnet/icbfc/ICBFC.py`.

Expected: forward/backward succeeds, every required gradient is finite and non-zero, candidates are active, router variance is non-zero after the controlled smoke optimization, and `0 < delta_ratio < 0.2`.

- [ ] **Step 5: Verify repository scope**

Run: `git diff --check origin/main...HEAD`, inspect `git diff --stat`, ensure no `data/`, checkpoint, work directory, or unrelated generated artifact is tracked, and verify the active GPU-1 worktree remains unchanged.

Expected: only ICBFC, its docs/tests/config/tools, and necessary integration changes.

- [ ] **Step 6: Commit any TDD-backed verification fixes**

Commit only if Step 1–5 required fixes; otherwise leave the verified HEAD unchanged.

---

### Task 9: Push fast-forward to main and launch fresh seeds 0, 1, and 2

**Files:**
- External state: `git@github.com:loosiu/COXNet_develop.git` branch `main`
- Runtime output: `work_dir/coxmamba/rgbtdroneperson/icbfc/seed{0,1,2}`

**Interfaces:**
- Consumes: Task 8 verified HEAD and an idle physical GPU 0.
- Produces: updated remote main and a live tmux session `icbfc_gpu0` running the sequential launcher.

- [ ] **Step 1: Verify push is fast-forward immediately before pushing**

Run: `git fetch origin main`, verify `git merge-base --is-ancestor origin/main HEAD`, inspect `git log --oneline origin/main..HEAD`, and confirm the remote main SHA has not moved outside the branch history.

Expected: HEAD is a strict fast-forward of current remote main.

- [ ] **Step 2: Push and verify remote SHA**

Run: `git push origin HEAD:main`, then `git ls-remote origin refs/heads/main`.

Expected: remote main equals local verified HEAD; no force option is used.

- [ ] **Step 3: Verify fresh training targets and GPU availability**

Check physical GPU 0, tmux session names, lock state, and each seed work directory. If a target directory contains a checkpoint, archive it to a timestamped sibling rather than delete it.

Expected: GPU 0 is available and seed 0 starts from epoch 1 with no resume checkpoint.

- [ ] **Step 4: Launch sequential training**

Run: `tmux new-session -d -s icbfc_gpu0 'cd /data/siwoo/COXNet-ICBFC && exec bash tools/run_icbfc_seeds_gpu0.sh'`.

Expected: one parent launcher and one seed-0 trainer use physical GPU 0; seeds 1 and 2 remain queued in the same script.

- [ ] **Step 5: Verify live activation evidence**

Inspect tmux, `nvidia-smi`, process arguments, the seed-0 console log, and the first available ICBFC diagnostics. Verify the configured work directory and seed, and distinguish process launch from completed training.

Expected: seed 0 is genuinely running from epoch 1; candidate count/support, relation/router statistics, and `delta_ratio` are finite and non-degenerate. Do not claim AP or training completion.
