# Project instructions

## Required environment

- Activate the `sehwa` conda environment before working in this repository, including running Git, GitHub, Hugging Face, Python, and uv commands.
- This server is shared. The user's GitHub and Hugging Face account configuration is scoped to `sehwa`; use that environment for all account access. Do not change shared or other users' authentication settings.
- Shell activation does not persist between separate command sessions. Start each new shell session with:

  ```bash
  source /workspace/miniforge3/etc/profile.d/conda.sh
  conda activate sehwa
  ```

- The uv environment setup described in `README.md` has already been completed. Use the existing setup; do not repeat installation or recreate environments unless the task requires it.
- Run the appropriate project command after activating `sehwa`, including `uv run` when using the project's uv environment.

## GPU usage

- Do not use GPUs unless the user explicitly requests GPU usage for the task.
- Implementing code does not authorize running it on a GPU for validation. Use CPU-only checks by default; leave GPU-dependent validation unrun unless explicitly requested.
- For validation commands that may initialize an accelerator, explicitly disable GPU access (for example, `CUDA_VISIBLE_DEVICES=""`; also use `JAX_PLATFORMS=cpu` for JAX) and select CPU devices in the relevant framework.

## Storage locations

- The repository root already contains `datasets/`, `checkpoints/`, and `experiments/`. Use these directories for their respective artifacts.
- Store datasets in `datasets/`.
- Store model checkpoints in `checkpoints/`.
- Store experiment results, including logs, metrics, plots, and reports, in `experiments/`.
- Configure download, training, and evaluation output paths to follow these locations. Resolve these paths relative to the repository root, even when running commands from a subdirectory.
- Keep `datasets/`, `checkpoints/`, and `experiments/` ignored by Git. Do not force-add or commit their contents.

## Project orientation

- `README.md` describes the openpi-based RL Token pipeline and stage-1 training and serving entry points.
- `rlt_online_rl/README.md` describes the real-robot online RL runtime, launch order, and evaluation flow.
- Core model and training code lives in `src/openpi/`; the online RL runtime lives in `rlt_online_rl/`.

## TRLT context and session handoff

Before continuing TRLT work in a new session:

1. Activate `sehwa`, inspect Git status and the current branch/commit, and preserve existing changes. TRLT work was implemented on branch `TRLT`; do not reset or switch away from a user's checkout automatically.
2. Read [`docs/trlt/README.md`](docs/trlt/README.md) for the architecture, objectives, tensor/time contract, configurations, CLI commands, and planned experiment stages.
3. Read [`docs/trlt/validation.md`](docs/trlt/validation.md) for verification procedures and the locations of actual execution evidence.
4. Inspect the relevant local manifests, logs, caches, and checkpoints before stating what has completed. Documentation examples are not evidence that a command ran. Artifact paths below are relative to the repository root and are intentionally Git-ignored; they may be absent on another checkout.

Implementation map:

- `src/openpi/models/trlt.py`: TRLT CNN, fusion schedules, readout, tactile decoder, and reconstruction objective. `src/openpi/models/rl_token.py` supplies the existing RLT encoder/visual decoder.
- `src/openpi/trlt/`: common config/schema, T-Rex adapter and preparation, frozen feature/cache handling, representation/probe training, and evaluation metrics.
- `configs/trlt/`: the four tracked configuration examples. `scripts/prepare_trex.py`, `scripts/cache_trex_features.py`, and `scripts/trex_offline.py` are the main execution entry points.

### Recorded milestone: 2026-11-14

This is a historical handoff, not a live process-status report. Verify later artifacts before using it as current state.

- Exactly four comparisons are implemented: `rlt_original`, `trlt_self_only`, `trlt_cross_only`, and `trlt_alternating`. Stage A learns reconstruction; Stage B freezes the representation and trains a supervised action MLP. Online RL, action refinement, and action-expert training/inference are outside scope.
- Dataset revision: `8b478400a9167d59ede0fdbfea616cb5e9791885`. Squeeze selection contains 173 demonstrations / 152,169 frames. Fixed object-stratified episode split, seed 42: train 121, validation 26, test 26. Train has 35,597 query windows. Reuse saved splits and train-only normalization; do not resplit per model/seed or use whole-dataset statistics.
- Exact-file source download, dataset validation, split/normalization preparation, and bounded synthetic/real-data GPU checks were run. Evidence: `experiments/trlt/validation/implementation_report.md`, `experiments/trlt/implementation_manifest.json`, and dataset manifests described in `docs/trlt/validation.md`. These smoke checks are not benchmark results.
- Frozen visual cache extraction completed for train and validation: 7,174 + 1,483 = 8,657 anchors, float32 `[768,2048]` features with masks, in `datasets/trex_squeeze/visual_cache/`. Cache hash: `ebf5712beaa2c080cdd1ad555a955089b9a27d72fa46e7015d5e0c004f53814e`. Evidence: `experiments/trlt/visual_cache_extraction/report.md` and `cache_validation.json` in the same directory. Test features were not extracted as part of this milestone. Do not automatically rebuild the completed cache.
- Full batch 32 / microbatch 32 / accumulation 1 passed bounded GPU optimizer checks on the 96 GB RTX PRO 6000; peak observed TRLT process memory was about 49.5 GiB in those checks. See `experiments/trlt/batch32_check/report.md`. Local run configs are `experiments/trlt/configs/*_b32.json`; tracked defaults in `configs/trlt/` still use effective batch 32 / microbatch 4 / accumulation 8. These are the requested offline training defaults, not a reproduction of the legacy RLT optimizer recipe. There are 1,113 optimizer updates per full train epoch at batch 32, including the partial final batch.
- The user manually launched `trlt_alternating_b32.json`, seed 0, device 0, with `--max-steps 5000`. Do not infer completion or a running process from that launch. Inspect `experiments/trlt/trlt_alternating/seed0/representation/` and `checkpoints/trlt/trlt_alternating/seed0/representation/`, plus live processes/GPU state when relevant.
- Training-idle diagnosis identified missing console progress and CPU data preparation, including repeated Parquet reads. `runner.py` now prints initialization/JIT/step/validation progress and records data/update times; `data.py` retains up to 256 small episode tables rather than 2. Changes apply to newly launched Python processes only. CPU sample-equivalence checks and all 15 data tests passed; see `experiments/trlt/training_diagnostics/2026-11-14.json`. The user's running job was not stopped or restarted.

### Remaining experiment workflow and maintenance

- The planned comparison is Stage A representation training, best validation checkpoint selection, frozen latent export, Stage B action-probe training/selection, then final test evaluation for all four models and seeds 0/1/2. No CLI command should implicitly launch the next stage.
- The user reserved long training and final evaluation for their own decision after reviewing implementation. Do not automatically start long runs, extract final test features, or perform final test evaluation. A manually launched user job does not authorize launching additional jobs. Before GPU work, follow the GPU rules above and check for existing processes; do not duplicate or interrupt the user's job.
- Preserve the fixed temporal contract: 30 Hz; queries every 3 frames; three-camera anchor refresh every 15 frames; tactile history `[t-9,...,t]` across 10 deformation sensors; current 58D state; commanded action targets `[t,...,t+5]`, with boundary masks. All models share splits, visual features, normalization, and sampling. See the README for exact masks and ordering.
- Keep the test split out of debugging, training, checkpoint selection, and hyperparameter selection. The primary final metric is normalized action MSE averaged within each episode, then equally across test episodes. Report per-seed values and aggregates; do not claim online control performance from the offline probe.
- For resume, first inspect the actual `last` checkpoint and matching config/provenance; logs alone are not a saved optimizer state. Do not prescribe `--resume` merely because a run was launched.
- After meaningful implementation or protocol changes, update the relevant `docs/trlt/` Markdown and this handoff if its pointers or milestones changed. Store generated execution results, manifests, timings, and metrics under `experiments/trlt/`; link them from documentation. Record date, exact config/command, device when used, actual steps, and pass/fail/blocked/not-run status. Clearly distinguish completed checks, user-started runs, and planned experiments. Keep this file an orientation index rather than copying full logs into it.

### Training observability: 2026-11-14

- TRLT Stage A/B now have tqdm train/validation bars and opt-in W&B tracking;
  see `docs/trlt/README.md` (Training progress and W&B). Implementation is in
  `src/openpi/trlt/tracking.py` and `runner.py`; CLI flags are in `scripts/trex_offline.py`.
- These changes apply only to new processes. No existing user job was restarted.
- CPU/offline validation evidence lives in `experiments/trlt/tracking_validation/`;
  consult its report for actual results. Online W&B authentication/upload and GPU
  execution were not validated as part of this change.


### Training data prefetch: 2026-11-14

- Stage A/B now use spawned Torch DataLoader CPU workers and bounded prefetch,
  following the existing RLT/pi trainer pattern. Defaults:
  `--num-workers 2 --prefetch-factor 2`; `--num-workers 0` selects synchronous
  debugging and does not use prefetch. `num_workers` counts CPU loader processes;
  `prefetch_factor` counts batches dispatched ahead per worker. They are
  independent and need not match. At most their product is dispatched ahead per
  loader, including work in progress (not necessarily all ready in memory).
  More workers increase CPU loading concurrency; a larger prefetch factor adds
  buffering and host memory use, without changing batch/microbatch size. The
  defaults are not benchmark-tuned; GPU throughput remains unmeasured.
  Loader implementation: `src/openpi/trlt/loading.py`; CLI: `scripts/trex_offline.py`.
- Gradients, finite checks, sequential microbatch accumulation and optimizer
  updates run in one JIT function. Full gradients no longer transfer to the CPU
  each step. Per-step metric synchronization remains. Fixed sampling/masks,
  partial batches and the consumed resume cursor are preserved; loader options
  are recorded as runtime metadata outside checkpoint identity.
- Cold tactile cache creation uses per-cache file locks for multiple workers.
  Existing caches are reused. Changes apply only to newly launched processes;
  the user's running job was not restarted or interrupted.
- CPU-only validation artifacts and exact commands:
  `experiments/trlt/loading_validation/report.md` and `summary.json` in that
  directory. Recorded result: 33 regression tests plus one explicit worker CPU
  isolation test passed; 8 actual train queries matched exactly across synchronous
  and worker loading. Synthetic tests completed 15 successful optimizer updates.
  GPU throughput, utilization and long training were not validated.

### W&B metric groups: 2026-11-14

- W&B now groups charts under `progress`, `timing`, `stage_a`, and `stage_b`,
  each with train/validation where applicable. Aggregation counts and mask valid
  fractions remain local JSONL diagnostics and are excluded from W&B.
- Existing uploaded history is preserved. See `docs/trlt/README.md` and CPU-only
  evidence in `experiments/trlt/tracking_groups_validation/`.
