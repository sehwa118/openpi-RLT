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

### Experiment budget decision: 2026-11-14

- User selected Stage A **15,000 total optimizer updates**, no epoch limit and
  no early stopping, for all four architectures. The tracked `configs/trlt/`
  files now set `representation.max_steps=15000`, `max_epochs=null`,
  `patience=null`. Validation/best selection still occurs at epoch boundaries
  and at the final step. Stage B keeps all existing defaults (max100epochs,
  patience10, AdamW1e-3, batch256, hidden256×2).
- Python `TrainConfig` retains legacy epoch defaults; use the tracked JSON files
  for the selected experiment policy. Existing ignored `*_b32.json` run configs
  and running processes were not modified. Do not assume an old run uses the new
  policy, or resume it without inspecting saved config/provenance.
- CPU-only verification evidence and execution status:
  `experiments/trlt/stage_a_15000_validation/report.md`. This change does not
  authorize starting additional GPU jobs or final test evaluation.

### User-authorized experiment launch: 2026-11-14

- The user requested starting the full experiment, including training and final
  evaluation, then changed the scope to **one seed only**. Use seed0 for all four
  models; do not launch seeds1/2. Stage A remains15000updates, no epoch cap or
  early stopping. Stage B keeps the existing defaults.
- Active run index: `experiments/trlt/active_run.json`. Run directory:
  `experiments/trlt/runs/squeeze_15k_seed0_20261114T080906Z/`. An explicit28-stage
  queue is running in tmux socket `trlt-runs`, session `trlt-15k-seed0`, windows
  `train` and `reports`. Inspect `status.json`, `events.log`, `current_process.json`,
  actual stage logs and live processes before claiming progress or restarting.
- First launched stage: `rlt_original/seed0/Stage_A`, GPU0. Training startup was
  observed; later steps/completion must be verified. Per-run checkpoints are under
  `checkpoints/trlt/runs/<run_id>/`; latents are under
  `datasets/trex_squeeze/latents/<run_id>/`. New CLI `--log-dir` isolates training
  logs from earlier debug paths. Config snapshots keep effective batch32/micro4,
  loader workers2/prefetch2; W&B disabled.
- The explicit queue completes all four models' Stage A/B, seals the selected
  checkpoint hashes, then extracts shared test features and evaluates all four
  models. Source changes, occupied GPU or a failing stage prevent the next job
  from starting. Never duplicate the queue or resume without inspecting actual
  checkpoint/provenance. Stage start/end/failure reports persist in `events.log`;
  chat push notifications after the assistant turn ends are unavailable.

### W&B fresh-restart request: 2026-11-14

- User requested W&B online and then explicitly a fresh restart from step0.
  The previous disabled-W&B run was stopped at387updates; logs are preserved.
  It is no longer running. Its `superseded.json` records the replacement path.
- Prepared replacement is indexed by `experiments/trlt/active_run.json`:
  `experiments/trlt/runs/squeeze_15k_seed0_wandb_20261114T081820Z/`.
  All8 Stage A/B commands specify online W&B, seed0 and no resume. Stage A
  remains15000updates; Stage B defaults unchanged. Eight CPU-only CLI dry runs
  passed. Prepared session name: `trlt-15k-seed0-wandb` on socket `trlt-runs`.
- User subsequently **explicitly approved** uploading metrics, experiment
  configuration and normalization statistics to
  `rnsrns118-seoul-national-university/trlt` and starting the experiment. The
  earlier automatic-review rejection is resolved. Authorization evidence and
  launch Git commit are in `wandb_destination_approval.json` in the run folder.
- Replacement launched at2026-11-14T08:21:45Z, tmux socket `trlt-runs`, session
  `trlt-15k-seed0-wandb`. First stage: `rlt_original/seed0/Stage_A`, starting
  from step0 on GPU0. Inspect actual logs and `wandb_run.json` before reporting
  current progress/upload status. All8 Stage A/B commands enable online W&B.
  Existing sehwa authentication was reused; no shared settings changed.

### Completed single-seed experiment: 2026-11-15

- The active W&B experiment `squeeze_15k_seed0_wandb_20261114T081820Z` completed
  all28 stages at2026-11-15T04:17:42Z. All four Stage A runs reached15000updates;
  Stage B stopped with patience10. All final test evaluations and plots completed.
- Final primary test normalized episode-macro MSE (seed0,26 episodes):
  RLT0.071770421, self-only0.068767073, cross-only0.063485767,
  alternating0.064652117. Cross-only has the lowest observed error (11.54% below
  RLT). Only seed0 was requested; across-seed variability is unmeasured. Paired
  episode-bootstrap95% intervals for all differences versus RLT include zero.
- Detailed evidence: `experiments/trlt/runs/squeeze_15k_seed0_wandb_20261114T081820Z/final_report.md`,
  `summary.json`, `plots/`, `events.log`, and per-model evaluation outputs.
  All selections were sealed before test extraction/evaluation. Equal episode
  IDs and valid-element counts were verified across all final outputs.
- Test visual extraction added1573anchors to the existing shared cache, now
  10230anchors. This supersedes the earlier milestone's test-not-extracted status.
  Do not rebuild completed caches or restart the completed queue automatically.
- Experiment review entry points: root `README.md` summarizes the protocol and
  final results; `docs/trlt/README.md` section "Reviewing the completed experiment"
  maps the actual configs, commands, training/W&B logs, selected checkpoints,
  test predictions, and shared dataset provenance to their files. Artifact
  links point to Git-ignored files on this server, not files shipped in a clone.

### Optional hand-local geometry: 2026-11-15

- Added `geometry_mode=none|mlp|rope3d`; existing four experiment configs remain
  geometry-disabled. Six new examples live in `configs/trlt/geometry/{mlp,rope3d}/`.
  No additional benchmark/GPU job was launched.
- Contract and commands: `docs/trlt/geometry.md`. CPU FK/cache implementation:
  `src/openpi/trlt/geometry.py`, `scripts/prepare_trex_geometry.py`. Model ops:
  `src/openpi/models/trlt_geometry.py`, wired into fusion in `trlt.py`.
- Each hand uses its own Sharpa root frame with wrist proxy[0,0,0]. Sensor points
  approximate the elastomer mesh AABB center on each distal link. The tracked
  spec pins upstream XML/STL hashes and joint mappings; this is not calibrated
  surface/contact geometry. XYZ uses raw historical joint states aligned to
  each selected tactile timestamp, never current state repeated over history.
- MLP adds shared XYZ embedding to tactile tokens. Pair-selective 3D RoPE applies
  only within the same hand to wrist/tactile and tactile/tactile scores; head,
  state, readout and cross-hand edges keep ordinary scores. Wrist patches share
  the wrist proxy and retain patch embeddings. See docs for rotary block/scale.
- Train121/106673frames and validation26/22051frames geometry caches were computed
  on CPU at `datasets/trex_squeeze/geometry_cache/58845ba72575ae5547f17e1f298e610a411793f380846ea8078177569fc0c3cf/`.
  Test geometry is not prepared. Do not automatically rebuild completed caches.
- Evidence: `experiments/trlt/geometry_validation/report.md`, `verification.json`
  and the adjacent pytest logs/XML. Geometry16tests + selected regression59tests
  passed;10optimizer-heavy regressions were deselected after slow CPU compilation.
  Two reduced synthetic CPU pipelines each did StageA2updates, latent export and
  StageB1update with optimizer JIT disabled. Existing four architectures' initial
  parameters/latents exactly matched pre-change source. GPU/full optimizer-JIT
  performance, new benchmark training and final evaluation remain unrun.

### User-authorized geometry experiment launch: 2026-11-15

- User requested training/evaluation with cross-only and alternating after the
  MLP/RoPE implementation. Prepared four combinations: `mlp_trlt_cross_only`,
  `mlp_trlt_alternating`, `rope3d_trlt_cross_only`, `rope3d_trlt_alternating`;
  seed0 only, Stage A15000updates/batch32/micro4, Stage B defaults unchanged.
- Active index: `experiments/trlt/geometry_active_run.json` (also `active_run.json`).
  Run: `experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/`.
  Tmux socket `trlt-runs`, session `trlt-geometry-seed0-v2`, windows train/reports.
  Replacement launched on2026-11-15 on GPU0, initially bounded MLP GPU preflight.
  Inspect actual status/logs before claiming later stages or completion.
- Explicit32-stage queue: two bounded full-shape GPU optimizer preflights using
  train/validation only; four Stage A/export train+validation/Stage B pipelines;
  seal all best checkpoint hashes; prepare test geometry; four test latent
  exports/evaluations; comparison inputs, summary, plots and final report.
  The completed10230-anchor visual cache is reused, not rebuilt.
- Online W&B continues at the previously approved
  `rnsrns118-seoul-national-university/trlt`;8 main Stage A/B runs have distinct
  names/group. No raw datasets/media are uploaded. See `authorization.json`.
- Existing uncommitted implementation is preserved and snapshotted, not reset
  or committed. `plan.json` records exact source/config/input hashes; changes
  stop the next stage. A failing stage stops the queue; no implicit resume/retry.
- Historical RLT/cross-only/alternating no-geometry results are reused only for
  final comparison. New variants receive distinct comparison labels to prevent
  duplicate model/seed aggregation; raw evaluations are preserved.
- Do not duplicate or restart the queue from a launch record. Inspect status,
  events, processes and actual last checkpoint/config/provenance first. Logs and
  final results live under this new run; checkpoints/latents use the same run ID
  under their standard repository roots. Earlier completed results are retained.

- Initial attempt `squeeze_geometry_15k_seed0_20261115T073517Z` was stopped
  during preflight because its launch helper lacked the multiprocessing main
  guard. No optimizer checkpoint/update or main training existed. Logs and
  `superseded.json` are retained; replacement helper is spawn-safe. Active tmux
  session is `trlt-geometry-seed0-v2`. Inspect current evidence before reporting.

- Replacement started at2026-11-15T07:42:06Z but its first GPU preflight initially
  entered WAITING_GPU because another user's process occupied GPU0. No main
  training step had run at that observation. The queue waits automatically;
  consult `launch_observation.json` and live events for later progress.
