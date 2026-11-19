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

### Fusion and geometry results summary: 2026-11-16

- Single-document Korean architecture/protocol/results summary:
  `docs/trlt/architecture_and_results.md`, linked from the TRLT README.
- Verified both runs' actual completed status, config snapshots and final metrics.
  Geometry run `squeeze_geometry_15k_seed0_20261115T074104Z` completed all32stages
  at2026-11-16T05:37:39Z, including all four final evaluations. This supersedes
  earlier launch/waiting and test-geometry-not-prepared observations.
- Geometry MLP cross/alternating primary MSE:0.075159157/0.065073679; RoPE
  cross/alternating:0.067697001/0.073425785. All exceed their same-architecture
  no-geometry seed0 baseline. Only MLP cross has a positive paired episode-
  bootstrap95% interval excluding zero; training-seed variability is unmeasured.
- This documentation update ran no new training, extraction or evaluation.

### Alternating no encoder sensor ID,10k: 2026-11-16

- User requested MLP/3D RoPE with tactile encoder sensor-ID embedding removed,
  alternating only, Stage A10000updates. Added default-true
  `ModelConfig.use_sensor_id_embedding`; false removes encoder `sensor_embed`.
  Decoder `sensor_query`, all other embeddings and RoPE hand grouping remain.
  Default serialization omits the new true field to preserve old identities.
- Two seed0 config examples:
  `configs/trlt/geometry/no_sensor_id/{mlp,rope3d}_trlt_alternating_10k.json`.
  Stage B unchanged. Historical ID-on runs used15000updates, so comparison changes
  both ID and budget. Do not describe it as a matched-budget ID-only ablation.
- CPU evidence: `experiments/trlt/no_sensor_id_validation/report.md`;
  16tests passed, six parameter/latent cases exactly matched old source with
  ID enabled, eight historical saved config identities preserved. The new run's
  cpu_preflight.json records cache checks and four CLI dry runs.
- Authorized20-stage queue launched via tmux socket `trlt-runs`, session
  `trlt-no-sensor-id-10k`. Run:
  `experiments/trlt/runs/squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z/`.
  Index: `experiments/trlt/no_sensor_id_active_run.json` (also active/geometry index).
  It performs two bounded full-shape GPU optimizer checks, both Stage A/B
  pipelines, seals selections, reuses test geometry, evaluates both and writes
  comparison reports. Online W&B continues in the approved project. Inspect
  actual status/logs before claiming updates or completion. No resume/extra seed.
- Existing user documentation changes were preserved. Do not duplicate this
  queue or rebuild completed caches. Source/config/input hashes are verified
  during GPU waiting and immediately before each stage. Changes or failures stop
  the queue. Check actual last checkpoint/provenance before any recovery.

- Both new full-shape GPU preflights actually passed: MLP and RoPE each
  completed1optimizer update +4queryvalidation on GPU0 with batch32/micro4
  and encoder sensor ID disabled. Reports are `*_gpu_preflight.json` inside
  the run, summarized in `experiments/trlt/no_sensor_id_validation/gpu_preflight_summary.json`.
  These checks are not the two10kbenchmark results; inspect current progress.

### Completed no-encoder-sensor-ID experiment: verified 2026-11-17

- Run `squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z` completed all20stages
  at2026-11-16T13:39:08Z, including both final test evaluations and reports.
  Both Stage A runs reached10000updates. Final seed0/26episode primary MSE:
  MLP0.066686607; RoPE3D0.076711648. Existing test target counts agree.
- Historical same-geometry ID-on15k comparisons: MLP+2.48%, RoPE+4.48%MSE;
  both paired episode-bootstrap95% intervals include zero. ID and training
  budget both changed, so this is not an isolated ID-only effect.
- `docs/trlt/architecture_and_results.md` now includes these results, actual
  selected steps and links to the final report/historical_comparisons.json.
  This supersedes earlier launch/preflight-only observations; do not restart
  or duplicate the completed queue. No training/evaluation was rerun for summary.

### 3DTacDex Assembly preparation: 2026-11-17

- User authorized the first preparation scope on new branch `TRLT-3DTacDex`
  (explicitly approved this spelling because existing `TRLT` prevents a
  `TRLT/3DTacDex` ref). Scope: Assembly download, command audit, local LeRobot
  conversion and validation. No Assembly GPU/training/evaluation is authorized.
- **Include all source versions, including `nouse`**, at the user's explicit
  request. Source metadata:124segments /109original trajectories /22069frames;
  `nouse` is a historical early-version label, not a verified failure label.
  Preserve group/label/trajectory/frame/position provenance and missing-command
  masks. Optional `--cohort paper` exists but is not the requested conversion.
- Source pin, tensor/time/action contract and commands:
  `docs/trlt/3dtacdex_assembly.md`. CLI: `scripts/prepare_3dtacdex.py`;
  implementation: `src/openpi/trlt/tacdex_{source,download,lerobot}.py`.
  Data: `datasets/3dtacdex_assembly/{source,lerobot_all}/`. Existing T-Rex
  loaders, model configs, caches and completed experiments are unchanged.
- Execution evidence/status:
  `experiments/trlt/3dtacdex_assembly/preparation/pipeline_status.json`, stage
  logs and reports in that directory. Inspect these and live processes before
  duplicating/restarting preparation. Historical launch messages are not proof
  of completion. All four preparation stages completed on2026-11-17; source
  SHA256 matched, all22069numeric/source/instructionrows and44138RGBframes
  matched exactly, and248actual LeRobot boundary samples passed. Eight CPU
  tests passed. See `report.md`, `source_audit.json`, `lerobot_all_validation.json`
  and `summary.json`; no benchmark training/evaluation was performed.
- Converted dataset: `datasets/3dtacdex_assembly/lerobot_all/` (124episodes,
  109source trajectories). All22069frames are retained;10missing hand-command
  frames have false element masks, not synthesized labels. `segments.csv` and
  `source_group_preview.png` in the report directory support manual review.
  The CLI keeps derived Arrow caches in `datasets/3dtacdex_assembly/reader_cache/`
  without changing account/auth paths. Do not rebuild the completed data blindly.
- Initial SDK download lost its temporary partial file on interruption;
  `download_transport_migration.json` records this. The replacement downloader
  uses durable10MiB completion records,16HTTPworkers and refreshed signed URLs.
  Reuse its `.ranges.part`/`.ranges.json`; do not delete them or revert to SDK
  transfer over that file. Full SHA256 remains required before success.
- Conversion uses installed LeRobot v2.1 with lossless CPU H.264 RGB, native
  float64 tactile/state/command storage and explicit command masks. Original
  source timestamps remain separate from nominal5Hz video/container PTS.
  Raw command units/independent command timing are not asserted from joint-state
  units. No experiment split or train normalization is fitted in this scope.

### 3DTac-Dex four-model experiment implementation: 2026-11-17

- User requested the experiment plan and additional code, not a training launch.
  Plan/contract: `docs/trlt/3dtacdex_experiments.md`. Four seed0 configs are in
  `configs/trlt/3dtacdex_assembly/`; Stage A15000updates/batch32/micro4, Stage B
  defaults unchanged. No Assembly GPU/backbone extraction/benchmark/final test
  has been authorized or run in this implementation session.
- Native `[8,15,3]` tactile uses a shared ordered-taxel MLP and native reconstruction;
  state/action22D, two external cameras/512expected prefix tokens. Default image
  model serialization preserves historical T-Rex identities. Geometry is disabled.
- CPU protocol prepared under `datasets/3dtacdex_assembly/protocol/`: all124segments
  including nouse; source-trajectory grouped and group/label-stratified seed42.
  Train/validation/test:77/16/16original trajectories,89/17/18segments,
  14819/3472/3778frames. Reuse saved split and masked train-only state/action/taxel
  normalization; never use LeRobot whole-container statistics for training.
- Implementation: `src/openpi/trlt/tacdex_data.py`, `tacdex_experiment.py`, native
  encoder in `models/trlt.py`; shared runner/metrics now accept element action masks.
  Preparation/verification: `scripts/{prepare_tacdex_experiment,verify_tacdex}.py`.
  `scripts/tacdex_offline.py` delegates explicit stages to the shared runner.
- `scripts/plan_tacdex_experiment.py plan` writes config snapshots and34ordered
  command argv arrays, never launches a queue. `seal` fixes all four Stage A/B
  selected checkpoint hashes before Assembly test CLI access; changed selections
  are rejected. Final macro/paired-bootstrap unit is the original source trajectory,
  not a LeRobot segment. Source lineage remains in latent/prediction outputs.
- Actual CPU checks/evidence, commands and unrun work:
  `experiments/trlt/3dtacdex_assembly/implementation/report.md`. GPU/full-backbone
  two-camera extraction, full-size optimizer checks, training and final test remain
  separate execution steps. No existing T-Rex caches/results or active-run indices
  were rebuilt, resumed, or modified.

### Assembly modality/budget revision: 2026-11-18

- User requested removing fusion modality embeddings and changing Stage A to
  8000optimizer updates with effective batch64. All four Assembly configs now
  set `use_modality_embedding=false`, batch64/micro4/accumulation16; epoch cap and
  early stopping remain disabled. Camera/patch/sensor/time embeddings and decoder
  queries remain. Original RLT has no TRLT modality embedding to remove.
- Default-true `ModelConfig.use_modality_embedding` preserves old T-Rex and earlier
  config identities by omitting the true field from model serialization. False
  removes the encoder parameter, rather than retaining a trainable zero tensor.
- User explicitly chose to KEEP existing train/validation/test with validation
  checkpoint selection. Reuse existing saved split/normalization. Both Stage A/B
  use train for optimizer updates, validation for selection, and test only for
  final evaluation after sealing. Stage B defaults remain unchanged.
- Current prepared plan (not launched):
  `experiments/trlt/3dtacdex_assembly/plans/assembly_8k_b64_no_modality_seed0_20261118_v2_prepared/plan.json`.
  Earlier15k/batch32 plans are historical and superseded for future execution.
- Contract: `docs/trlt/3dtacdex_experiments.md`. Actual CPU revision evidence:
  `experiments/trlt/3dtacdex_assembly/revisions/20261118_no_modality_8k_b64/report.md`.
  No GPU job, extraction, benchmark training or final test was launched.

### Assembly full microbatch and W&B verification: 2026-11-18

- User corrected Stage A to effective batch64 / microbatch64 / accumulation1 and
  requested native Assembly W&B settings plus train logging every10optimizer steps.
  All four Assembly configs retain8000updates and no modality embedding. Existing
  split/train normalization and Stage B optimizer policy remain unchanged.
- Config `tracking` enables the approved project
  `rnsrns118-seoul-national-university/trlt`, log_every_steps10. Main Stage A/B names
  are `assembly/<model>/seed0/stage_a|stage_b`, grouped by run ID; future preflights
  explicitly disable W&B. CLI `--wandb-log-every` overrides config; old T-Rex
  defaults remain disabled/every1. Validation events and every-step local JSONL
  are retained. Tracking/data summaries are runtime metadata outside identity.
- User explicitly authorized GPU validation in this turn. GPU0 was idle before
  the sequential checks. Actual two-camera prefix extraction passed24train/val
  anchors, shape512x2048. Four full-size models each passed1optimizer update at
  batch64/micro64 plus4validation queries. Frozen exports and an alternating
  probe passed12bounded updates. No final-test samples or benchmark training ran.
- Actual GPU checks used offline W&B. SDK files confirm native2cameras/8pads,
  state/action22D, history3, modality=false and64/64/1. The actual12-update probe
  has train logging only at10 and all validation events.28latest CPU test cases
  pass (includes corrected SDK initial-config record reader), plus28CLI dry runs.
- A single explicitly labeled online logging-verification run replayed two actual
  probe events and minimal configuration; remote API confirmed step10/metadata:
  `https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/9c6s4a63`.
  No raw media, normalization arrays or model artifacts were uploaded in this
  online check. Initial30s timeout was recovered on the SAME run ID; auth unchanged.
- Current plan, NOT launched:
  `experiments/trlt/3dtacdex_assembly/plans/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/plan.json`.
  Older micro4 plans are superseded. Evidence/commands/configs/logs:
  `experiments/trlt/3dtacdex_assembly/revisions/20261118_full_b64_wandb10/report.md`.
  Bounded validation did not authorize an8000-step launch or final test evaluation.

### User-authorized Assembly experiment launch: 2026-11-18

- User reviewed the final plan and explicitly requested starting the experiment.
  Approved scope: all34stages,4models,seed0; Stage A8000updates/batch64/micro64/
  accumulation1, no modality embedding; Stage B defaults; online W&B every10
  train steps; validation-based selection and sealed final test evaluation.
- Launched at2026-11-18T01:23:03Z, GPU0, tmux socket `trlt-runs`, session
  `trlt-assembly-8k-seed0`, windows `train` and `reports`. Run ID:
  `assembly_8k_fullb64_wandb10_seed0_20261118_prepared`.
  Run root: `experiments/trlt/3dtacdex_assembly/runs/<run_id>/`.
  Active indices: `experiments/trlt/assembly_active_run.json`, dataset-local
  `active_run.json`, and global `experiments/trlt/active_run.json`; prior global
  index is preserved in the run folder. Earlier completed experiment files stay.
- Initial actual observation: protocol reuse completed (stage1/34); stage2
  `cache_train` started with a live CUDA Python process. This is a launch-time
  observation, not a claim of any later optimizer update or completion.
  Inspect actual `status.json`, `progress.json`, `events.jsonl`/`events.log`,
  `current_process.json`, stage logs, W&B identities and live processes first.
- Run files include `approved_plan.json`, config/source snapshots (113files),
  `launch_manifest.json`, exact command argv, authorization and source/input hashes.
  Stage order: shared train/val features;4preflights;4Stage A/export train+val/
  Stage B pipelines;seal;shared test features;4test exports/evaluations;summary.
  Fresh model initialization; no resume. Never duplicate this queue.
- Parent/observer remain CPU-only. Each GPU command checks GPU0 is idle first.
  Source/config/protocol/converted-input/backbone changes or any failing stage
  stop the next launch; no automatic retry/resume. Per-run locks prevent duplicate
  launches. Other users' jobs are never stopped. A source edit needs actual
  checkpoint/provenance inspection before recovery, not a blind restart.
- Checkpoints: `checkpoints/trlt/3dtacdex_assembly/<run_id>/`; latents:
  `datasets/3dtacdex_assembly/latents/<run_id>/`. W&B:
  `rnsrns118-seoul-national-university/trlt`,8main runs grouped by run ID.
  Training W&B starts at main Stage A; feature/preflight stages do not create
  main training runs. Final outputs include `final_report.md`, `summary.json`,
  `summary.csv`, `comparison_validation.json` and sealed selections.

### Optimized Assembly feature extraction restart: 2026-11-18

- User explicitly requested stopping current extraction/queue, deleting the freshly
  produced shared cache and optimizing/re-extracting. Old queue
  `assembly_8k_fullb64_wandb10_seed0_20261118_prepared` stopped at stage2/cache_train.
  Deleted78anchors/327361813bytes from `datasets/3dtacdex_assembly/visual_cache/`.
  Old logs/status/snapshots and all source/protocol/normalization remain preserved.
- Added `src/openpi/trlt/feature_loading.py` (spawned CPU iterable DataLoader),
  `AssemblyDataset.iter_anchors` (each episode/camera opened once, sequential exact
  native rows), batched prefix API in features.py and explicit CLI feature knobs.
- Four configs use feature batch32, prefix_microbatch_size1, workers2/prefetch2.
  Image encoding/transfer is batched; Gemma uses JIT-mapped batch1 arithmetic.
  Fully batched Gemma differed from batch1 by2.85% relative L2 and was rejected.
  Hybrid measured relative L2=9.39e-6/cosine>=0.9999999996 over32train anchors;
  sixteen saved batch1 references each satisfyrelative L2<1e-4. RGB/masks/times
  remain exact. Cache identity records extraction algorithm/fixed batch/prefix;
  do not mix legacy and optimized chunks. Legacy default1/0worker path is kept.
- Actual CPU11tests passed; real64anchor worker2/GPU/cache/tail path passed.
  Train-only decoder benchmark11.47s→2.57s for32anchors; GPU prepared-input
  throughput5.21→7.79anchors/s. Full-stage throughput must come from actual logs.
  Evidence/commands: `experiments/trlt/3dtacdex_assembly/feature_optimization_20261118/`.
- New run `assembly_8k_fullb64_batched_seed0_20261118`, source snapshot115files,
  launched2026-11-18T01:53:50Z. Tmux socket`trlt-runs`, session
  `trlt-assembly-batched-seed0`, train/reports windows. Active indices point here.
  Earlier run is STOPPED_BY_USER/superseded; do not restart or duplicate it.
- New queue begins from empty shared cache. Existing authorized four-model
  Stage A8000/batch64/micro64, Stage B, W&B10, fixed split and final test protocol
  remain unchanged. Inspect new status/progress/events and actual extraction
  reports before claiming train/validation features or optimizer runs completed.


- Optimized production train/validation extraction actually completed, verified
  2026-11-18 02:10 UTC: 4,973 + 1,162 = 6,135 fresh anchors; exact saved-protocol
  key sets, no test keys, matching metadata identity and eight distributed
  shape/finite/mask samples passed. Cache hash:
  `abb20dc6ecb8e7979aea11e8d6330837c9e5fc624aff3cc09b34a005765b4c1e`.
  Full-stage extraction/write rates 6.754 / 6.559 anchors/s (backbone load separate).
  Evidence: `feature_optimization_20261118/production_cache_validation.json`
  and run `feature_extraction_{train,validation}.json`, under the Assembly paths
  above. Queue advanced to stage 4/34 `rlt_original_gpu_preflight`; inspect live
  status before reporting any later training/evaluation. Do not rebuild the
  completed shared cache or duplicate/restart the queue.


### Completed Assembly four-model experiment: verified 2026-11-19 UTC

- Run `assembly_8k_fullb64_batched_seed0_20261118` completed all 34 stages
  at 2026-11-18T16:11:22Z, including all four final test evaluations. All Stage A
  last checkpoints contain 8000 updates; selected A steps are 8000/7888/7656/8000.
  Final seed0 normalized source-trajectory macro MSE: RLT 0.059761459, self
  0.092136257, cross 0.090131010, alternating 0.085491817. TRLT errors exceed
  RLT by 54.17% / 50.82% / 43.06%; all paired trajectory-bootstrap 95% delta
  intervals are positive. Only seed0 was requested; training-seed variation
  remains unmeasured. This is offline prediction, not online control success.
- Same 16 original test trajectories and per-trajectory valid counts; 248112
  valid elements/model. Primary means, common protocol/cache provenance and
  test-after-selection timing were verified. Evidence in the run directory:
  `final_report.md`, `summary.json`, `summary.csv`, `comparison_validation.json`,
  `final_selections.json`, `results_review.json`, per-model evaluation files.
  Korean results and limitations: `docs/trlt/3dtacdex_experiments.md`.
- This supersedes earlier preflight/training/test-not-extracted observations.
  Completed test extraction added 1267 anchors to the shared cache (7402 total).
  Do not rebuild the completed cache or duplicate/restart the queue. No training,
  extraction or evaluation was rerun to summarize these results.


### User-authorized Assembly self-only 2-layer follow-up: 2026-11-19

- User requested another self-attention-only experiment with 2 fusion layers.
  Tracked config: `configs/trlt/3dtacdex_assembly/depth/trlt_self_only_2layer.json`;
  exact previous executed config except `num_stages:4→2`. Seed0, tactile history3,
  no modality embedding, Stage A8000/batch64/micro64/accumulation1; Stage B and
  W&B10 unchanged. Existing full visual cache/splits/normalization reused.
- New run `assembly_self2_8k_fullb64_seed0_20261119T002037Z` launched at
  2026-11-19T00:22:58Z on GPU0, tmux socket `trlt-runs`, session
  `trlt-assembly-self2-seed0`, train/reports windows. Active Assembly/global
  indices and `depth_active_run.json` point here. Nine explicit stages: GPU
  preflight, Stage A, train/validation exports, Stage B, selection seal, test
  export/evaluation, comparison report. No implicit resume/retry; source/config/
  input/cache changes or a busy GPU stop the next stage. Inspect actual status,
  logs and saved state before claiming progress or recovery; never duplicate.
- Seal CLI supports explicit base-selection reuse/replacement; unchanged models
  retain their old verified records, self-only gets new validation selections.
  Default four-model test gating is preserved. CPU2tests +7CLI dry runs, baseline
  hashes and isolated historical-metric report replay passed. Evidence under
  `experiments/trlt/3dtacdex_assembly/depth_2layer_validation/`.
- Prior test results were already inspected before this depth follow-up was
  requested; report it as exploratory. New comparisons label self4/self2
  separately; historical RLT/self4 results are reused, not retrained. Existing
  completed runs are retained and should not be restarted or overwritten.

- Actual 2-layer full-shape GPU preflight passed: one optimizer update plus
  all 3472 validation queries at64/64/1 on GPU0. Main Stage A started fresh
  at2026-11-19T00:24:55Z; online W&B run `tjyjid6l`. Evidence:
  run `gpu_preflight_observation.json` and `depth_2layer_validation/summary.json`.
  This is startup/preflight evidence, not 8000-step or final-test completion.


### Completed Assembly self-only 2-layer experiment: verified 2026-11-19

- Run `assembly_self2_8k_fullb64_seed0_20261119T002037Z` completed all 9 stages
  at 2026-11-19T03:26:42Z, including new final test evaluation. Stage A actually
  reached8000updates; selected A step7888 / B epoch12 (last B epoch22).
  Test primary MSE0.081193840: 11.88% below same-setting self4 0.092136257,
  but35.86% above RLT0.059761459. This is lowest observed TRLT primary error
  among current Assembly runs, seed0 only. Paired original-trajectory bootstrap
  95% CI for self2−self4: [-0.016479,-0.005356]; no seed variability measured.
- Actual configs differ only in fusion depth4→2; all five evaluated settings
  share protocol/features and16original test trajectory IDs/counts (248112valid
  action elements/model). New validation checkpoint selections were sealed
  before new test export/evaluation. Existing baseline results were reused.
  Evidence in run: `final_report.md`, `summary.json`, `results_review.json`,
  `final_selections.json`, raw evaluation files and events. Korean summary:
  `docs/trlt/3dtacdex_experiments.md`.
- Supersedes the preflight/startup-only observation. Do not restart/duplicate
  the completed queue or rebuild features. This depth follow-up was requested
  after earlier test results were inspected; preserve exploratory provenance.
  No training/extraction/evaluation was rerun for the results review.


### User-authorized Assembly self2 modality-on follow-up: 2026-11-19

- User requested restoring modality embedding with self attention2layers.
  New config `configs/trlt/3dtacdex_assembly/modality/trlt_self_only_2layer_modality.json`
  differs from the actual previous self2 config only in
  `use_modality_embedding:false→true`. Seed0, tactile3, Stage A8000/batch64/
  micro64/accumulation1, Stage B defaults and W&B10 remain unchanged. Existing
  splits/train-only normalization/full7402-anchor visual cache are reused.
- Run `assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z` launched
  at 2026-11-19T03:50:26.710345+00:00 on GPU0. Tmux socket `trlt-runs`, session
  `trlt-assembly-self2-modality-seed0`, train/reports windows. Global/Assembly
  active indices and `modality_active_run.json` point here; `depth_active_run.json`
  still identifies the completed previous modality-off2 experiment. Nine stages:
  GPU preflight, Stage A, train/validation exports, Stage B, seal, test export/
  evaluation, report. Inspect actual status/logs/checkpoints; never duplicate or
  infer completion from launch. Source/input/config/cache changes, occupied GPU
  or stage errors stop the queue; no implicit retry/resume.
- CPU serialization regression1test,7CLI dry runs, historical checkpoint hashes
  and isolated report replay passed. Evidence under
  `experiments/trlt/3dtacdex_assembly/modality_2layer_validation/`. Enabled true
  can be omitted by legacy model-config serialization; check explicit run config
  and actual `encoder/modality_embed` shape[3,512], not field absence alone.
- Compare distinct on/off2 labels and historical RLT/self4 without retraining
  old results. New selections are sealed before new test evaluation. This is
  a follow-up after prior test inspection, seed0 only; preserve exploratory
  provenance and do not claim online control success.

- Actual modality-on full-shape GPU preflight passed:1optimizer update and
  all3472validation queries at64/64/1 on GPU0. Checkpoint contains finite
  `encoder/modality_embed` shape[3,512]. Main Stage A started fresh at
  2026-11-19T03:52:34.818280+00:00. Evidence: run
  `gpu_preflight_observation.json` and `modality_2layer_validation/summary.json`.
  These are preflight/startup observations; inspect live state for later updates
  or completion, not this record.


### Completed Assembly self2 modality-on experiment: verified 2026-11-19

- Run `assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z` completed all9stages
  at2026-11-19T07:06:42Z, including final evaluation. Stage A actually8000updates;
  selected A7192/B epoch13 (last B23). Primary test MSE0.080929551 vs same2layer
  modality-off0.081193840: -0.33% error. Paired on−off trajectory-bootstrap95%
  CI[-0.005464,+0.005207] includes zero; improvement remains unestablished.
  On remains35.42% above RLT0.059761459. Only seed0; no seed variability measured.
- Actual selected best has finite `encoder/modality_embed` [3,512]. Only model
  setting change is false→true; restore ModelConfig defaults when comparing
  serialization (enabled true can be omitted). Same16original test trajectory
  IDs/counts and248112valid elements/model, common normalization/features;
  primary means and selection-before-test timing verified. Evidence in run:
  `final_report.md`, `summary.json`, `results_review.json`, raw evaluation and
  `final_selections.json`. Korean summary: `docs/trlt/3dtacdex_experiments.md`.
- Supersedes startup/preflight-only observations. No retraining/extraction/
  evaluation was performed to review results. Preserve exploratory designation
  (follow-up after previous test review); never restart/duplicate the completed
  queue or rebuild the feature cache automatically.
