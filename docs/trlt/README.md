# TRLT and T-Rex squeeze offline action prediction

This independent pipeline compares exactly `rlt_original`, `trlt_self_only`,
`trlt_cross_only`, and `trlt_alternating`. It trains reconstruction representations,
freezes them, then fits supervised action probes. Online actors/critics, rewards,
action refinement, and action-expert inference/training are outside this pipeline.
Validation procedures and local execution-report locations are listed in [validation.md](validation.md).

## Environment and artifacts

Run from the repository root. Start **every new shell** with:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
```

Use the existing uv environment (`uv run --no-sync`); do not reinstall it. All
relative paths resolve against this repository, including when called elsewhere.
Model commands require an explicitly selected GPU and fail on backend failure:

```bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

Choose an actually allocated, idle device on the shared server. Run one GPU job at
a time. Host preparation commands can use `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.

| Location | Contents |
|---|---|
| `datasets/trex_squeeze/` | Pinned source, exact allowlist, splits, statistics, decoded preprocessing maps, visual and latent caches |
| `checkpoints/trlt/` | Representation/probe best and last checkpoint, optimizer/resume state |
| `experiments/trlt/` | Run manifests, losses, metrics, predictions, latency, validation reports |

These artifact directories are ignored by Git. Configuration examples live in
`configs/trlt/`; no experimental artifact is force-added.

## Source and sample contract

The [dataset](https://huggingface.co/datasets/zekaiwang/trex_dataset) is LeRobot v3.
Our adapter directly reads local allowlisted Parquet/MP4 files; it does not invoke
LeRobot's downloader. The pinned revision used during implementation is
`8b478400a9167d59ede0fdbfea616cb5e9791885`.

Only `motor_primitive == "squeeze"` selects episodes; captions are instructions,
never exact-match selection criteria. Actual metadata at that revision contains
173 selected episodes, 152,169 frames and 11 object categories. Source episode IDs
are preserved. Shared files may also contain other primitives; only selected row
and time intervals enter samples or statistics. Never cut arbitrary MP4 bytes.

RGB order/mapping:

1. `observation.images.head_left` → `base_0_rgb`
2. `observation.images.left_wrist` → `left_wrist_0_rgb`
3. `observation.images.right_wrist` → `right_wrist_0_rgb`

Tactile order is left thumb/index/middle/ring/pinky, then right in that order;
source keys are `observation.images.tactile_{side}_deform_{finger}`. Raw tactile
videos are excluded. `observation.tactile_force` is preserved when reading source
Parquet but is never model input, probe input or reconstruction target.

State is actual `observation.state[58]`; target is absolute commanded `action[58]`.
The source joint ordering is left arm `[0:7]`, left hand `[7:29]`, right arm
`[29:36]`, right hand `[36:58]`. Preserve source units and names from
`source/meta/info.json`; the pipeline performs no delta/gripper conversion,
frame shift, or 32-dimensional padding/truncation. The source info schema names
every joint but does not declare a unit field; raw-unit errors retain those
source numeric units without asserting a calibrated physical conversion.
State and action statistics are
computed separately from unique valid **train** frames with std floor `1e-6`.

The adapter copies native 8-bit Y-plane deformation codes, avoiding RGB and
limited/full-range conversion. It verifies 240×240, applies float32 area resize to
64×64, then divides by255. Zero-contact black frames remain valid. Values are not
physical force. Decoded preprocessing maps are immutable, disk-backed and keyed
by source pointers/revision/resize; trainable CNN outputs are never cached.

| Batch field | Shape/type |
|---|---|
| `visual_features`, `visual_valid_mask` | float32 `[B,Nv,Dv]`, bool `[B,Nv]` |
| `tactile`, `tactile_valid_mask` | float32 `[B,L,Ns,H,W,C]`, bool `[B,L,Ns]` |
| `spatial_valid_mask` | Optional bool, broadcastable to tactile |
| `tactile_timestamps` | Canonical float32 `[B,L,Ns]`; model also accepts `[B,L]` |
| `proprioception` | Normalized float32 `[B,58]` |
| `query_timestamp`, `visual_timestamp` | float32 `[B]`, episode-local seconds |
| `episode_id`, `anchor_frame_index`, `query_frame_index` | Integer `[B]` |
| `action_target`, `action_valid_mask` | Normalized float32 `[B,6,58]`, bool `[B,6]` |

True means valid. Object/caption stay host metadata. Geometry is unavailable and
has zero contribution. Missing/padded values are excluded before CNN/attention
and from losses; invalid action tail labels never enter loss or metrics.

At30Hz query frames are0,3,6,…; anchor is `15*floor(query/15)`; tactile contains
`query-9,…,query`; targets are commanded actions `query,…,query+5`. Negative
history and episode-tail targets are invalid. Inputs use nearest previous actual
timestamps with configurable lag (default1.5frames), never future interpolation.
The ten tactile observations span9/30seconds. Visual context changes all three
cameras together at an anchor; actual decoded timestamp is cached. Each window
is an independent fusion call with a fresh learned q, not recurrent latent state.

## Architecture and objectives

| Configuration | Representation input | Encoder | Loss |
|---|---|---|---|
| `rlt_original` | Anchor final VLM image features | Existing RLT encoder,2layers | Visual MSE |
| `trlt_self_only` | Same visual + tactile history + current state/visual age | Joint attention,4stages | Visual + tactile MSE |
| `trlt_cross_only` | Same TRLT inputs | Bidirectional cross attention,4stages | Same TRLT loss |
| `trlt_alternating` | Same TRLT inputs | Intra/cross/intra/cross | Same TRLT loss |

All latent shapes are `[B,1,512]`. TRLT uses8heads, FFN ratio4, dropout0. CNN weights
are shared across sensor/frame: three3×3 stride2 convolutions, channels32/64/128,
per-pixel LayerNorm and GELU, coordinate channels and a flattened spatial Dense
readout. Add camera/patch/modality/sensor/relative-time embeddings and a state MLP.
The CNN and fusion train from scratch. Three TRLT modes have identical parameter
shapes and independent training; original RLT parameter count is different.

Let U contain tactile tokens and one state token. Every stage computes all
updates from the same old hidden states with shared stream projections/FFN.

| Schedule/stage | V queries read | U queries read | q reads |
|---|---|---|---|
| Joint self | V,U | V,U | V,U,q |
| Cross | U | V | V,U,q |
| Intra-modal | V | U | V,U,q |

V/U never read q. All-masked attention updates are zero. Dense masked attention
is used; no sparse FLOPs reduction is claimed.

Every model uses the existing `RLTokenDecoder`, independently trained with2
layers, position-specific learned queries, query self-attention and z
cross-attention, causal=False, output dimension Dv. The decoder receives only z
as content. Its optional `target_valid_mask=None` preserves the original parameter
naming/default path. Masked calls isolate invalid reconstruction queries. RLT
encoder sanitizes invalid padding; the legacy trainer's mask-dropping behavior
is documented and is not copied by this runner. Existing CLI/registry/serving
entry points remain available.

Visual targets are stop-gradient original final-layer PaliGemma image outputs;
visual MSE divides by valid feature elements. Tactile decoder uses sensor/time
queries with self-attention and z cross-attention, reconstructing observed resized
maps. Tactile loss averages pixel error within each valid sensor/frame, then
averages valid sensor/frames; all-invalid loss is0. Defaults lambda_visual=1 and
lambda_tactile=1. Sensor/history/latest errors are logged. Counts and valid fractions
remain in local JSONL for aggregation diagnostics and are excluded from W&B.

Frozen π0.5 receives three images and caption, with deterministic existing resize,
padding and normalization. `extract_prefix_embeddings(image_only=True,train=False)`
returns **final-layer** image positions. Caption tokens condition the prefix but
are not appended to fusion. State-to-text tokenization is disabled. A zero32D
Observation placeholder is unused by prefix-only execution; actual58D state is
kept separately. The checkpoint includes action-expert parameters as part of the
existing model container, but no action-expert forward/gradient/optimizer runs.

Feature cache identity includes revision, source manifest, checkpoint inventory,
preprocessing, camera order, captions/tokenizer and dtype; chunks identify source
episode and anchor. Feature identity also records highest float32 matmul precision; default reduced
precision on Blackwell produced batch-dependent numerical differences during
verification, so offline launchers explicitly select highest precision.
The checkpoint inventory hashes small metadata content and
large-file size/mtime, rather than rereading every multi-GB shard. Latent caches
add checkpoint **content** hashes, split/statistics/temporal provenance, and verify
shard hashes. Mismatches fail; rebuilding requires an explicit flag.

## Explicit execution stages

These commands are examples for later execution. **Full training and final test
were not automatically started during implementation.** Check plan output before
downloading or extracting the full feature cache.

```bash
# Metadata-only fetch, immutable revision, exact files/bytes and disk plan.
uv run --no-sync python scripts/prepare_trex.py plan \
  --dataset-revision 8b478400a9167d59ede0fdbfea616cb5e9791885
uv run --no-sync python scripts/prepare_trex.py split --split-seed 42

# Exact allowlist, resume/reuse existing files; decimal GB of missing source data.
uv run --no-sync python scripts/prepare_trex.py download --max-download-gb 100 --workers 6
# Retry only recorded failures if necessary:
uv run --no-sync python scripts/prepare_trex.py download --retry-failed --max-download-gb 100
uv run --no-sync python scripts/prepare_trex.py validate
uv run --no-sync python scripts/prepare_trex.py normalize
```

Plan reports metadata+48Parquet+476MP4 files (~96.91GB source including metadata; initial new payload~96.85GB). Download space
headroom covers source files only; decoded maps/features require additional space.
Validation opens both stream boundaries and checks pointer durations. A grouped
H264/MP4 packet-PTS scan verifies every selected stream interval's exact frame
count,30Hz cadence and first/last timestamps; each unique file is demuxed once,
using up to8 CPU worker processes (`--packet-workers`). Only selected interval
summaries are persisted, and no model/GPU computation occurs. Missing
files raise; no implicit full download is possible. `--max-episodes` and
`--skip-video-decode` limit validation explicitly. Download failures and completed
file manifests are persisted for retry.

The split algorithm starts with one train/validation/test episode per object,
then assigns residual quotas minimizing squared deviation from per-object70/15/15
while satisfying exact global totals, followed by deterministic seeded shuffle.
Inspect the saved algorithm/object quotas rather than independently resplitting.
A sealed global query manifest under `query_manifests/<hash>.json` stores every
split's source episode/query/anchor/valid-horizon count. The same manifest hash
is verified across configurations, training seeds, latent caches and comparisons.
Train121/106,673frames, validation26/22,051frames, test26/23,445frames at this revision.
Exact duplicate source trajectories crossing splits block validation; collection
session/physical object identity and near-duplicate absence cannot be certified.
This evaluates unseen demonstrations of seen object categories, not object holdout.

```bash
# Size-only cache plan; no backbone initialization.
uv run --no-sync python scripts/cache_trex_features.py --plan-only --split train
# Actual extraction: repeat for validation. Use the same output cache/config/checkpoint.
uv run --no-sync python scripts/cache_trex_features.py --extract --split train --device 0
uv run --no-sync python scripts/cache_trex_features.py --extract --split validation --device 0
```

`--max-anchors`, `--episode-ids`, `--backbone-checkpoint`, `--output-dir`, `--config`,
`--data-root`, `--split-manifest`, `--normalization`, and `--force-cache-rebuild`
are supported. Source revision comes from the pinned local manifest. Features
stream one anchor chunk at a time; the plan estimates768×2048float32 per anchor
and actual extraction verifies shape. Full test feature extraction may be deferred
until checkpoints/hyperparameters are fixed.

```bash
# Inspect a resolved configuration without model initialization.
uv run --no-sync python scripts/trex_offline.py train-representation \
  --config configs/trlt/trlt_alternating.json --dry-run

# Full Stage A example; execute only after reviewing implementation.
uv run --no-sync python scripts/trex_offline.py train-representation \
  --config configs/trlt/trlt_alternating.json --seed 0 --device 0
# --max-steps N overrides the configured step budget (e.g. bounded debug).
# --resume restores the last state; the step budget is a total, not extra steps.

# Stage B: export frozen best representation, separately per split.
uv run --no-sync python scripts/trex_offline.py export-latents \
  --config configs/trlt/trlt_alternating.json --seed 0 --split train --device 0 \
  --representation-checkpoint checkpoints/trlt/trlt_alternating/seed0/representation/best
uv run --no-sync python scripts/trex_offline.py export-latents \
  --config configs/trlt/trlt_alternating.json --seed 0 --split validation --device 0 \
  --representation-checkpoint checkpoints/trlt/trlt_alternating/seed0/representation/best
uv run --no-sync python scripts/trex_offline.py train-probe \
  --config configs/trlt/trlt_alternating.json --seed 0 --device 0 \
  --train-latents datasets/trex_squeeze/latents/trlt_alternating/seed0/train \
  --validation-latents datasets/trex_squeeze/latents/trlt_alternating/seed0/validation
```

Repeat for the four config JSON files and seeds0,1,2 with the identical split,
statistics and anchor feature cache. No command implicitly starts another stage.
Stage A AdamW1e-4, decay1e-4, effective batch32/micro4, global-norm clip1; Stage B
MLP256/GELU/256/GELU/output348, AdamW1e-3, batch256, decay1e-4.
The common optimizer also clips probe gradients to global norm1.0. The four tracked
experiment configs set Stage A to **15,000 optimizer updates**, with
`representation.max_steps=15000`, `max_epochs=null` and `patience=null`: no epoch
limit and no early stopping. Epoch permutations and partial final batches remain
unchanged. Validation and best/last checkpoints are produced every completed epoch
and at the final step, including a partial epoch; validation does not stop training
when patience is null. At effective batch32, each full epoch has1,113updates, so
15,000updates end in epoch14. Stage B retains its initial defaults: at most100epochs,
validation every epoch, patience10. These are experiment settings, not performance
claims. RLT best selection uses validation visual loss; TRLT uses weighted total;
probe uses normalized action MSE. Different reconstruction objectives do not rank
representation quality across models. Accumulation uses valid-element and
sensor/frame counts; final partial batch size is logged. Resume preserves exact
shuffle cursor and RNG protocol, rejecting different config/provenance. Saved
parameters and optimizer buffers restore exactly; independently compiled GPU
training continuations may differ by floating point rounding (one CNN value by
one ULP observed). Continuation is tested with strict numerical tolerance, while
freeze and checkpoint content hashes require exact equality.

The Python `TrainConfig` retains epoch-based defaults for legacy callers; pass the
tracked JSON configs to select this experiment's step-based policy. Optional
positive `max_epochs` and `patience` remain supported. The CLI `--max-steps` overrides
the configured step limit, while any configured epoch/early-stopping limits still
apply. At least a step or epoch limit is required. Older local
`experiments/trlt/configs/*_b32.json` files are previous run configs and are not
updated automatically. Existing processes/checkpoints are not migrated; inspect
config/provenance before resuming. CPU verification evidence for the new policy is
in `experiments/trlt/stage_a_15000_validation/report.md`.

The probe input is exactly `concat(flatten(stop_gradient(z)), normalized state,
visual_age_seconds)`. The optimizer contains probe parameters only. No raw tactile,
force, action labels, IDs, object label or absolute progress enters the probe.
Original RLT z stays fixed within an anchor; its probe still gets current state/age.
TRLT also conditions fusion on state, so RLT/TRLT differences cannot be attributed
purely to tactile. Three TRLT configurations compare the fusion schedules.

## Final evaluation (not executed automatically)

After fixing both stages and hyperparameters, extract test features using
`cache_trex_features.py --extract --split test`, then export frozen test latents:

```bash
uv run --no-sync python scripts/trex_offline.py export-latents \
  --config configs/trlt/trlt_alternating.json --seed 0 --split test --device 0 \
  --representation-checkpoint checkpoints/trlt/trlt_alternating/seed0/representation/best
uv run --no-sync python scripts/trex_offline.py evaluate \
  --config configs/trlt/trlt_alternating.json --seed 0 --device 0 \
  --latents datasets/trex_squeeze/latents/trlt_alternating/seed0/test \
  --probe-checkpoint checkpoints/trlt/trlt_alternating/seed0/probe/best \
  --normalization datasets/trex_squeeze/normalization.json
uv run --no-sync python scripts/trex_offline.py summarize experiments/trlt/*/seed*/evaluation/metrics.json
uv run --no-sync python scripts/report_trlt.py plot experiments/trlt/*/seed*/evaluation/metrics.json
uv run --no-sync python scripts/report_trlt.py rebuild \
  --predictions experiments/trlt/trlt_alternating/seed0/evaluation/predictions.jsonl.gz \
  --normalization datasets/trex_squeeze/normalization.json \
  --output experiments/trlt/rebuilt_metrics.json
```

Primary metric: normalized action MSE averaged within each episode over valid
query×horizon×58elements, then averaged equally across episodes. Also report micro
normalized MSE, raw-unit MSE/RMSE, joint/group/hand/arm/horizon/first-action errors,
object/episode results and macro averages. Overlapping labels at distinct query
and horizon offsets are intentionally counted. Summary retains each seed, sample
std, missing-seed indicators, paired differences/relative changes versus RLT,
and optional episode-level paired bootstrap95%CI. Never bootstrap individual
frames as independent observations or select a favorable test seed. Predictions
contain source episode/object/query/anchor/offset/mask, GT, prediction and per-joint
error; reports/plots can be rebuilt. This measures offline predictability, not
closed-loop success, online RL, or transfer to Paxini hardware.

## Debug and serving integration

```bash
# Small synthetic GPU forward/backward/optimizer/cache/probe smoke.
uv run --no-sync python scripts/trex_offline.py smoke --model trlt_alternating --device 0 --max-steps 2
# Independent unit/integration checks; serialize GPU jobs.
uv run --no-sync pytest -q src/openpi/models/trlt_test.py src/openpi/trlt/runner_test.py src/openpi/trlt/integration_test.py
# Host-only preparation tests.
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync pytest -q src/openpi/trlt/data_test.py

# Real train/validation only: first6queries per chosen episode, at most3steps.
uv run --no-sync python scripts/verify_trlt.py features \
  --train-episodes 110 131 --validation-episodes 167 --device 0
uv run --no-sync python scripts/verify_trlt.py models \
  --train-episodes 110 131 --validation-episodes 167 --device 0 --max-steps 3
```

Smoke outputs use separate `features_smoke`, `latents_smoke`, `checkpoints/trlt/smoke`
and `experiments/trlt/real_smoke` paths. Existing checkpoint paths are rejected
rather than overwritten; choose another output/checkpoint root for repeated runs.
These short debug runs never establish benchmark performance.

`openpi.trlt.cache.CachedContext` owns rolling observations, separately from model
parameters. `reset(episode_id)` clears history/context; `update_visual(...)`
atomically replaces the canonical three-camera context; `append_tactile(...)`
accepts chronological normalized observations. `encode(callback,current_state=...,
query_time=...)` builds a causal tensor batch and returns the callback's
`[B,1,d]` latent. Current state must already use the saved train statistics. A
caller can use `model.apply(variables,batch,method=model.encode,train=False)` as the
callback. No backbone or action expert is called on tactile updates. This is a
local adapter, not a change to the existing online serving protocol.

Latency reports synchronize with `block_until_ready`, separate first compile/call
from steady-state encoding and probe, and distinguish VLM anchor, CNN+fusion,
and fusion-only paths. They include device/backend, shapes, dtype and parameter
count. Fusion-only timing uses synthetic projected inputs with runtime arguments;
it is a component timing, not a complete sensor-to-action latency claim.

## Training progress and W&B

Stage A (`train-representation`) and Stage B (`train-probe`) show a tqdm training
bar by default: optimizer updates/total, rate, ETA, epoch, loss, data preparation
seconds and update seconds. Validation has its own batch bar. Initialization and
JIT messages remain on stderr; stdout remains the final JSON. `--no-progress`
disables bars for redirected logs. Existing `train.jsonl`/`validation.jsonl` stay
available independently of W&B. Changes affect newly launched processes only.

W&B is opt-in (`--wandb disabled`, the default). Add these flags to either
training command when you choose to launch it:

```bash
--wandb online --wandb-project trlt --wandb-entity YOUR_TEAM \
  --wandb-name trlt_alternating-seed0 --wandb-group squeeze
```

Use the credentials already configured for the `sehwa` environment, or supply
`WANDB_API_KEY` securely in that process environment. No shared authentication
configuration is modified. `--wandb offline` records locally for later manual
`wandb sync /absolute/path/to/offline-run-*`; it does not contact the service.
Local W&B files and `wandb_run.json` (run ID/URL) live under the stage's
`experiments/trlt/<model>/seed<seed>/<stage>/` directory. Training configuration,
provenance, device information, per-update losses/metrics/timings and validation
objective/best value are recorded. Sensor/history vectors become indexed scalar
series in source ordering. Charts use `optimizer_step` as the x-axis, including
validation recorded at the same update as training.

After verifying an actual `last` checkpoint, `--resume` restores training state;
with online tracking it also reuses the saved W&B run ID/project/entity. Changing
that saved project/entity is rejected. Offline W&B sessions are separate local
segments (W&B does not support offline run resume). W&B options are separate from
optimizer identity, so tracking can be enabled when resuming an older checkpoint.
Existing JSONL history is not automatically uploaded. Enabling W&B does not alter
checkpoint cadence or start any further pipeline stage.


## CPU data loading and GPU updates

Stage A/B use the existing RLT/pi training pattern: a Torch DataLoader with
spawned CPU workers and bounded batch prefetch, followed by a JIT-compiled JAX
training step. Defaults are `--num-workers 2 --prefetch-factor 2`; use
`--num-workers 0` for synchronous debugging. These runtime settings are logged
in `manifest.json` and do not change optimizer/checkpoint identity.

`num_workers` is the number of CPU loader processes. `prefetch_factor` is the
number of batches dispatched ahead **per worker**. They are independent and do
not have to be equal. For example, 2 workers × factor 2 dispatches up to 4
batches ahead, 4 × 2 dispatches up to 8 with more CPU concurrency, and 2 × 4
also dispatches up to 8 but retains only 2 loading processes. Dispatched work
includes batches being prepared, so this is not a guarantee that all are ready.
More workers can increase loading throughput when CPU/storage capacity permits;
a larger prefetch buffer can absorb temporary variation but cannot fix sustained
slow loading. Both can increase host RAM use. Neither changes effective batch
size, microbatch size or gradient accumulation. With `--num-workers 0`, prefetch
is unused (the CLI still requires a positive factor if supplied). Defaults 2/2
are initial settings, not a measured optimum for this server.

The adapter retains the TRLT fields/masks, fixed epoch permutation, and final
partial batch rather than using the legacy loader's Observation schema and
`drop_last=True`. Workers perform sample reading, optional CPU MP4 preprocessing,
collation and mask counts while the accelerator computes the current update.
They select the CPU backend and cannot access CUDA. Train and validation each
retain their own workers until the stage exits. Queues hold at most
`num_workers * prefetch_factor` dispatched batches per loader. Cold tactile
cache builds are locked per episode/sensor; completed caches are reused.

Gradient calculation, finite checks, accumulation and optimizer application run
inside one JIT call. Only loss/diagnostics and a finite flag are copied to the
host; full gradient trees are no longer copied each step. Invalid gradients do
not enter the optimizer. Accumulation uses a sequential JAX scan (with a separate
smaller tail microbatch), preserving bounded activation memory and effective-batch
mask denominators. Per-step diagnostics/timing still synchronize the update;
CPU prefetch overlaps this interval, but continuous GPU utilization is not promised.

Only successfully consumed batches advance the saved shuffle cursor. Prefetched
samples do not advance resume state, and bounded training does not dispatch
batches past its requested step limit. Existing checkpoint/provenance checks
remain required before choosing `--resume`. These changes apply to new Python
processes only; no running user job is restarted.

CPU validation evidence and exact commands are recorded in
`experiments/trlt/loading_validation/report.md`. GPU performance validation is
not part of this implementation change.


W&B metric groups (new processes):

| Top-level group | Contents |
|---|---|
| `progress/train/*`, `progress/validation/*` | Step, epoch, training examples and validation improvement flag |
| `timing/train/*` | Data preparation and update seconds |
| `stage_a/train/*`, `stage_a/validation/*` | Reconstruction loss, visual/tactile loss and weighted terms, latest/past and per-sensor/per-history errors; validation value/best_value |
| `stage_b/train/*`, `stage_b/validation/*` | Action probe loss; validation value/best_value |

All groups use `optimizer_step` as the x-axis. W&B excludes all aggregation
`*_count` metrics and mask `*_valid_fraction` metrics; local JSONL and loss
aggregation are unchanged. W&B system metrics/config remain available. Previously
uploaded metric names are retained in existing runs; new names do not delete old
charts or history on resume.
