# Frozen LingBot-Touch-Tiny: Assembly self-only 2 layers

This branch implements the user-selected PaXini replacement, not another full
experiment result. Original scratch-MLP configs, parameter trees and serialized
identities remain the defaults. New config:
`configs/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only_2layer.json`.

## Input and model contract

- The frozen upstream encoder is loaded from the pinned local source and tiny
  bundle. Its parameters are absent from the JAX Stage A optimizer.
- **All eight pads are encoded jointly**, once per batch with a shared coverage
  layout: `[B,4,5,30,3]`. Each real thumb/index/middle/ring slot contains tip15
  then pulp15; pinky is false coverage. Different missing-pad patterns are
  batched separately. No pad is invented as another anatomical finger.
- Original `clip(raw/30,-1,1)` is used for **all three axes, including signed Z**.
  The upstream positive-Z normalization helper is bypassed; low-level encode
  preserves the selected values. This differs from upstream's pretrained
  positive-Z signal convention. No absolute value, offset or extra scale is used.
- The original three-row history remains the reconstruction target. Encoder-only
  input duplicates its oldest row: `[t-2,t-2,t-1,t]`. For incomplete histories,
  each present pad fills unavailable slots from this query's earliest/preceding
  valid observation. Missing pads have coverage false. All source frames are
  at or before the query and remain within the same segment/window; no full
  episode is encoded then sliced. Copies do not increment observation counts.
  Noncausal encoder attention can still mix filled context; this policy is
  explicitly part of the cache identity.
- Static geometry is tracked in `configs/trlt/lingbot_touch/paxini_v1.json`.
  Original 15-taxel tip/pulp coordinates, mounts and neutral-pose URDF transforms
  are pinned to official 3DTacDex commit
  `e3177b30be2a60568d1426ce2c588e8c496652f4`. Complete tip+pulp geometry is
  registered per finger using a fixed reference pose and a shared isotropic
  length scale. It is an **approximate static registration**, not calibrated
  spherical/contact geometry or per-frame articulated pad geometry. Source
  arrays, transforms, units, hashes and limitations are retained in the spec.
- Tiny yields `[B,2*120,768]`. Taxel mean pooling is performed **separately for
  each pad and tubelet**, producing `[B,2,8,768]`. All eight pads can exchange
  context inside LingBot. Pooling by pad does not undo that mixing.
- Stage A uses fixed exponential temporal weights, `tau=0.20 seconds`, based on
  the mean timestamps of each tubelet's unique real observations. A tubelet with
  no real observation receives zero readout weight; all-invalid sensors produce
  zero and a false fusion mask. Synthetic copies never count as observations.
- A trainable 768→512 projection plus existing sensor/time embeddings yields
  **8 tactile fusion tokens**. Fusion remains self-only2layer, modality embedding
  disabled, sensor ID enabled. Visual/state inputs and both reconstruction
  decoders/losses retain the existing contract.

## Environments and artifacts

Start every shell:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
cd /workspace/users/sehwa/openpi-RLT/.worktrees/lingbot-tacdex
```

The shared project `.venv` remains unchanged. Its Torch2.7.1+cu126 cannot execute
kernels on this server's Blackwell GPU, so extraction uses the separately prepared
`.venv-lingbot-touch` with Torch2.7.1+cu128. `scripts/setup_lingbot_extraction_env.sh`
reproduces that isolated environment when needed; it does not replace the training
uv environment. Training continues to use `uv run --no-sync` with the existing JAX.
User-facing CLIs bootstrap this checkout's `src` before importing openpi; direct
Python/module/pytest commands should additionally set `PYTHONPATH="$PWD/src"`
because the shared editable installation points at another checkout.

Cache: `datasets/3dtacdex_assembly/lingbot_touch_cache/paxini_joint_v1/{split}/`.
Each immutable numpy shard is SHA256 checked and has query/frame/time identity.
The manifest binds dataset inventory/protocol, fixed normalization, exact upstream
source/config/weights, this adapter's source SHA256, static geometry and padding.
Extraction runtime/version/device/float32 precision is recorded and must match on
resume. Committed shards are reused; incompatible or corrupt outputs are refused.
Features are cached before trainable projection and temporal readout. Training
requires complete train/validation manifests and verifies masks/counts/timestamps.

Existing source split, train-only state/action statistics and visual caches are
reused. No data conversion, resplitting, visual-cache rebuild, test extraction,
benchmark training or final evaluation happens implicitly.

## Explicit commands

Commands below are a workflow, not a script that runs all stages automatically.
Do not rerun completed cache extraction unless the manifest says incomplete.
Choose an actually available GPU before GPU commands.

```bash
CONFIG=configs/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only_2layer.json

# CPU plan: inspect exact input/cache identity without initializing the encoder.
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/cache_tacdex_lingbot.py \
  --config "$CONFIG" --split train --plan-only

# Frozen extraction only. An interrupted command resumes committed shards.
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu .venv-lingbot-touch/bin/python scripts/cache_tacdex_lingbot.py \
  --config "$CONFIG" --split train --device cuda:0
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu .venv-lingbot-touch/bin/python scripts/cache_tacdex_lingbot.py \
  --config "$CONFIG" --split validation --device cuda:0

# Review the training command first; this does not initialize GPU or train.
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/tacdex_offline.py \
  train-representation --config "$CONFIG" --dry-run

# Stage A only: seed0, 8000 total optimizer updates, batch64/micro64, no early stop.
uv run --no-sync python scripts/tacdex_offline.py train-representation \
  --config "$CONFIG" --device 0 --seed 0
```

Default new checkpoint root is
`checkpoints/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only/seed0/`, with
`representation/{best,last}` and later `probe/{best,last}`. New logs are under
`experiments/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only/seed0/`.
They are separate from all historical MLP runs. A config alone does not authorize
resume; inspect the actual `last` optimizer/config/provenance before using it.
W&B is disabled in this new example until explicitly selected via `--wandb online`
and the desired project/entity. Training selection still uses validation only.

After Stage A has completed and `best` exists:

```bash
REP=checkpoints/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only/seed0/representation/best
LATENTS=datasets/3dtacdex_assembly/latents/lingbot_touch_self2/seed0
uv run --no-sync python scripts/tacdex_offline.py export-latents --config "$CONFIG" \
  --device 0 --representation-checkpoint "$REP" --split train --output-dir "$LATENTS/train"
uv run --no-sync python scripts/tacdex_offline.py export-latents --config "$CONFIG" \
  --device 0 --representation-checkpoint "$REP" --split validation --output-dir "$LATENTS/validation"
uv run --no-sync python scripts/tacdex_offline.py train-probe --config "$CONFIG" \
  --device 0 --train-latents "$LATENTS/train" --validation-latents "$LATENTS/validation"

# Seal this one requested variant's Stage A and B validation-selected bests.
PROBE=checkpoints/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only/seed0/probe/best
SEAL=experiments/trlt/3dtacdex_assembly/lingbot_touch/trlt_self_only/seed0/final_selections.json
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/seal_tacdex_lingbot.py \
  --representation "$REP" --probe "$PROBE" --output "$SEAL"
```

Only after selection sealing and a separate final-evaluation decision:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu .venv-lingbot-touch/bin/python scripts/cache_tacdex_lingbot.py \
  --config "$CONFIG" --split test --device cuda:0 --selection-seal "$SEAL"
uv run --no-sync python scripts/tacdex_offline.py export-latents --config "$CONFIG" \
  --device 0 --representation-checkpoint "$REP" --split test --output-dir "$LATENTS/test" --selection-seal "$SEAL"
uv run --no-sync python scripts/tacdex_offline.py evaluate --config "$CONFIG" --device 0 \
  --latents "$LATENTS/test" --probe-checkpoint "$PROBE" \
  --normalization datasets/3dtacdex_assembly/protocol/normalization.json --selection-seal "$SEAL"
```

The new single-variant seal verifies both checkpoint hashes and their linkage to
one representation, the selected LingBot config and the fixed protocol. It does
not relax the historical four-model selection seal. Changing selected checkpoints
invalidates test access. The primary metric remains original-trajectory-macro
normalized action MSE; this does not measure online control success.

## Actual implementation verification

Evidence is under `experiments/trlt/lingbot_touch_validation/` (ignored artifacts):

- `assembly_pytest.xml` / `.log`: signed-Z preservation, joint pad mapping,
  padding/counts, fixed-time pooling/all-invalid safety, actual frozen tiny CPU
  inference, resumable/tamper-checked cache, projection gradients, backward config
  identity and reduced synthetic Stage A1update → frozen export → Stage B1update
  → validation metric → checkpoint seal/tamper rejection.
- `assembly_gpu_preflight/report.json`, `console.log`, `check.py`: actual first64
  train queries and4validation queries, real tiny features extracted on CPU,
  full-shape JAX GPU batch64/micro64 Stage A **one optimizer update passed**.
  This is a bounded implementation check, not an8k benchmark result.
- `sensor_review/`: fixed geometry derivation and independent contract checks.
- `assembly_train_dry_run.json`, `assembly_cache_plan.json`: reviewable CLI plans.

Consult the actual cache manifests and the parent preparation report for current
full train/validation extraction completion. No final test samples were used in
these implementation checks.


## Preparation completed: 2026-11-22

Full train/validation tactile caches and actual reader checks are now complete.
See `experiments/trlt/lingbot_touch_validation/ready_to_train.md` and
`ready_to_train.json` for exact commands, source/cache identities and execution evidence.
T-Rex35,597/7,359queries and Assembly14,819/3,472queries are complete;
synchronous and2-worker tensors match exactly for both splits/datasets.
Each full-shape actual-data GPU optimizer preflight passed exactly1update:
T-Rexbatch32/micro4 and Assemblybatch64/micro64. Benchmark Stage A/B and final
source test evaluation have not started. Do not rebuild these completed caches.
