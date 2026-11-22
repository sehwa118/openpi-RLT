# pi0.5 baseline: local 3DTacDex assembly

This uses `scripts/train.py`, standard pi0.5 flow matching, and no RLT modules.
Branch: `training/pi05-3dtacdex-baseline`, created from local `main`.

## Data and model contract

- Episodes **0–49**, 12,235 frames, 5 Hz; no validation split. This is a pipeline/throughput check, not a success-rate evaluation.
- Two external RGB cameras: `observation.images.main` → `base_0_rgb`, `observation.images.secondary` → `left_wrist_0_rgb`. The second slot name does not imply a wrist-mounted camera. Both are resized with padding to 224×224. No third image is computed.
- State/action: 6 JAKA arm joints + 16 LEAP **source-order** joints, absolute joint targets, float32. No delta transform. Tactile and wrist pose are excluded.
- The first 50 episodes contain 96 invalid action elements. The action validity mask is queried over the same action horizon and excludes these targets from loss and normalization. The extra 10 padded dimensions are also excluded from loss. Other configs retain their original loss when no mask is supplied.
- State is quantile-normalized and discretized into the prompt. Model vectors are padded from 22 to 32 dimensions. Language task comes from `meta/tasks.jsonl`.
- Default action horizon 50 = 10 seconds of dataset time. Episode tails are clamped by LeRobot. This is independent of how many actions a future robot controller executes before replanning.
- Two images produce 2×256 = **512 vision tokens**. Text + state has **200 padded tokens** (first inspected frame: 90 valid tokens). Prefix = **712** padded positions; action expert suffix = **50** positions; training attention spans **762** positions. Each action position represents a continuous 32-vector, not 32 discrete action tokens. Public inference output is **(50,22)**.

The dataset robot is JAKA MiniCobo + LEAP, not OMY. This validates the software pipeline only. Before using OMY demonstrations, define the actual OMY state/action semantics, joint order, units, camera mapping, control rate and limits; recompute normalization on that dataset. Inference here reads a recorded observation and never commands hardware.

## Training video encoding

For the root cause, measured improvement, and validation limits, see the
[data loading incident report (Korean)](data_loading_bottleneck.md).

The baseline configs use `datasets/3dtacdex_assembly/lerobot_default` and default to two data workers.
This is a training copy of all 124 episodes, encoded with the installed
LeRobot `encode_video_frames()` defaults: AV1 (`libsvtav1`), `yuv420p`, GOP 2,
CRF 30. The configs still select episodes 0–49. The original RGB lossless copy
remains at `datasets/3dtacdex_assembly/lerobot_all`.

To reproduce or resume the conversion from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python scripts/reencode_lerobot_videos.py --jobs 8 --cpus-per-job 2
```

The converter uses temporary PNGs per video and publishes the destination only
after every video passes frame-count, timestamp, resolution and GOP validation.
Parquet checksums must match the source; image statistics are recalculated.
Resolution, frame order, timestamps, state/action values and normalization remain
unchanged. RGB pixels change because the standard encoding is lossy. Conversion
provenance is in `meta/video_reencoding.json`; the historical lossless conversion
manifest is retained as `meta/source_assembly_conversion.json`.

PyAV remains explicitly selected because this environment's TorchCodec cannot
load its FFmpeg shared libraries. Short GOP videos fix the excessive decoding
without requiring a decoder installation change.

## Environment

Run from the repository root, using the existing uv environment:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
cd /workspace/users/sehwa/openpi-RLT
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Choose a config: `pi05_tacdex_lora` or `pi05_tacdex` (full fine-tuning).
Both keep pi0.5 discrete state input enabled; both default to EMA disabled.
Both enable W&B logging by default in `rnsrns118-seoul-national-university/trlt`,
matching the existing TRLT experiments. The run name is `--exp-name`; local W&B
files go to `experiments/pi05_tacdex/wandb`. Use `--no-wandb-enabled` to disable
logging, or override `--project-name` / `--wandb-entity` to change the destination.
LoRA uses the upstream freeze filter: transformer backbone weights are frozen,
LoRA and non-transformer parameters (including the vision encoder) are trainable.
It is not an adapters-only vision-frozen mode.

## Compute normalization and inspect real data

```bash
CONFIG=pi05_tacdex_lora
RUN=my_lora_run
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/pi05_tacdex.py stats "$CONFIG" --exp-name "$RUN"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/pi05_tacdex.py inspect "$CONFIG" --exp-name "$RUN" --batch-size 1
```

Statistics are calculated directly from all selected parquet vectors, including the same clamped action windows used for training, without decoding videos or dropping a partial final batch. Inspection exercises the real LeRobot video loader, normalization and tokenizer. Norm stats live under `experiments/pi05_tacdex/assets/<config>/tacdex_50` and are copied into saved checkpoints.

## Train

```bash
STEPS=1000
BATCH=1
mkdir -p "experiments/pi05_tacdex/$RUN"
set -o pipefail
uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name "$RUN" \
  --num-train-steps "$STEPS" --batch-size "$BATCH" \
  --num-workers 2 --log-interval 1 --save-interval 1000 \
  --profile-dir "experiments/pi05_tacdex/$RUN" \
  2>&1 | tee "experiments/pi05_tacdex/$RUN/train.log"
```

For a short GPU dry-run choose `STEPS=3`, `BATCH=1`, and a fresh run name. Checkpoints go to `checkpoints/<config>/<run>/<step>`. The existing trainer numbers directories by zero-based loop index: 3 optimizer updates save directory `2`. To continue use the same run/config and `--resume` with a larger total `--num-train-steps`; `--overwrite` deletes that run's prior checkpoints.

`training_profile.json` records per-step synchronized time, losses, total wall time including initialization/save, and JAX device memory counters (`peak_bytes_in_use`, when available). First-step time includes compilation; later `seconds` values exclude data loading and saving. `initial_data_wait_seconds` records worker startup and the first batch; each step also records `data_wait_seconds` for fetching the next batch. Preallocation is disabled for useful memory measurement. NVIDIA process/device memory can exceed JAX buffer counters. Memory depends on batch, horizon, prompt length, training mode and hardware; measure each intended configuration. Estimate a longer run using initialization + first compile step + remaining steps × steady-state time, plus data/checkpoint overhead. A three-step run is only a rough estimate.

For full fine-tuning set `CONFIG=pi05_tacdex` and repeat stats, inspect and train. Use a separate run name. To view all CLI options:

```bash
uv run --no-sync python scripts/train.py "$CONFIG" --help
```

Useful overrides: `--model.action-horizon 25`, `--model.max-token-len 128`, `--data.num-episodes 50`, `--data.local-root /absolute/path`, `--weight-loader.params-path /absolute/checkpoint/params`, `--lr-schedule.peak-lr 2e-5`. Keep the model's action dimension 32 for base checkpoint compatibility. After changing dataset selection or action horizon, use a distinct `--data.assets.asset-id` and recompute stats with the same options. Training and inference must use the same architecture and data settings. CLI steps/batch/LoRA selection require no code edits.

## Recorded-observation inference after training

Use the same `CONFIG`, `RUN`, `STEPS` and any model/data overrides as training:

```bash
uv run --no-sync python scripts/pi05_tacdex.py infer "$CONFIG" \
  --exp-name "$RUN" --num-train-steps "$STEPS"
```

Loads `checkpoints/<config>/<run>/<STEPS-1>` and its saved normalization assets, runs three predictions on dataset frame 0, checks finite `(horizon,22)` outputs, and writes `inference.json` and `inference_actions.npy` under the experiment directory. The first prediction includes JIT compilation; subsequent times are warm inference.

For a websocket server with the registered default model settings, the standard entry point also accepts these configs (custom model overrides above are supported by the recorded-observation CLI):

```bash
uv run --no-sync python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config "$CONFIG" --policy.dir "checkpoints/$CONFIG/$RUN/$((STEPS-1))"
```

Client observations must use adapter keys `state` (22,), `main`, `secondary` (RGB uint8 HWC or float CHW in [0,1]), and `prompt`. Returned actions are source-order absolute joint targets; this is not an OMY control adapter.
