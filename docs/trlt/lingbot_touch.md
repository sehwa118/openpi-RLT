# T-Rex: frozen LingBot-Touch-Tiny tactile features

This branch implements the T-Rex experiment: **96 points/finger, frozen tiny,
self-only four-stage fusion, seed0, Stage A15,000 updates**, Stage B unchanged.
One tactile token per sensor is passed to fusion (ten tokens total). The earlier
CNN experiment used100 history/sensor tokens, so this comparison also changes
temporal readout and token count. This is an offline reconstruction/action-probe
experiment, not robot-control evaluation.

## Input and temporal contract

The extractor decodes native240×240 deformation **Y-plane codes**, gathers the
pinned upstream96-point weighted native map and divides once by255. No64×64
resize enters the LingBot input. The decoder still reconstructs the original
64×64 ten-frame history used by earlier Stage A runs.

Left and right hands are encoded in separate calls/batch elements with each
hand's five actual fingers in thumb/index/middle/ring/pinky order. Source-native
pixel identity is used on both hands; no unsubstantiated left flip is introduced.
The official T-Rex collection code stores SDK240×240 arrays without image
transforms, but correspondence to LingBot's canonical-right sampling geometry
is **an explicit, unverified experimental assumption**. It is recorded as
`source_native_assumed_unverified` in every cache identity, not described as
calibrated surface geometry.

Each query's past ten-frame window is encoded independently. The upstream
Transformer is noncausal, so full-episode encoding followed by temporal slicing
would leak future information and is never used. Missing leading history is
filled with the earliest available in-window observation; internal gaps use the
last observation. Sensors absent for the entire window have zero coverage and
produce no context tokens. This padding can affect encoder context. Only
original valid observations determine tubelet masks and mean timestamps;
wholly artificial tubelets are excluded from downstream pooling. Zero contact
remains a valid observation.

Tiny emits five temporal tubelets. Each sensor's96 taxel features are averaged
within each tubelet, retaining `[5,10,768]` frozen features per query. A fixed
masked temporal softmax weights them by `exp((tubelet_time-query_time)/0.15)`.
Tubelet times are the means of original valid physical timestamps, in seconds.
All-invalid sensors produce zero features and a false fusion mask. A trainable
768→512 projection, the existing sensor ID and weighted relative-time embedding
prepare ten sensor tokens for fusion. The frozen Torch model is never part of
the JAX parameter tree/optimizer. Stage B also freezes the trained representation.

## Setup and CPU planning

Activate the user-scoped environment before every shell session:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
```

The source clone is pinned at `dec8025b212e7099b235cc2ef13a4921a7e59545` under
`third_party/lingbot-touch/`; the matching model is
`checkpoints/lingbot-touch-tiny/`. Weights, config, sampling-map and reference
geometry SHA256 values are checked. Python upstream sources and the actual
project adapter code also enter cache identity. Existing environments are reused.

On this RTX PRO6000 Blackwell, the shared training environment's Torch2.7.1+cu126
cannot execute Torch CUDA kernels for sm120. Frozen extraction uses a separate
`.venv-lingbot-touch` environment with Torch2.7.1+cu128, prepared using
`scripts/setup_lingbot_extraction_env.sh`. **JAX training still uses the existing
`uv run --no-sync` environment.** The separate environment does not alter shared
Torch/JAX packages or account settings. CPU extraction can use the existing uv
environment with `--device cpu` instead.

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python \
  scripts/cache_lingbot_touch_features.py \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --split train --plan-only
```

The plan enumerates35,597 train queries and about5.48GB of feature payload.
Sampled96-point source observations are cached per episode, avoiding repeated
video decoding per query and avoiding a full-resolution intermediate cache.
Existing visual caches, splits, query sampling and train-only state/action
normalization are reused.

## Explicit extraction, then training

Select an idle allocated GPU; these commands are **separate steps**, never an
automatically launched training queue. Extraction resumes by validating and
reusing committed query chunks. Invalid provenance requires a new cache path;
there is no implicit overwrite/rebuild. `--max-samples` permits bounded checks
but does not create a completed-split marker unless it covers the entire split.

```bash
PYTHONPATH=src .venv-lingbot-touch/bin/python scripts/cache_lingbot_touch_features.py \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --split train --device 0 --batch-size 4
PYTHONPATH=src .venv-lingbot-touch/bin/python scripts/cache_lingbot_touch_features.py \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --split validation --device 0 --batch-size 4

# Stage A: no following stage launches implicitly.
uv run --no-sync python scripts/trex_offline.py train-representation \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --seed 0 --device 0
```

The default representation path is
`checkpoints/trlt/lingbot_touch_tiny96/trlt_self_only/seed0/representation/`.
Logs are under the corresponding `experiments/trlt/` path. Override output/log
paths for isolated runs; never overwrite another experiment. CPU loader workers
read immutable features; they do not run Torch inference or initialize GPUs.

```bash
# Export selected representation for each split explicitly.
uv run --no-sync python scripts/trex_offline.py export-latents \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --seed 0 --device 0 \
  --representation-checkpoint checkpoints/trlt/lingbot_touch_tiny96/trlt_self_only/seed0/representation/best \
  --split train
uv run --no-sync python scripts/trex_offline.py export-latents \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --seed 0 --device 0 \
  --representation-checkpoint checkpoints/trlt/lingbot_touch_tiny96/trlt_self_only/seed0/representation/best \
  --split validation
uv run --no-sync python scripts/trex_offline.py train-probe \
  --config configs/trlt/lingbot_touch/trlt_self_only.json --seed 0 --device 0 \
  --train-latents datasets/trex_squeeze/latents/lingbot_touch_tiny96/trlt_self_only/seed0/train \
  --validation-latents datasets/trex_squeeze/latents/lingbot_touch_tiny96/trlt_self_only/seed0/validation
```

Stage B uses hidden256×2, AdamW1e-3, batch256, max100epochs/patience10.
The commands above use the existing CLI's default disabled W&B; enable using
`--wandb online --wandb-project trlt --wandb-entity rnsrns118-seoul-national-university`
on the individual training commands if desired. No raw tactile data is uploaded.

Test tactile extraction is blocked unless `--allow-test` is supplied explicitly.
After checkpoint selection is finalized, that final-stage operation and test
latent export/evaluation use separate explicit commands. Do not inspect test
features while implementing/debugging or choosing tau/padding/model settings.
The primary metric remains normalized action MSE averaged within each episode
then equally over test episodes. Label this experiment `lingbot_touch_tiny96`
when comparing with older self-only results to avoid duplicate model/seed labels.

## Integrity and actual verification evidence

The cache binds source revision, subset/split/query hashes, normalization,
physical video/Parquet source-file size/mtime inventory, temporal settings, sensor order,
upstream content hashes, sampling/hand assumptions, adapter content and dtype.
The size/mtime inventory detects ordinary local source replacement; it is not a
new content hash of every source video. The sampled96 preprocessing arrays also have SHA256 checks verified on episode open.
Each query chunk has independent array
payload checksums and an embedded source key. Complete split manifests bind the
exact ordered query set. Source identities are checked again before completion.
Frozen tactile identity propagates into Stage A resume, latent export and Stage B
provenance checks. Adding/removing/changing tactile caches is rejected.

Validation artifacts live in `experiments/trlt/lingbot_touch_validation/`.
`host.xml`, `model.xml`, `pipeline.xml` and associated logs distinguish actual
completed tests from planned GPU checks. `sensor_review/independent_trex.json`
records independent sampler equivalence. Parent-run GPU/cache execution reports
are separate artifacts; consult their actual status before claiming completion.
There are no benchmark results for this new encoder in this document.


## Preparation completed: 2026-11-22

Full train/validation tactile caches and actual reader checks are now complete.
See `experiments/trlt/lingbot_touch_validation/ready_to_train.md` and
`ready_to_train.json` for exact commands, source/cache identities and execution evidence.
T-Rex35,597/7,359queries and Assembly14,819/3,472queries are complete;
synchronous and2-worker tensors match exactly for both splits/datasets.
Each full-shape actual-data GPU optimizer preflight passed exactly1update:
T-Rexbatch32/micro4 and Assemblybatch64/micro64. Benchmark Stage A/B and final
source test evaluation have not started. Do not rebuild these completed caches.
