# 3DTacDex Assembly preparation

This is a separate data-preparation path for a future RLT/TRLT comparison on
single-arm LEAP manipulation. It does not launch representation training,
feature extraction, action probes, test evaluation, or robot control.

## Source and scope

- Hub: <https://huggingface.co/datasets/kingchou007/3dtacdex>
- Revision: `c58157b2acea2b8bf7fea2767e2c00c609b8c579` (release 1.0.1).
- Exact archive: `assembly/assembly.zarr.zip`, 81,101,454,421 bytes.
- SHA256: `fbe888bbce81553f9afd05c50430656c66de992109749bfa3a26124a4d0e1652`.
- Source: JAKA MiniCobo, right 16-joint LEAP, two fixed external RGB cameras,
  eight PaXini pads, 15 three-axis taxels per pad, nominal 5 Hz.
- **Default conversion includes all 124 segments / 22,069 frames**, as requested.
  The 89 segments / 15,625 frames labeled `nouse` are retained. This is a
  historical early-version label, not a verified failure/corruption label.
  `--cohort paper` optionally selects the other 35 segments / 6,444 frames;
  these counts describe archive metadata, not successful demonstrations.

The immutable ZIP is read directly with the already-installed Zarr 3 reader's
Zarr v2 support. No environment recreation or Zarr downgrade is needed. The
downloader fetches only the README and Assembly archive, verifies the full ZIP's
SHA256, and records local identity/size/mtime for subsequent checksum reuse.
Other tasks are not fetched. Original depth and optional fields remain in the
ZIP even when they are not copied into the training-facing conversion.

## LeRobot contract

The output follows **the repository's installed LeRobot v2.1** metadata, Parquet,
and video schema. The installed `LeRobotDataset` is exercised by the validator;
this does not copy T-Rex's LeRobot v3 layout or introduce T-Rex dimensions.

| Feature | Storage | Meaning |
|---|---|---|
| `observation.images.main` | video, RGB 720x1280 | Original main fixed camera |
| `observation.images.secondary` | video, RGB 768x1024 | Original auxiliary fixed camera, not a wrist/ego camera |
| `observation.state` | float64 `[22]` | Six canonical arm joints, then sixteen hand joints, source order/sign |
| `observation.wrist_pose` | float64 `[6]` | Already-corrected FTP-1 wrist pose, JAKA base; meters and rotation-vector radians |
| `observation.tactile` | float64 `[8,15,3]` | Original uncalibrated signed readings; no image conversion, scaling, or Newton claim |
| `observation.tactile_valid_mask` | bool `[8]` | Zero contact is valid; conversion requires finite observations |
| `action` | float64 `[22]` | Raw recorded arm commanded joint positions followed by hand commanded joint positions |
| `action_valid_mask` | bool `[22]` | Finite source command; hand also requires `raw/hand_command_present` |
| `source.timestamp` | float64 scalar | Original absolute/source timestamp |
| `source.elapsed_seconds` | float64 scalar | Original timestamp minus this segment's first timestamp |
| `source.global_frame_index` | int64 scalar | Row in the Assembly Zarr |
| `source.frame_index` | int64 scalar | Frame in the original source trajectory; gaps retained |
| `source.position` | int64 scalar | Position in the source record list, distinct from original frame counters |
| `source.segment_id` | int64 scalar | Original Zarr segment, never a new trajectory identity |
| `source.episode_id` | string | Original trajectory identity, shared across its segments |
| `source.group`, `source.label` | strings | Source version and `nouse`/`task` label |
| `source.instruction` | string | Exact source task instruction, also used by native LeRobot task mapping |
| `source.record_sha256` | string | Original source record digest |

Pad order is thumb tip/pulp, index tip/pulp, middle tip/pulp, ring tip/pulp.
The detailed source attributes and lineage are copied to
`meta/assembly_conversion.json`. Segments remain separate LeRobot episodes,
with their common original trajectory IDs retained for a future grouped split.

Actions are **not copied from observations or synthesized from future states**.
Missing action elements are zero storage padding with a false mask. They must
never enter an action loss, target count, or normalization estimate. The raw ZIP
retains original nonfinite values and presence masks. Raw command units and
sign/order are preserved; the archive does not explicitly document their units
or independent command timing/latency. These need resolving before real-robot
reuse. Canonical joint-state units do not establish raw command units.

LeRobot's container `timestamp` and video PTS are `frame_index / 5`, while both
original timestamp fields are retained exactly. No source observation is
interpolated, duplicated, reordered, or dropped. Future temporal windows must
use actual source elapsed times and segment boundaries rather than assume
the nominal video clock equals the physical collection clock.

RGB is encoded with CPU `libx264rgb`, CRF 0, `ultrafast`, without RGB resizing or
chroma subsampling. Full validation decodes **every RGB frame** to RGB24 and
requires exact pixel equality. Numeric/source columns are also checked exactly
for every row. The installed LeRobot reader's default torch transform can infer
float32 from Python lists; stored Parquet remains float64. Boundary reader checks
compare its documented dtype conversion separately from exact storage checks.

LeRobot's required per-episode descriptive statistics are generated, with
spatially sampled image statistics. They include all converted episodes and
are **not training normalization**. No experiment split or normalization has
been fitted. A later training adapter must compute mask-aware train-only stats.
LeRobot's built-in `train: 0:N` metadata names the whole converted container;
it is not our eventual experimental train split.

## Commands

Start each shell at the repository root with the existing environment:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

uv run --no-sync python scripts/prepare_3dtacdex.py plan
uv run --no-sync python scripts/prepare_3dtacdex.py download
uv run --no-sync python scripts/prepare_3dtacdex.py audit
uv run --no-sync python scripts/prepare_3dtacdex.py convert --cohort all
uv run --no-sync python scripts/prepare_3dtacdex.py validate
```

The CLI itself also disables accelerators. No command starts the next stage.
`--threads 2` means two CPU codec threads per camera. Temporary bitstreams are
atomically promoted after encoding. An existing unrelated or incompatible
output is never deleted. Completed matching conversions are reused. For an
interrupted conversion, `convert --cohort all --resume` continues after committed
episodes with matching provenance; an interrupted metadata commit fails for
inspection. `--output-dir` can select a new destination for recovery. The archive
download uses sixteen bounded HTTP workers, exact Content-Range/length checks,
durable 10 MiB chunk completion records, refreshed Hub-signed URLs, and a lock.
Only committed chunks are reused; full-file SHA256 is required before success.
README/account access uses the existing Hub configuration. Authentication
settings are not changed.

Data defaults to `datasets/3dtacdex_assembly/{source,lerobot_all}/`. Reports and
logs default to `experiments/trlt/3dtacdex_assembly/preparation/`. Paths are
resolved relative to the repository root. All data/results stay Git-ignored.
The CLI places derived Hugging Face Arrow reader caches in
`datasets/3dtacdex_assembly/reader_cache/` without changing `HF_HOME` or auth.

For local inspection after successful conversion, use the installed reader:

```python
import os

from openpi.trlt.config import root_path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_DATASETS_CACHE"] = str(root_path("datasets/3dtacdex_assembly/reader_cache"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/3dtacdex_assembly",
    root=root_path("datasets/3dtacdex_assembly/lerobot_all"),
    video_backend="pyav",
)
sample = dataset[0]
print(len(dataset), dataset.num_episodes)
print(sample["source.group"], sample["source.label"], sample["source.episode_id"])
```

## Validation and next stage

Implementation: `src/openpi/trlt/tacdex_source.py`, `tacdex_lerobot.py` and
`scripts/prepare_3dtacdex.py`. CPU tests: `src/openpi/trlt/tacdex_test.py`.
They cover missing command masks, lineage/frame integrity, nonmonotonic time,
an actual lossless LeRobot roundtrip, output reuse/provenance mismatch,
unrelated-output preservation, and corrupt numeric target detection.

Actual execution evidence belongs in the local
[preparation directory](../../experiments/trlt/3dtacdex_assembly/preparation/).
The implementation's synthetic checks are not validation of downloaded rows.
Consult `source_audit.json`, `lerobot_all_validation.json`, and `report.md` for
actual source/conversion completion; documentation commands are not evidence.

The existing four T-Rex configs remain unchanged. The Assembly adapter, native
taxel encoder/reconstruction, two-camera prefix support, mask-aware 22D probe,
grouped split and train-only normalization are now implemented; see the
[Assembly experiment plan](3dtacdex_experiments.md). GPU feature extraction,
benchmark training and final test evaluation require a separate launch decision.

## Completed preparation: 2026-11-17

The all-source preparation completed at **2026-11-17 16:40:38 UTC**. The full
81.1 GB ZIP SHA256 matched; all 124 segments / 109 original trajectories /
22,069 frames were converted to `datasets/3dtacdex_assembly/lerobot_all/`.
No `nouse` segment or missing-command frame was dropped. Ten frames lack hand
commands and retain false element masks. All converted numeric/source/instruction
rows and all **44,138 RGB frames** matched exactly. The installed reader passed
all 248 episode-boundary samples; eight CPU regression tests passed.

Read the [execution report](../../experiments/trlt/3dtacdex_assembly/preparation/report.md),
[per-segment inventory](../../experiments/trlt/3dtacdex_assembly/preparation/segments.csv),
and [source-group preview](../../experiments/trlt/3dtacdex_assembly/preparation/source_group_preview.png).
The report links the actual commands, source pin, code snapshots, validation
counts and 372 output-file hashes. These are server-local Git-ignored artifacts.
Model training, feature extraction, experimental splitting/normalization and
test action evaluation remain outside this completed preparation scope.
