# Hand-local tactile geometry: MLP and wrist/tactile 3D RoPE

This extension adds `geometry_mode=none|mlp|rope3d` to the existing three TRLT
fusion schedules. Original RLT and the four existing example configs keep their
previous behavior. Geometry does not launch another experiment stage.

## Coordinate and time contract

- Each hand has its **own fixed local frame**, the Sharpa MJCF root
  `left_hand_C_MC` / `right_hand_C_MC`. Its origin is our **wrist proxy**;
  it is not the camera optical center or a calibrated anatomical landmark.
- A sensor representative point is fixed to its distal phalanx. It is the
  **axis-aligned bounding-box center of the elastomer mesh**, transformed by the
  MJCF geom pose. This approximates the pad volume center, not a calibrated
  surface center, contact point, or center of mass. Thumb and other fingers use
  their respective meshes. The precise choice is recorded in the geometry spec.
- Raw `observation.state[7:29]` / `[36:58]` map to the upstream 22 hand hinges
  in radians. FK uses the original joint values, not normalized state or action
  targets. Arm angles are unused. Output is in meters.
- `tactile_positions[B,L,10,3]` and `geometry_valid_mask[B,L,10]` follow the
  existing tactile history and left/right thumb-to-pinky sensor order.
- For each **actual selected tactile timestamp**, use the latest preceding state
  within the temporal lag limit. Float32 timestamp tolerance is limited to two
  ULPs (minimum 1e-7 s), as in tactile sampling. Padded history has invalid, zero
  geometry. Do not repeat the current query pose across history.
- Positions from different hands are never treated as physical positions in one
  shared world frame. Across-time local positions express finger articulation;
  they do not encode wrist motion through the world.

The small tracked spec is
[`configs/trlt/geometry/sharpa_wave_v1.json`](../../configs/trlt/geometry/sharpa_wave_v1.json).
It pins upstream repository commit
`ba4047ce3e9f3d2013d60de600e136897e90ccb8`, source XML/STL SHA256 hashes, joint
transforms/order, representative offsets and sensor order. Its self hash is
`58845ba72575ae5547f17e1f298e610a411793f380846ea8078177569fc0c3cf`.
The upstream hand description is Apache-2.0. No simulator, Pinocchio, network
access or new dependency is needed for normal FK/cache preparation or training.

## Model variants

Both variants retain the CNN, sensor ID, modality and relative-time embeddings.
They use a common fixed length scale: `scaled_xyz = xyz_m / 0.01`. This is a
configuration choice, not a train/test-estimated normalization or a tuned result.
Axis- or sensor-specific normalization is not applied.

**MLP:** shared `3 -> 128 -> GELU -> 512`, added once to the tactile tokens
before fusion. No geometry vector is added to visual tokens. Invalid geometry
contributes exactly zero, including MLP biases. `mlp` does not apply RoPE.

**3D RoPE:** rotate Q and K after their projection in every fusion stage. V and
both reconstruction decoders remain unchanged. Each 64D attention head uses
16 dimensions per X/Y/Z axis and leaves 16 dimensions unchanged. Each axis has
8 rotary pairs, with frequency `base ** (-2*i/16)`, `base=10000`.

Wrist image patches all share their hand's wrist proxy `[0,0,0]`. Their existing
camera/patch embeddings remain. These coordinates describe the **observation's
hand association**, not the 3D location of scene content in each image patch.
The wrist proxy is fixed in the chosen local frame, so no camera-time FK is
needed; the existing visual anchor refresh and timestamps remain unchanged.

For each attention edge, select the rotated Q/K dot product **only** when both
tokens have valid geometry and belong to the same hand. Consequently:

| Edge | Dot product |
|---|---|
| Same-hand wrist ↔ tactile | 3D RoPE |
| Same-hand tactile ↔ tactile, including different history times | 3D RoPE |
| Same-wrist patch ↔ patch | Zero-position rotation; ordinary dot product |
| Cross-hand pairs | Ordinary dot product |
| Any pair involving head camera, state or latent readout q | Ordinary dot product |

The fusion schedule's existing attention mask is applied afterwards and remains
unchanged. In cross-only, same-hand wrist ↔ tactile edges carry geometry; no
new tactile ↔ tactile edges are enabled. Invalid geometry falls back to the
ordinary score without discarding an otherwise valid tactile observation.
This is a **pair-selective axial 3D RoPE**, not one global rotary coordinate
system for every token. Implementation computes ordinary and rotated dense
logits and selects per edge. Memory/throughput overhead on GPU is unmeasured.

## Explicit preparation and use

Activate `sehwa` in every shell, as in the main README. The checked-in spec is
sufficient for ordinary preparation:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python \
  scripts/prepare_trex_geometry.py cache --split train
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python \
  scripts/prepare_trex_geometry.py cache --split validation
```

Cache files live at
`datasets/trex_squeeze/geometry_cache/<geometry_hash>/episode_<id>.npz`.
They contain per-frame XYZ and timestamps with dataset/subset/spec provenance
and value hashes. Existing valid files are reused; corrupt or mismatched files
raise. The dataset adapter never implicitly creates a geometry cache or downloads
anything. Optional `--episode-ids` restricts preparation within the named split.
Test preparation is an explicit later command, after selection is fixed; it is
not part of implementation validation.

To regenerate the spec from its pinned small upstream inputs, run the separate
`fetch-model` and `derive` commands. Source XML/STL files are stored under
`datasets/trex_squeeze/geometry_sources/<upstream_commit>/`. `derive` refuses to
replace a different spec; use a new `--spec` for a deliberate protocol change.

Six example configs are provided under:

- `configs/trlt/geometry/mlp/{trlt_self_only,trlt_cross_only,trlt_alternating}.json`
- `configs/trlt/geometry/rope3d/{trlt_self_only,trlt_cross_only,trlt_alternating}.json`

They retain the existing Stage A/B budgets and set `geometry_available=true`.
For example, inspect a training command without initialization or training:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python \
  scripts/trex_offline.py train-representation \
  --config configs/trlt/geometry/rope3d/trlt_cross_only.json --dry-run
```

Default geometry artifact paths are isolated by
`geometry_<mlp|rope3d>/<architecture>/seed<seed>/` under the existing checkpoints,
experiments and latent-cache roots. Geometry spec identity propagates through
training, resume, latent export and probe/evaluation provenance checks. Different
MLP/RoPE hyperparameters live in model config/checkpoint identity; allocate a new
output path for deliberate hyperparameter variants. Existing visual features,
splits and state/action statistics are reused. Existing geometry-disabled model
config serialization is preserved for old checkpoint identities.

The optional `CachedContext.append_tactile(..., positions=..., geometry_valid_mask=...)`
accepts already-computed hand-local sensor points `[10,3]` for that observation.
It selects the points together with the tactile history; it does not infer past
positions from the normalized current state. Geometry-free callers retain their
previous input schema. Online robot deployment remains outside this extension.

## Validation and interpretation

CPU verification entry point: `scripts/verify_trlt_geometry.py`. It checks raw
XML/STL/scipy reference FK, exact synchronous/worker loading of eight actual train
queries, exact parameters/latents against pre-change model source, and bounded
synthetic Stage A -> latent export -> Stage B paths for MLP and RoPE. It never
reads test episodes or uses a GPU. Fresh timestamps isolate its small artifacts.

Execution evidence and exact commands:
[`experiments/trlt/geometry_validation/report.md`](../../experiments/trlt/geometry_validation/report.md).
The geometry sanity plot is a rendering of model-derived proxies, not a physical
calibration validation. No new benchmark training or final test is implied. The existing GPU `smoke` CLI
rejects geometry configs explicitly; use this CPU verification entry point or
the explicit training stages.

Historical XYZ provides information derived from **past joint states**, whereas
the baseline receives only current state explicitly. Improvements would combine
that extra information with the geometry encoding. A matched joint-history
control would be needed to separate these effects. MLP also adds parameters;
RoPE introduces a positional bias rather than a matched learned parameter count.
All experiment choices must be selected on train/validation before final test.

## Authorized experiment queue (2026-11-15)

The user subsequently requested actual training/evaluation for cross-only and
alternating with both geometry candidates, seed0. The explicit32-stage queue was
launched as a fresh replacement under
`experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/`, tmux socket
`trlt-runs`, session `trlt-geometry-seed0-v2`. It starts with bounded GPU optimizer
checks for MLP/RoPE on train/validation, then runs Stage A15000updates and default
Stage B for all four combinations. Online W&B continues in the previously
approved project. Test geometry/evaluation occurs only after selection sealing.

Inspect `status.json`, `events.log`, stage logs and current processes for actual
progress. Launch is not evidence that training or evaluation completed. The
mutable lookup is `experiments/trlt/geometry_active_run.json`. Exact config/source
snapshots, command queue, authorizations and observation commands are in the
run's `README.md` and `plan.json`. The prior completed experiment is preserved;
its no-geometry RLT/cross-only/alternating metrics are reused for comparison.
