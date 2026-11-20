# Stage B Attention Actor Experiment

This experiment evaluates structured Stage B action probes/actors on top of **frozen Stage A** visuotactile representations (`trlt_cross_only` `LatentDataset` caches), keeping Stage A parameters completely frozen (`jax.lax.stop_gradient(z)`).

## Architectures Compared

| Config | Variant Name | Architecture | Mask Mode | Params (58D, z=512) | Schedule | Budget / Patience |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/trlt/attention_actor/probe_mlp.json` | `mlp` | `ActionProbe` (2x256 GELU MLP) | N/A | 301,660 | constant `1e-3` | 100 epochs / 10 |
| `configs/trlt/attention_actor/probe_resmlp.json` | `resmlp` | `ResMLPActionProbe` (2x336 Pre-LN ResMLP) | N/A | 764,412 | cosine `5e-4 -> 5e-5` (200w) | 150 epochs / 20 |
| `configs/trlt/attention_actor/probe_part_attn_full.json` | `part_attention_full` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `full` | 764,380 | cosine `5e-4 -> 5e-5` (200w) | 150 epochs / 20 |
| `configs/trlt/attention_actor/probe_part_attn_kinematic.json` | `part_attention_kinematic` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `kinematic` | 764,380 | cosine `5e-4 -> 5e-5` (200w) | 150 epochs / 20 |
| `configs/trlt/attention_actor/probe_part_attn_hierarchical.json` | `part_attention_hierarchical` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `hierarchical` | 764,380 | cosine `5e-4 -> 5e-5` (200w) | 150 epochs / 20 |

## Body-Part Tokenization & Attention Masks

### Supported Action Dimensions
- **58D T-Rex (`TREX_PARTS`, 12 tokens)**:
  `left_arm` `[0:7]`, `left_thumb` `[7:12]`, `left_index` `[12:16]`, `left_middle` `[16:20]`, `left_ring` `[20:24]`, `left_pinky` `[24:29]`,
  `right_arm` `[29:36]`, `right_thumb` `[36:41]`, `right_index` `[41:45]`, `right_middle` `[45:49]`, `right_ring` `[49:53]`, `right_pinky` `[53:58]`.
- **22D OMY+LEAP (`OMY_LEAP_PARTS`, 5 tokens)**:
  `arm` `[0:6]`, `index` `[6:10]`, `middle` `[10:14]`, `ring` `[14:18]`, `thumb` `[18:22]`.

### Attention Mask Modes (`mask[i, j] == True`: query `i` attends to key `j`)
1. `full`: All-to-all attention (`144/144` edges on 58D, `25/25` on 22D).
2. `kinematic`: Self + parent-child kinematic tree + bimanual arm-to-arm bridge (`34/144` on 58D, `13/25` on 22D).
3. `hierarchical`: Top-down arm coordination and intra-hand finger synergy (`64/144` on 58D, `21/25` on 22D):
   - Arms attend only to arms (zero finger-to-arm feedback).
   - Fingers attend to their own arm and same-side sibling fingers (zero cross-hand finger leakage).

### Conditioning & Zero-Init Stability
Global conditioning `cond = MLP([z, visual_age_seconds])` modulates every `AdaLNTransformerBlock` (`shift`, `scale`, `gate`) with DiT-style **zero-initialized** modulation (`adaln_modulation`) and `final_adaln`.
Because `cond` excludes proprioception, sparse masks (`kinematic`, `hierarchical`) strictly isolate unlinked body parts.

## Shared PRO 6000 GPU Server Workflow

These instructions follow the shared server rules (`RTX PRO 6000 ×1, 96 GB`, `--device 0`):

### Server Session & Directory Rules
1. Connect with your **personal account** (`ssh gpu` — confirm `whoami` is `jiwoo`, never `root`; `root` is strictly for exceptional admin needs after Slack notice and with `SetEnv LC_WHO=jiwoo`).
2. Keep personal code (`uv sync` / personal `~/.conda/envs`), outputs, and caches under `/workspace/users/jiwoo/...` (`/home` is ephemeral across container recreation; never place files or symlinks directly under `/workspace`, never modify `/workspace/miniforge3` or `/workspace/IsaacLab`, and never use a root-created team `.venv`).
3. Keep shared Stage A datasets, `normalization.json`, Stage A checkpoints, and frozen `LatentDataset` caches once under `/workspace/teams/tactile/...` and read them in place by path (no copying).
4. Start long runs inside `tmux`, check free VRAM first with `nvitop`, and keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` (enforced automatically by `scripts/trex_offline.py` and `scripts/run_attention_actor_stage_b.sh`).
5. **Slack & Security Etiquette**:
   - Share GPU start/end in `#gpu-server` if a run uses `>= 20 GB` VRAM or `>= 1 hour` (or schedule in advance if near-full memory / multi-day), and share if generating `>= 100 GB` of disk artifacts.
   - Never pass tokens/passwords/API keys in CLI arguments or prompts (`/workspace/.team/log/` logs every command); authenticate interactively (`gh auth login`, `hf auth login`, `wandb login`) so secrets stay in your personal home.
   - Never kill processes by name (`pkill python`); inspect PID owner first with `ps -o user,pid,etime,args -p <PID>`.
   - After finishing, verify cleanliness with `ls /workspace`, `nvitop`, `tmux ls`, and `git status --short`.

### Export Frozen Stage A Latents Once per Seed (If Not Already Cached)
Point `--data-root`, `--normalization`, and `--representation-checkpoint` to the existing shared Stage A artifacts under `/workspace/teams/tactile/...`, and write the latent cache once into the shared team location (or your `--out-root` if private):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_CACHE_DIR=/workspace/users/jiwoo/.cache/uv
export HF_HOME=/workspace/users/jiwoo/.cache/hf

for SEED in 0 1 2; do
  for SPLIT in train validation test; do
    PYTHONPATH=src python scripts/trex_offline.py export-latents \
      --config configs/trlt/trlt_cross_only.json \
      --device 0 --seed "$SEED" \
      --data-root /workspace/teams/tactile/<dataset_dir> \
      --normalization /workspace/teams/tactile/<dataset_dir>/normalization.json \
      --representation-checkpoint /workspace/teams/tactile/<stage_a_dir>/seed${SEED}/representation/best \
      --output-dir /workspace/teams/tactile/<latents_dir>/trlt_cross_only/seed${SEED}/${SPLIT} \
      --split "$SPLIT"
  done
done
```

### Run All Stage B Variants with `scripts/run_attention_actor_stage_b.sh`
First verify the exact commands without touching GPU (`--dry-run`):

```bash
bash scripts/run_attention_actor_stage_b.sh \
  --latent-root /workspace/teams/tactile/<latents_dir>/trlt_cross_only \
  --normalization /workspace/teams/tactile/<dataset_dir>/normalization.json \
  --out-root /workspace/users/jiwoo/attention_actor_runs \
  --cache-root /workspace/users/jiwoo/.cache \
  --device 0 \
  --seeds "0 1 2" \
  --dry-run
```

Then run the actual sweep inside `tmux`:

```bash
bash scripts/run_attention_actor_stage_b.sh \
  --latent-root /workspace/teams/tactile/<latents_dir>/trlt_cross_only \
  --normalization /workspace/teams/tactile/<dataset_dir>/normalization.json \
  --out-root /workspace/users/jiwoo/attention_actor_runs \
  --cache-root /workspace/users/jiwoo/.cache \
  --device 0 \
  --seeds "0 1 2"
```

### Equivalent Explicit `scripts/trex_offline.py` Commands (Single Variant / Custom Output)
```bash
# Train a single variant (example: part_attention_kinematic, seed 0) reading shared team latents
PYTHONPATH=src python scripts/trex_offline.py train-probe \
  --config configs/trlt/attention_actor/probe_part_attn_kinematic.json \
  --device 0 --seed 0 \
  --train-latents /workspace/teams/tactile/<latents_dir>/trlt_cross_only/seed0/train \
  --validation-latents /workspace/teams/tactile/<latents_dir>/trlt_cross_only/seed0/validation \
  --output-dir /workspace/users/jiwoo/attention_actor_runs/checkpoints/seed0/probe_part_attention_kinematic \
  --log-dir /workspace/users/jiwoo/attention_actor_runs/logs/seed0/probe_part_attention_kinematic

# Evaluate held-out test split
PYTHONPATH=src python scripts/trex_offline.py evaluate \
  --config configs/trlt/attention_actor/probe_part_attn_kinematic.json \
  --device 0 --seed 0 \
  --latents /workspace/teams/tactile/<latents_dir>/trlt_cross_only/seed0/test \
  --probe-checkpoint /workspace/users/jiwoo/attention_actor_runs/checkpoints/seed0/probe_part_attention_kinematic/best \
  --normalization /workspace/teams/tactile/<dataset_dir>/normalization.json \
  --output-dir /workspace/users/jiwoo/attention_actor_runs/logs/seed0/eval_part_attention_kinematic

# Summarize paired results against legacy MLP (`trlt_cross_only`) or ResMLP (`trlt_cross_only__resmlp`)
PYTHONPATH=src python scripts/trex_offline.py summarize \
  /workspace/users/jiwoo/attention_actor_runs/logs/seed*/eval_*/metrics.json \
  --baseline trlt_cross_only \
  --output /workspace/users/jiwoo/attention_actor_runs/logs/summary_vs_mlp.json

PYTHONPATH=src python scripts/trex_offline.py summarize \
  /workspace/users/jiwoo/attention_actor_runs/logs/seed*/eval_resmlp/metrics.json \
  /workspace/users/jiwoo/attention_actor_runs/logs/seed*/eval_part_attention_*/metrics.json \
  --baseline trlt_cross_only__resmlp \
  --output /workspace/users/jiwoo/attention_actor_runs/logs/summary_vs_resmlp.json
```

