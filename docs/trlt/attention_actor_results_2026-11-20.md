# Stage B Attention Actor Experiment Results (2026-11-20)

- **Date:** 2026-11-20
- **Branch:** `attention-actor` (based on `497e08b`)
- **Dataset / Task:** `trex_squeeze` (`58D` bimanual dexterous manipulation, `H=6` action chunk, `11` test objects, `26` held-out test episodes, `35,597` train samples)
- **Frozen Stage A Representation:** `trlt_cross_only` (`squeeze_15k_seed0_wandb_20261114T081820Z`, `seed0` cache size `143 MB`)
- **Seed Evaluated:** `seed 0` (`seeds 1, 2` pending)
- **Environment:** `inst-i108` (`NVIDIA RTX PRO 6000 Blackwell`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, observed GPU memory usage ~`656 MiB`)
- **Runner Script:** `scripts/run_attention_actor_stage_b.sh`

## 1. Experiment Setup & Reproducibility

All five Stage B actor/probe variants were trained on identical frozen Stage A (`trlt_cross_only`) `seed0` `LatentDataset` caches (`[z (512D, stop_gradient), proprioception (58D), visual_age_seconds]`), evaluated on the identical 26-episode `test` split, and summarized with episode-paired bootstrap (`2000` resamples, `seed=42`).

```bash
bash scripts/run_attention_actor_stage_b.sh \
  --python /workspace/users/jiwoo/openpi-RLT/.venv/bin/python \
  --latent-root /workspace/teams/tactile/openpi-RLT/datasets/trex_squeeze/latents/squeeze_15k_seed0_wandb_20261114T081820Z/trlt_cross_only \
  --normalization /workspace/teams/tactile/openpi-RLT/datasets/trex_squeeze/normalization.json \
  --out-root /workspace/users/jiwoo/attention_actor_runs \
  --cache-root /workspace/users/jiwoo/.cache \
  --device 0 \
  --seeds "0"
```

### Variants Compared

| Variant | Config File | Architecture | Mask Mode (`58D`, `12x12`) | Params | LR Schedule | Budget / Patience | Role |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `mlp` | `configs/trlt/attention_actor/probe_mlp.json` | `ActionProbe` (`2x256` GELU MLP) | N/A | 301,660 | `constant 1e-3` | 100 ep / 10 | Legacy Stage B baseline (identical to `configs/trlt/trlt_cross_only.json` probe) |
| `resmlp` | `configs/trlt/attention_actor/probe_resmlp.json` | `ResMLPActionProbe` (`2x336` Pre-LN ResMLP) | N/A | 764,412 | `cosine 5e-4 -> 5e-5` (200w) | 150 ep / 20 | Parameter- & schedule-matched non-attention control |
| `part_attention_kinematic` | `configs/trlt/attention_actor/probe_part_attn_kinematic.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `kinematic` (`34/144`) | 764,380 | `cosine 5e-4 -> 5e-5` (200w) | 150 ep / 20 | Self + parent-child (`arm<->finger`) + bimanual arm bridge |
| `part_attention_hierarchical` | `configs/trlt/attention_actor/probe_part_attn_hierarchical.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `hierarchical` (`64/144`) | 764,380 | `cosine 5e-4 -> 5e-5` (200w) | 150 ep / 20 | `arm->arms` (no finger-to-arm) & `finger->same-side arm+fingers` |
| `part_attention_full` | `configs/trlt/attention_actor/probe_part_attn_full.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `full` (`144/144`) | 764,380 | `cosine 5e-4 -> 5e-5` (200w) | 150 ep / 20 | All-to-all body-part attention (`0.004%` param diff vs `resmlp`) |

---

## 2. Overall Held-Out Test Metrics (`seed 0`, 26 Episodes)

| Model ID | Variant | `ep_macro` (Primary) | `micro_mse` | `obj_macro` | `rmse` | `first_action_mse` | `vs MLP` (`Δ`, 95% CI) | `vs ResMLP` (`Δ`, 95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `trlt_cross_only` | `mlp` | 0.06349 | 0.05890 | 0.07180 | 0.03802 | 0.001320 | `base` | — |
| `trlt_cross_only__resmlp` | `resmlp` | 0.06186 | 0.05735 | 0.06897 | 0.03759 | 0.001280 | `-2.56%` (`-0.00163`, `[-0.00457, +0.00165]`) | `base` |
| `trlt_cross_only__part_attention_kinematic` | `kinematic` | 0.04043 | 0.03665 | 0.04654 | 0.02573 | 0.000534 | **`-36.32%`** (`-0.02306`, `[-0.03082, -0.01671]`) | **`-34.64%`** (`-0.02143`, `[-0.03018, -0.01401]`) |
| `trlt_cross_only__part_attention_hierarchical` | `hierarchical` | 0.03765 | 0.03375 | 0.04116 | 0.02668 | 0.000572 | **`-40.70%`** (`-0.02584`, `[-0.03608, -0.01824]`) | **`-39.14%`** (`-0.02421`, `[-0.03467, -0.01575]`) |
| `trlt_cross_only__part_attention_full` | `full` | **0.03392** | **0.03099** | **0.03925** | **0.02478** | **0.000512** | **`-46.56%`** (`-0.02956`, `[-0.04163, -0.02022]`) | **`-45.16%`** (`-0.02793`, `[-0.04124, -0.01773]`) |

---

## 3. Body-Part Group Breakdown (`group_mse`, Unnormalized)

| Variant | `arms` | `left_arm` | `right_arm` | `hands` | `left_hand` | `right_hand` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp` (`trlt_cross_only`) | 0.003501 | 0.003236 | 0.003767 | 0.000791 | 0.000921 | 0.000662 |
| `resmlp` | 0.003443 | 0.002730 | 0.004155 | 0.000768 | 0.000926 | 0.000609 |
| `part_attention_kinematic` | **0.001481** | **0.001136** | **0.001826** | 0.000402 | 0.000421 | 0.000382 |
| `part_attention_hierarchical` | 0.001916 | 0.001301 | 0.002532 | **0.000328** | 0.000385 | **0.000271** |
| `part_attention_full` | 0.001502 | 0.001153 | 0.001851 | 0.000332 | **0.000372** | 0.000292 |

---

## 4. Per-Object Breakdown (`normalized_mse`)

| Object | `mlp` | `resmlp` | `part_attn_full` | `part_attn_hierarchical` | `part_attn_kinematic` | Best |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `air blower` | 0.04317 | 0.04892 | **0.02264** | 0.02428 | 0.02299 | `full` |
| `ball` | 0.08476 | 0.08452 | **0.04982** | 0.06353 | 0.05974 | `full` |
| `bubble wrap` | 0.07449 | 0.06384 | **0.03962** | 0.04563 | 0.05662 | `full` |
| `clothespin` | 0.05218 | 0.05212 | 0.01852 | **0.01668** | 0.01845 | `hierarchical` |
| `foam earplug` | 0.02269 | 0.01776 | **0.00850** | 0.00908 | 0.01021 | `full` |
| `hair clip` | 0.03374 | 0.02802 | 0.01565 | **0.01438** | 0.02272 | `hierarchical` |
| `hand grip exerciser` | 0.33692 | 0.32761 | 0.21401 | **0.20348** | 0.25051 | `hierarchical` |
| `paper clamp` | 0.07570 | 0.06589 | **0.03437** | 0.04161 | 0.03558 | `full` |
| `sponge` | 0.02796 | 0.02975 | **0.01037** | 0.01435 | 0.01315 | `full` |
| `stress ball aka squeeze ball` | 0.01640 | 0.01690 | **0.00699** | 0.00794 | 0.00894 | `full` |
| `tweezer` | 0.02180 | 0.02338 | **0.01128** | 0.01183 | 0.01303 | `full` |

---

## 5. Key Findings & Mechanistic Interpretation

1. **Pure architectural gain (capacity controlled):**
   Scaling the flat MLP from `301,660` (`mlp`) to `764,412` (`resmlp`) with Pre-LN residual blocks and cosine scheduling yields only `-2.56%` (`95% CI [-0.00457, +0.00165]`, crossing zero / not statistically significant). Keeping parameters fixed at `764,380` while switching to `PartAttentionActionProbe` produces large, statistically significant reductions of **`-34.64%` to `-45.16%`** vs `resmlp` (`-36.32%` to `-46.56%` vs legacy `mlp`).
2. **Immediate-action precision (`first_action_mse`):**
   `part_attention_full` lowers single-step action error from `0.001320` (`mlp`) / `0.001280` (`resmlp`) to `0.000512` (`-61.2%` vs `mlp`, `-60.0%` vs `resmlp`).
3. **Why `full > hierarchical > kinematic` overall — diagnosed cleanly by `arms` vs `hands`:**
   - `kinematic` achieves the best **`arms`** error (`0.001481`) because it permits bidirectional `arm <-> finger` feedback, but lacks direct intra-hand finger-to-finger edges (`hands`: `0.000402`).
   - `hierarchical` achieves the best **`hands`** error (`0.000328`, `right_hand: 0.000271`) and wins all three high-finger-synergy pinch/squeeze objects (`hand grip exerciser`, `clothespin`, `hair clip`) because sibling fingers within each hand can attend to one another; however, blocking `finger -> arm` feedback hurts `arms` (`0.001916`, especially `right_arm: 0.002532`).
   - `full` preserves both `arm <- finger` feedback (`arms: 0.001502`) and intra-hand / cross-hand coordination (`hands: 0.000332`), achieving the best overall MSE and winning `8/11` objects outright (`11/11` over `mlp` and `resmlp`).

---

## 6. Caveats & Next Experiments

- **Multi-seed statistical check:** Current numbers cover `seed 0` only (`missing_expected_seeds: ["1", "2"]`). Exporting `trlt_cross_only` Stage A latents for `seed 1, 2` (or running multi-Stage-A checks) is needed for multi-seed variance estimates.
- **Mask follow-up hypothesis (`hierarchical_bidir` / per-head masking `[H, N, N]`):** The `arms` vs `hands` trade-off directly suggests testing bidirectional `finger -> arm` feedback on top of `hierarchical`, or assigning specialized per-head masks (`kinematic` head + `intra-hand` head + `bimanual` head + `full` head).
- **Stage A cross-backbone generalization:** Evaluate `PartAttentionActionProbe` on cached `seed 0` latents from `rlt_original` (vision-only) and `rope3d_trlt_cross_only` / `trlt_alternating` to verify whether body-part attention preserves or amplifies tactile representation gains.
- **Stage 2 tactile routing:** Integrate `TouchPartAttentionActionProbe` (`PerFingerTouchCrossAttention` with `LingBot-Touch` `HaT-Tiny`) once `HaT-Tiny` checkpoints are converted.
