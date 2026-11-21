# Stage B 어텐션 액터(Attention Actor) 실험 결과 (2026-11-20)

- **실험 일자:** 2026-11-20
- **브랜치:** `attention-actor` (기준 커밋 `497e08b`)
- **데이터셋 / 태스크:** `trex_squeeze` (`58D` 양손·다지 정밀 조작, `H=6` 액션 청크, 테스트 `11`개 물체 / `26`개 에피소드, 학습 `35,597` 샘플)
- **동결 Stage A 표현:** `trlt_cross_only` (`squeeze_15k_seed0_wandb_20261114T081820Z`, `seed0` 캐시 크기 `143 MB`)
- **평가 시드:** `seed 0` (`seed 1, 2` 추후 진행 예정)
- **실행 환경:** `inst-i108` (`NVIDIA RTX PRO 6000 Blackwell`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, 실측 GPU VRAM 약 `656 MiB`)
- **실행 스크립트:** `scripts/run_attention_actor_stage_b.sh`

## 1. 실험 설정 및 재현 방법

Stage B 액터·프로브 5종을 모두 동일한 동결 Stage A(`trlt_cross_only`) `seed0` `LatentDataset` 캐시(`[z (512D, stop_gradient), proprioception (58D), visual_age_seconds]`) 위에서 학습한 뒤, 동일한 26개 `test` 에피소드에서 평가하고 에피소드 쌍대 부트스트랩(paired episode bootstrap, `2000` resamples, `seed=42`)으로 비교했습니다.

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

### 비교 모델 5종

| 변형 (`variant`) | 설정 파일 | 아키텍처 | 마스크 (`58D`, `12x12`) | 파라미터 수 | LR 스케줄 | Epoch (Patience) | 비교 역할 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `mlp` | `configs/trlt/attention_actor/probe_mlp.json` | `ActionProbe` (`2x256` GELU MLP) | 해당 없음 | 301,660 | `constant 1e-3` | 100 (10) | 기존 Stage B 기준선 (`configs/trlt/trlt_cross_only.json` 프로브와 동일 설정) |
| `resmlp` | `configs/trlt/attention_actor/probe_resmlp.json` | `ResMLPActionProbe` (`2x336` Pre-LN ResMLP) | 해당 없음 | 764,412 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) | 파라미터 수·스케줄을 맞춘 비-어텐션 대조군 |
| `part_attention_kinematic` | `configs/trlt/attention_actor/probe_part_attn_kinematic.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `kinematic` (`34/144`) | 764,380 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) | 자기 자신 + 부모–자식(`팔↔손가락`) + 양팔 연결(손가락 간 직접 연결 없음) |
| `part_attention_hierarchical` | `configs/trlt/attention_actor/probe_part_attn_hierarchical.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `hierarchical` (`64/144`) | 764,380 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) | `팔→양팔`(손가락→팔 차단) & `손가락→같은 쪽 팔+형제 손가락` |
| `part_attention_full` | `configs/trlt/attention_actor/probe_part_attn_full.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `full` (`144/144`) | 764,380 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) | 전체 부위 간 완전 연결 어텐션 (`resmlp` 대비 파라미터 차이 `0.004%`) |

---

## 2. 전체 테스트 성능 요약 (`seed 0`, 26개 에피소드)

| 모델 ID | 변형 | `ep_macro` (주지표) | `micro_mse` | `obj_macro` | `rmse` | `first_action_mse` | `vs MLP` (`Δ`, 95% CI) | `vs ResMLP` (`Δ`, 95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `trlt_cross_only` | `mlp` | 0.06349 | 0.05890 | 0.07180 | 0.03802 | 0.001320 | `기준선` | — |
| `trlt_cross_only__resmlp` | `resmlp` | 0.06186 | 0.05735 | 0.06897 | 0.03759 | 0.001280 | `-2.56%` (`-0.00163`, `[-0.00457, +0.00165]`) | `기준선` |
| `trlt_cross_only__part_attention_kinematic` | `kinematic` | 0.04043 | 0.03665 | 0.04654 | 0.02573 | 0.000534 | **`-36.32%`** (`-0.02306`, `[-0.03082, -0.01671]`) | **`-34.64%`** (`-0.02143`, `[-0.03018, -0.01401]`) |
| `trlt_cross_only__part_attention_hierarchical` | `hierarchical` | 0.03765 | 0.03375 | 0.04116 | 0.02668 | 0.000572 | **`-40.70%`** (`-0.02584`, `[-0.03608, -0.01824]`) | **`-39.14%`** (`-0.02421`, `[-0.03467, -0.01575]`) |
| `trlt_cross_only__part_attention_full` | `full` | **0.03392** | **0.03099** | **0.03925** | **0.02478** | **0.000512** | **`-46.56%`** (`-0.02956`, `[-0.04163, -0.02022]`) | **`-45.16%`** (`-0.02793`, `[-0.04124, -0.01773]`) |

---

## 3. 신체 부위 그룹별 오차 (`group_mse`, 비정규화)

| 변형 | `arms` | `left_arm` | `right_arm` | `hands` | `left_hand` | `right_hand` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp` (`trlt_cross_only`) | 0.003501 | 0.003236 | 0.003767 | 0.000791 | 0.000921 | 0.000662 |
| `resmlp` | 0.003443 | 0.002730 | 0.004155 | 0.000768 | 0.000926 | 0.000609 |
| `part_attention_kinematic` | **0.001481** | **0.001136** | **0.001826** | 0.000402 | 0.000421 | 0.000382 |
| `part_attention_hierarchical` | 0.001916 | 0.001301 | 0.002532 | **0.000328** | 0.000385 | **0.000271** |
| `part_attention_full` | 0.001502 | 0.001153 | 0.001851 | 0.000332 | **0.000372** | 0.000292 |

---

## 4. 물체별 성능 비교 (`normalized_mse`, 11개 물체)

| 물체 | `mlp` | `resmlp` | `part_attn_full` | `part_attn_hierarchical` | `part_attn_kinematic` | 최고 성능 |
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

## 5. 핵심 분석 및 구조적 해석

1. **용량(파라미터 수) 증가가 아닌 순수 아키텍처 이득 검증:**
   기존 평면 `mlp`(`301,660`)를 `764,412` 파라미터 Pre-LN `resmlp` + Cosine 스케줄로 키웠을 때는 `-2.56%`(`95% CI [-0.00457, +0.00165]`, 0을 포함하여 통계적 유의성 없음)에 그쳤습니다. 반면 동일한 용량(`764,380`, `0.004%` 차이)에서 `PartAttentionActionProbe`를 쓰면 **`resmlp` 대비 `-34.64% ~ -45.16%`** (`mlp` 대비 **`-36.32% ~ -46.56%`**)의 통계적으로 뚜렷한 개선이 나타났습니다.
2. **즉각 제어 오차(`first_action_mse`) 대폭 감소:**
   `part_attention_full`은 첫 스텝 액션 오차를 `0.001320`(`mlp`) / `0.001280`(`resmlp`)에서 `0.000512`로 낮춰 **`mlp` 대비 `-61.2%`, `resmlp` 대비 `-60.0%`** 개선을 보였습니다.
3. **`arms` vs `hands` 분해를 통해 본 `full > hierarchical > kinematic`의 구조적 원인:**
   - `kinematic`은 `팔↔손가락` 양방향 기구학 연결 덕분에 **`arms` 오차가 `0.001481`로 전체 1등**이지만, 같은 손 안의 손가락 간 직접 연결이 없어 `hands` 오차(`0.000402`)에서 손해를 봅니다.
   - `hierarchical`은 같은 손 안의 형제 손가락 간 협응을 열어 **`hands` 오차가 `0.000328`로 전체 1등**(`right_hand: 0.000271`)이며 손가락 정밀 파지·악력 물체 3종(`hand grip exerciser`, `clothespin`, `hair clip`)에서 1등을 기록했습니다. 다만 `손가락→팔` 상향 피드백을 막아 두었기 때문에 `arms` 오차(`0.001916`, 특히 `right_arm: 0.002532`)가 상대적으로 높아졌습니다.
   - `full`은 `팔←손가락` 상향 피드백(`arms: 0.001502`)과 손 내부·양손 협응(`hands: 0.000332`)을 모두 보존해 종합 1등을 달성했으며, 11개 중 8개 물체 1등(전체 11개 물체 모두 `mlp`·`resmlp` 대비 완승)을 기록했습니다.

---

## 6. 한계점 및 후속 실험 계획

- **다중 시드(`seed 1, 2`) 검증:** 현재 결과는 `seed 0` 기준(`missing_expected_seeds: ["1", "2"]`)이므로, 추가 시드에 대한 Stage A 잠재 캐시 export 및 Stage B 반복 검증이 필요합니다.
- **마스크 후속 가설 (`hierarchical_bidir` 및 헤드별 마스킹 `[H, N, N]`):** 위 `arms` vs `hands` 분석에서 확인된 대로 `hierarchical`에 `손가락→같은 쪽 팔` 상향 피드백을 열거나, 4개 어텐션 헤드별로 `kinematic`·손내부·양손 협응·전역 헤드를 분리 배정하는 `[4, 12, 12]` Per-Head Masking 비교가 유망합니다.
- **타 Stage A 표현(`rlt_original`, `rope3d_trlt_cross_only`, `trlt_alternating`) 교차 검증:** 시각 전용(`rlt_original`) 및 3D 기하 임베딩(`rope3d`) 캐시에서도 Stage B 어텐션 액터의 이득이 유지·증폭되는지 확인합니다.
- **Stage 2 `LingBot-Touch` 연동:** 변환 체크포인트 준비 시 `TouchPartAttentionActionProbe`(`PerFingerTouchCrossAttention` + `HaT-Tiny`) 실험으로 확장합니다.
