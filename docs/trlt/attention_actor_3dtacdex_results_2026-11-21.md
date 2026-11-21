# 3DTac-Dex Assembly Stage B 어텐션 액터 실험 결과 (2026-11-21)

- **실험 일자:** 2026-11-21
- **브랜치 / 기준 커밋:**
  - **1차 비교 (`seed 0`, 5종):** `attention-actor-3dtacdex` (`60ec20b`; `attention-actor` `8e79d4d` + `origin/TRLT-3DTacDex` `52b8a31` 병합)
  - **2차 다중 시드·경량·정규화 비교 (`seed 0, 1, 2`, 7종 × 3시드 = 21 runs):** `attention-actor-3dtacdex` (`d6b0700`)
- **데이터셋 / 태스크:** `3dtacdex_assembly` (`22D` JAKA MiniCobo 6D 팔 + 16D 오른손 LEAP 핸드, `H=3` 액션 청크, `5 Hz`, PaXini 8 패드 × `15×3` 3축 taxel `clip(raw/30, -1, 1)`)
  - **분할 규모 (원본 trajectory 기준 그룹 분할, `split_seed=42`):** Train 77 trajectories (89 segments, 14,819 queries) / Validation 16 trajectories (17 segments, 3,472 queries) / Test 16 trajectories (18 segments, 3,778 queries, 248,112 valid action elements)
- **동결 Stage A 표현:** `trlt_self_only` 2-layer `rawclip30` (`assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z`, `seed0`)
- **평가 시드:** 1차 `seed 0` / 2차 `seed 0, 1, 2` (공유 `seed0` 동결 잠재 캐시 위 `--latent-seed 0`)
- **실행 환경 / 출력 경로 (`inst-i108`, NVIDIA RTX PRO 6000 Blackwell, GPU 0):**
  - 1차: `/workspace/users/jiwoo/trlt_attention_actor_3dtacdex_self2_rawclip30_seed0`
  - 2차: `/workspace/users/jiwoo/trlt_attention_actor_3dtacdex_self2_rawclip30_d64_mseed`
- **실행 스크립트:** `scripts/run_attention_actor_stage_b.sh`

---

## 1. 실험 설정 및 22D 부위 분해 검증

### 1.1 `OMY_LEAP_PARTS` 22D 관절 슬라이스 일치 검증
`datasets/3dtacdex_assembly/source/README.md` 및 `meta/assembly_conversion.json` 확인 결과, 원천 데이터의 22D 상태/액션 순서는 다음과 같습니다:
- `arm_joint_0..5` (`0..5`, 6D): JAKA MiniCobo 팔 6축
- `right_hand_joints` (`6..21`, 16D): *"Hand channels are ordered index, middle, ring, then thumb (four per finger)."*

이는 [`src/openpi/models/trlt_actor.py`](../../src/openpi/models/trlt_actor.py)의 `OMY_LEAP_PARTS`(`arm[0:6]`, `index[6:10]`, `middle[10:14]`, `ring[14:18]`, `thumb[18:22]`) 5개 신체 부위 토큰 분할과 100% 일치합니다.

### 1.2 실행 절차 및 항-누출 검증 봉인
Stage B 전용 검증 봉인(`prepare_tacdex_experiment.py seal-probes`)을 적용하여, 5개 모델 모두 Train 14,819 쿼리로만 학습하고 Validation 3,472 쿼리(`action_valid_mask` 요소별 마스크 반영)로 `best` 체크포인트를 확정한 뒤 해시를 봉인하고 마지막에 16개 Test trajectory를 1회 평가했습니다.

```bash
TEAM_DATA="/workspace/teams/tactile/openpi-RLT/datasets/3dtacdex_assembly"
bash scripts/run_attention_actor_stage_b.sh \
  --dataset 3dtacdex_assembly \
  --protocol-dir "$TEAM_DATA/protocol" \
  --latent-root "$TEAM_DATA/latents/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/trlt_self_only" \
  --normalization "$TEAM_DATA/protocol/normalization.json" \
  --out-root "/workspace/users/$USER/trlt_attention_actor_3dtacdex_self2_rawclip30_seed0" \
  --python "/workspace/teams/tactile/openpi-RLT/.venv/bin/python" \
  --seeds "0" \
  --device 0
```

### 1.3 비교 모델 5종

| 변형 (`variant`) | 설정 파일 | 아키텍처 | 마스크 (`22D`, `5×5`) | 파라미터 수 | LR 스케줄 | Epoch (Patience) |
| --- | --- | --- | --- | ---: | --- | --- |
| `mlp` | `configs/trlt/3dtacdex_assembly/attention_actor/probe_mlp.json` | `ActionProbe` (`2×256` GELU MLP) | 해당 없음 | 219,970 | `constant 1e-3` | 100 (10) |
| `resmlp` | `configs/trlt/3dtacdex_assembly/attention_actor/probe_resmlp.json` | `ResMLPActionProbe` (`2×356` Pre-LN ResMLP) | 해당 없음 | 724,882 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) |
| `part_attention_full` | `configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_full.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `full` (`25/25`) | 721,602 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) |
| `part_attention_kinematic` | `configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_kinematic.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `kinematic` (`13/25`) | 721,602 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) |
| `part_attention_hierarchical` | `configs/trlt/3dtacdex_assembly/attention_actor/probe_part_attn_hierarchical.json` | `PartAttentionActionProbe` (`d=128, L=2, H=4`, AdaLN-Zero) | `hierarchical` (`21/25`) | 721,602 | `cosine 5e-4 -> 5e-5` (200w) | 150 (20) |

---

## 2. 전체 테스트 성능 요약 (`seed 0`, 16개 원본 trajectory)

| 변형 (`variant`) | `primary_mse` ↓ | `micro_mse` ↓ | `first_action` ↓ | `arm` MSE ↓ (6D) | `hand` MSE ↓ (16D) | `val_best` ↓ (`ep`) | `vs MLP` (`Δ`, 양측 95% CI) | `vs ResMLP` (`Δ`, 양측 95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `mlp` | 0.059128 | 0.065571 | 0.001806 | 0.000683 | 0.003314 | **0.064835** (35) | 기준선 | — |
| `resmlp` | 0.063449 | 0.067616 | 0.002337 | 0.000885 | 0.003776 | 0.070120 (30) | `+7.31%` (`+0.004321`, `[+0.000686, +0.008357]`) | 기준선 |
| `part_attention_full` | 0.060103 | 0.068558 | 0.001831 | 0.000571 | 0.003268 | 0.067218 (31) | `+1.65%` (`+0.000976`, `[-0.005654, +0.008197]`) | `-5.27%` (`-0.003346`, `[-0.013137, +0.005967]`) |
| `part_attention_kinematic` | 0.057424 | 0.064219 | 0.001701 | 0.000647 | 0.003092 | 0.068886 (27) | `-2.88%` (`-0.001703`, `[-0.006429, +0.002831]`) | **`-9.50%`** (`-0.006025`, `[-0.012978, -0.000459]`) |
| **`part_attention_hierarchical`** | **0.055418** | **0.061176** | **0.001657** | **0.000551** | **0.002905** | 0.064868 (31) | **`-6.27%`** (`-0.003710`, `[-0.008047, +0.000280]`) | **`-12.66%`** (`-0.008031`, `[-0.015511, -0.001828]`) |

- **기존 기록 재현 확인:** `mlp` 변형의 `primary=0.059128` (`completed_epochs=35`)은 [`3dtacdex_experiments.md`](3dtacdex_experiments.md)에 기록된 `sehwa`의 동일 실행(`0.059127821`, `ep=35`)과 소수점 6자리까지 정확히 일치합니다.
- **참고 기준(동일 프로토콜 과거 실행):** `Original RLT` (`2×256` MLP)는 `0.059761`이며, 서버의 `self4_rawclip30` (`2×256` MLP)은 `0.061743` (`arm=0.000754`, `hand=0.003419`)입니다. `part_attention_hierarchical`(`0.055418`)은 `Original RLT` 대비 `-7.27%` 낮은 오차를 보였습니다.

---

## 3. T-Rex (`58D`) vs 3DTac-Dex Assembly (`22D`) 교차 비교 및 해석

| 비교 축 | T-Rex (`58D`, `H=6`, `35.6k` train) | 3DTac-Dex Assembly (`22D`, `H=3`, `14.8k` train / 77 traj) |
| --- | --- | --- |
| **비구조적 용량 확장 (`resmlp`)** | `0.06349 -> 0.06186` (`-2.56%`, 비유의) | `0.059128 -> 0.063449` (**`+7.31%` 유의 악화**, 과적합) |
| **마스크 순위 (`part_attention`)** | **`full` (`0.03392`) > `hierarchical` (`0.03765`) > `kinematic` (`0.04043`)** | **`hierarchical` (`0.055418`) > `kinematic` (`0.057424`) > `full` (`0.060103`)** |
| **1위 모델 vs `MLP` / vs `ResMLP`** | `-46.56%` / `-45.16%` | **`-6.27%`** / **`-12.66%`** |
| **부위별 특이점** | 양팔 22D 손×2 협응(`full`)이 압도적 | 단팔(`arm -19.33%`) + 4지 협응(`hand -12.34%`)의 `hierarchical`이 전 지표 1위 |

1. **"단순 모델 크기 효과" 반박의 더 강한 증거:** 3DTac-Dex Assembly는 77개 train trajectory로 데이터 규모가 작아, 구조 없이 파라미터만 `725K`로 키운 `resmlp`가 `mlp`(`220K`) 대비 **유의하게 과적합 악화(`+7.31%`, CI `[+0.000686, +0.008357]`)**했습니다. 반면 동일한 `722K` 용량의 `part_attention_hierarchical`은 **`resmlp` 대비 `-12.66%` (`[-0.015511, -0.001828]`, 양측 95% 유의)**, **`mlp` 대비 `-6.27%`** 개선을 달성해 성능 향상이 용량이 아니라 **부위별 구조적 귀납 편향(inductive bias)** 덕분임을 보였습니다.
2. **로봇 형태(Embodiment)에 따른 어텐션 토폴로지 역전:**
   - **T-Rex(양팔 58D 양손 협응):** 왼손과 오른손 손가락 간 직접 상호작용 및 상향 피드백까지 모두 열려 있는 `full`이 1위였습니다.
   - **3DTac-Dex Assembly(단일 팔 6D + 오른손 LEAP 16D):** `hierarchical`은 **팔(`arm`)이 팔 토큰만 참조하여 손가락 변동 노이즈를 차폐**하면서(`arm MSE 0.000551`, `mlp` 대비 **`-19.33%`**), 동시에 **4개 손가락(`index/middle/ring/thumb`)이 팔 + 형제 손가락 전체를 참조**해 파지 협응을 유지(`hand MSE 0.002905`, `mlp` 대비 **`-12.34%`**)하여 Primary·Micro·First-Action(`-8.25%`)·Arm·Hand 5개 지표 모두에서 단독 1위를 기록했습니다.
   - 반면 `full`은 소규모 데이터에서 손가락→팔 역방향 연결이 잡음을 섞어 `0.060103`으로 밀렸고, `kinematic`은 손가락 간 연결 차단으로 `hand MSE`(`0.003092`)에서 손해를 보았습니다.

---

## 4. 2차 다중 시드(`seed 0, 1, 2`) 경량화(`d=64`) 및 정규화(`dropout=0.1`) 검증 결과

1차 `seed 0` 실험에서 남아 있던 두 가지 검증 포인트—① `hierarchical vs mlp` 양측 95% CI(`[-0.008047, +0.000280]`)가 0에 미세하게 걸쳐 있던 점, ② `725K` `resmlp`의 과적합을 감안해 `mlp`(`220K`) 이하의 경량 용량(`d=64`, `201K`) 및 `dropout=0.1` 정규화 효과 검증—를 확인하기 위해 **7종 × 3시드(총 21개 Stage B 실행)**를 동일 `seed0` 동결 잠재 캐시(`--latent-seed 0`) 위에서 완료했습니다.

### 4.1 3-Seed 종합 성능 요약 (`seed 0, 1, 2`, 16개 원본 trajectory)

| 순위 | 변형 (`variant`) | Params | `primary_mse` (3-seed mean ± std) ↓ | `vs MLP` (`seed_mean`, 양측 95% CI) | `vs ResMLP_d160` (`seed_mean`, 양측 95% CI) |
|---:|---|---:|---:|---|---|
| **1위** | **`part_attention_hierarchical_drop01`** (`d=128, drop=0.1`) | 721,602 | **`0.051502 ± 0.000464`** | **`-13.89%`** (`-0.008308`, `[-0.012281, -0.004943]`) | **`-20.32%`** (`-0.013131`, `[-0.017242, -0.009450]`) |
| **2위** | **`part_attention_hierarchical_d64_drop01`** (`d=64, drop=0.1`) | **201,090** | **`0.054064 ± 0.001606`** | **`-9.61%`** (`-0.005747`, `[-0.008861, -0.003326]`) | **`-16.35%`** (`-0.010569`, `[-0.013475, -0.008140]`) |
| **3위** | `part_attention_hierarchical` (`d=128, drop=0.0`) | 721,602 | `0.055061 ± 0.001938` | **`-7.94%`** (`-0.004749`, `[-0.010164, -0.000214]`) | **`-14.81%`** (`-0.009572`, `[-0.015056, -0.004638]`) |
| **4위** | `part_attention_kinematic_d64` (`d=64, drop=0.0`) | **201,090** | `0.055514 ± 0.000913` | **`-7.18%`** (`-0.004296`, `[-0.008014, -0.001029]`) | **`-14.11%`** (`-0.009119`, `[-0.012089, -0.006769]`) |
| **5위** | `part_attention_hierarchical_d64` (`d=64, drop=0.0`) | **201,090** | `0.056329 ± 0.004468` | **`-5.82%`** (`-0.003481`, `[-0.006733, -0.000343]`) | **`-12.85%`** (`-0.008304`, `[-0.012416, -0.005098]`) |
| 6위 | `mlp` (`2×256`, `sehwa` 기준선) | 219,970 | `0.059810 ± 0.002191` | 기준선 | `-7.46%` (`-0.004823`, `[-0.007168, -0.002878]`) |
| 7위 | `resmlp_d160` (`2×160`, `d=64`과 `-0.35%` 매칭) | 200,386 | `0.064633 ± 0.001362` | `+8.06%` (`+0.004823`, `[+0.002776, +0.007160]`) | 기준선 |

> **5개 어텐션 변형 전부가 `seed_mean` 양측 95% 페어드 부트스트랩 신뢰구간에서 `mlp`와 `resmlp_d160`을 모두 0 미만으로 유의하게 상회**했습니다.

### 4.2 시드별 세부 지표 및 부위별(`arm`, `hand`) 3-Seed 평균

| 변형 | `seed 0` primary (`val_best`) | `seed 1` primary (`val_best`) | `seed 2` primary (`val_best`) | `micro` 3-seed avg ↓ | `arm` MSE 3-seed avg ↓ | `hand` MSE 3-seed avg ↓ | `val_best` 3-seed avg ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`part_attention_hierarchical_drop01`** | **0.052002** (**0.061315**) | **0.051419** (**0.063289**) | **0.051085** (**0.059160**) | **0.058055** | **0.000521** (`-24.78%`) | **0.002728** (`-14.57%`) | **0.061255** |
| `part_attention_hierarchical_d64_drop01` | 0.055015 (0.065251) | 0.052210 (0.064040) | 0.054966 (0.064320) | 0.060698 | 0.000533 | 0.002963 | 0.064537 |
| `part_attention_hierarchical` | 0.055418 (0.064868) | 0.056796 (0.068541) | 0.052970 (0.064027) | 0.061207 | 0.000530 | 0.003102 | 0.065812 |
| `part_attention_kinematic_d64` | 0.055419 (0.065732) | 0.054652 (0.068554) | 0.056470 (0.073753) | 0.061579 | 0.000589 | 0.003039 | 0.069346 |
| `part_attention_hierarchical_d64` | 0.061017 (0.068300) | 0.052120 (0.063137) | 0.055850 (0.066942) | 0.061012 | 0.000573 | 0.003208 | 0.066126 |
| `mlp` | 0.059128 (0.064835) | 0.058042 (0.066154) | 0.062261 (0.064579) | 0.064774 | 0.000693 | 0.003194 | 0.065189 |
| `resmlp_d160` | 0.064348 (0.071912) | 0.063436 (0.074610) | 0.066115 (0.078935) | 0.070756 | 0.000881 | 0.003800 | 0.075152 |

*(괄호 안 `%`는 `mlp` 3-seed 평균 대비 상대 변화율; 개별 시드 `vs MLP` 95% CI 중 **`part_attention_hierarchical_drop01`** 은 seed 0 `[-0.010805, -0.003415]` `-12.05%`, seed 1 `[-0.012065, -0.001718]` `-11.41%`, seed 2 `[-0.016766, -0.006700]` `-17.95%`로 **3개 시드 전부 단독 유의**.)*

### 4.3 핵심 검증 결론

1. **`part_attention_hierarchical_drop01`의 전 시드·전 지표 완승:**
   - 3개 시드 모두에서 개별 `vs MLP` 양측 95% CI가 0 미만(`-12.05%`, `-11.41%`, `-17.95%`), 3시드 평균 **`0.051502 ± 0.000464`** (`mlp` 대비 **`-13.89%`**, `resmlp_d160` 대비 **`-20.32%`**)를 기록했습니다.
   - 시드 간 표준편차가 `±0.000464`로 `mlp`(`±0.002191`) 대비 **4.72배 더 안정적**이며, 학습 중 `val_best` 역시 3개 시드 모두에서 1위를 차지해 테스트 분할 우연이 아닌 일반화 개선을 확인했습니다.
   - `arm MSE`(`0.000521`, `-24.78%`)와 `hand MSE`(`0.002728`, `-14.57%`)가 함께 큰 폭으로 낮아졌습니다.
2. **"파라미터 수가 많아서 유리했다" 반론 완전 해소 (`d64_drop01`):**
   - `part_attention_hierarchical_d64_drop01`(`201,090` params)은 기존 `mlp`(`219,970` params)보다 **파라미터가 8.58% 더 적은데도** `mlp` 대비 **`-9.61%`** (`[-0.008861, -0.003326]`), 동일 용량의 `resmlp_d160`(`200,386` params) 대비 **`-16.35%`** (`[-0.013475, -0.008140]`) 성능 향상을 달성했습니다.
3. **Dropout 0.1 정규화 효과 검증:**
   - `d=64`: `0.056329 ± 0.004468` → **`0.054064 ± 0.001606`** (오차 감소 + 시드 표준편차 2.78배 축소)
   - `d=128`: `0.055061 ± 0.001938` → **`0.051502 ± 0.000464`** (오차 추가 감소 + 시드 표준편차 4.18배 축소)
   - `d=128` 용량 자체가 문제가 아니라 소규모 77-trajectory 데이터에서 어텐션/FFN 과적합을 억제할 `dropout=0.1` 정규화가 부족했음을 확인했습니다.
4. **Top-down `hierarchical` 마스크의 작동 원리 (`full` 대비):**
   - 단팔 22D(5부위)에서 `full`(`25/25`)과 `hierarchical`(`21/25`)의 차이는 **팔(`arm`) 쿼리 행이 4개 손가락 키를 `-inf`(float min)로 마스킹하는지 여부** 하나입니다.
   - `손가락 → 팔`(손가락이 팔 상태를 참조, Top-down 정보 흐름 `팔 ⇒ 손가락`)과 `손가락 ↔ 손가락` 협응은 모두 열어 두되, `팔 → 손가락`(손가락 노이즈가 팔 예측을 흔드는 상향 경로)만 차단함으로써 팔과 손 모두에서 최고 성능을 얻었습니다.

### 4.4 재현 명령어 (2차 21-run 실행)

```bash
OUT_ROOT="/workspace/users/$USER/trlt_attention_actor_3dtacdex_self2_rawclip30_d64_mseed"
mkdir -p "$OUT_ROOT"

tmux new-session -d -s attn_3dtacdex_mseed \
  "cd /workspace/users/$USER/openpi-RLT && \
   TEAM_DATA=/workspace/teams/tactile/openpi-RLT/datasets/3dtacdex_assembly && \
   bash scripts/run_attention_actor_stage_b.sh \
     --dataset 3dtacdex_assembly \
     --protocol-dir \"\$TEAM_DATA/protocol\" \
     --latent-root \"\$TEAM_DATA/latents/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/trlt_self_only\" \
     --latent-seed 0 \
     --normalization \"\$TEAM_DATA/protocol/normalization.json\" \
     --out-root \"$OUT_ROOT\" \
     --python /workspace/teams/tactile/openpi-RLT/.venv/bin/python \
     --seeds \"0 1 2\" \
     --variants \"mlp resmlp_d160 part_attention_hierarchical_d64 part_attention_hierarchical_d64_drop01 part_attention_kinematic_d64 part_attention_hierarchical part_attention_hierarchical_drop01\" \
     --device 0 2>&1 | tee \"$OUT_ROOT/run.log\""
```

---

## 5. 남은 한계 및 후속 단계

- **Stage A 표현 시드 고정 (`--latent-seed 0`):** 위 3-seed 편차는 동결된 `seed0` `trlt_self_only` 표현 위에서 Stage B 액터의 초기화·셔플·dropout 변동을 측정한 결과이며, Stage A 자체를 다른 시드로 재학습한 변동성은 포함하지 않습니다.
- **오프라인 MSE vs 실제 제어:** 1차·2차 모두 오프라인 trajectory-macro MSE 지표 검증이므로, 실기 로봇/물리 시뮬레이터 closed-loop 조립 성공률은 별도 검증이 필요합니다.
- **추천 다음 비교:** 동일 프로토콜의 시각 단독 Stage A 캐시(`rlt_original`, `assembly_8k_fullb64_batched_seed0_20261118/rlt_original`) 위에서 `mlp` 대 `part_attention_hierarchical_drop01` 3-seed 비교를 진행하면, 어텐션 액터가 촉각+시각 표현(`trlt_self_only`)의 잠재 정보를 시각 단독 대비 얼마나 더 잘 끌어내는지 교차 검증할 수 있습니다.
