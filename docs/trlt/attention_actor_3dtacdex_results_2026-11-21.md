# 3DTac-Dex Assembly Stage B 어텐션 액터 실험 결과 (2026-11-21)

- **실험 일자:** 2026-11-21
- **브랜치 / 기준 커밋:** `attention-actor-3dtacdex` (`60ec20b`; `attention-actor` `8e79d4d` + `origin/TRLT-3DTacDex` `52b8a31` 병합)
- **데이터셋 / 태스크:** `3dtacdex_assembly` (`22D` JAKA MiniCobo 6D 팔 + 16D 오른손 LEAP 핸드, `H=3` 액션 청크, `5 Hz`, PaXini 8 패드 × `15×3` 3축 taxel `clip(raw/30, -1, 1)`)
  - **분할 규모 (원본 trajectory 기준 그룹 분할, `split_seed=42`):** Train 77 trajectories (89 segments, 14,819 queries) / Validation 16 trajectories (17 segments, 3,472 queries) / Test 16 trajectories (18 segments, 3,778 queries, 248,112 valid action elements)
- **동결 Stage A 표현:** `trlt_self_only` 2-layer `rawclip30` (`assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z`, `seed0`)
- **평가 시드:** `seed 0` (`seed 0, 1, 2` 다중 시드 및 경량 `d=64` 후속 검증 예정)
- **실행 환경:** `inst-i108` (`NVIDIA RTX PRO 6000 Blackwell`, GPU 0), 출력 경로 `/workspace/users/jiwoo/trlt_attention_actor_3dtacdex_self2_rawclip30_seed0`
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

## 4. 통계적 한계 및 후속 실험 계획

### 4.1 현재 결과의 한계
- **표본 및 시드:** `seed 0` 단일 Stage B 실행이며, 16개 test trajectory 기반 양측 95% episode-bootstrap CI에서 `hierarchical vs resmlp`(`[-0.015511, -0.001828]`)와 `kinematic vs resmlp`(`[-0.012978, -0.000459]`)는 0 미만으로 확정되지만, **`hierarchical vs mlp`는 `[-0.008047, +0.000280]`으로 상한이 0을 미세하게(+0.000280) 걸치고 있습니다.**
- **소규모 데이터 과적합 징후:** `725K` `resmlp`의 급격한 과적합(`val_best 0.070120`)과 `722K` `full`의 상승을 볼 때, `14,819` 쿼리 규모에서는 `d=128`(`722K`)이 다소 큰 편입니다.
- **오프라인 MSE 대 실제 제어:** normalized MSE `-6.27% ~ -12.66%`는 제곱근(RMSE) 기준 약 `-3.2% ~ -6.5%` (Arm MSE `-19.33%`는 Arm RMSE 약 `-10.2%`) 감소에 해당하며, 오프라인 지표 개선이 곧바로 closed-loop 실기 조립 성공률을 보장하지는 않습니다.

### 4.2 후속 실험 (#1 경량화/정규화 + #2 다중 시드 `seed 0,1,2`)
1. **경량 `d=64` (~`201K`, 기존 `mlp` `220K`보다 작음) & Dropout `0.1` 비교:**
   - `configs/trlt/3dtacdex_assembly/attention_actor/`에 추가:
     - `probe_resmlp_d160.json` (`200,386` params, `d=64`와 `-0.35%` 파라미터 매칭)
     - `probe_part_attn_hierarchical_d64.json` (`d=64, L=2, H=4`, `201,090` params)
     - `probe_part_attn_hierarchical_d64_drop01.json` (`201,090` params, `dropout=0.1`)
     - `probe_part_attn_kinematic_d64.json` (`201,090` params)
     - `probe_part_attn_hierarchical_drop01.json` (`d=128`, `721,602` params, `dropout=0.1`)
2. **동일 동결 `seed0` 캐시 위 다중 시드(`--seeds "0 1 2" --latent-seed 0`):**
   - Stage A 캐시(`seed0`)를 공유한 채 Stage B를 `seed 0, 1, 2`로 반복 학습·봉인·평가하고, `summarize`의 `paired_vs_baseline.*.seed_mean`으로 시드 평균 에피소드별 95% CI를 산출합니다.

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
