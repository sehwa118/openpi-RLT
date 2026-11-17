# Fusion Transformer와 Geometry Embedding 실험 정리

작성 기준: **2026-11-17**, fusion 비교·geometry 추가·sensor ID 제거 실험의 학습 및 최종 test 평가 완료 결과.

Geometry 없는 fusion 비교에서는 **TRLT cross-only**가 가장 낮은 test MSE를 보였다.
이후 cross-only와 alternating에 MLP 또는 3D RoPE geometry encoding을 추가했지만,
**이번 seed 0에서는 geometry 없는 동일 구조보다 낮은 MSE를 얻지 못했다.**
추가로 alternating에서 encoder sensor ID를 제거하고 Stage A를 10k로 줄인 두 모델도 평가했다.
MLP·3D RoPE 모두 기존 ID 유지·15k 모델보다 MSE가 높았으나, 두 변경의 효과를 분리할 수는 없다.
모든 결과는 offline action prediction이며, 실제 로봇 제어 성공률을 측정한 결과는 아니다.

## 1. 공통 데이터와 학습·평가 방식

| 항목 | 실험 설정 |
|---|---|
| 데이터 | T-Rex의 `squeeze` 173 demonstrations, 152,169 frames, 11개 object category |
| 데이터 revision | `8b478400a9167d59ede0fdbfea616cb5e9791885` |
| 분할 | 고정 episode split: train 121 / validation 26 / test 26; split seed 42 |
| 학습 seed | **0만 실행** |
| 입력 | Head·left wrist·right wrist RGB의 frozen VLM feature; TRLT는 tactile history와 현재 state도 사용 |
| 시간 계약 | 30 Hz; query 간격 3 frames; RGB anchor 갱신 간격 15 frames |
| Tactile | 좌·우 각 5개 센서의 deformation map; 시점 `[t-9, ..., t]`, 64×64 grayscale |
| State / 예측 대상 | 현재 실제 관절 상태 58D / commanded action `[t, ..., t+5]`, 6×58D |
| 정규화·마스크 | Train-only state/action 통계; history padding과 action horizon 경계는 마스크 처리 |
| 최종 평가 | Test 26 episodes, 7,823 query windows; 모든 결과의 유효 action 원소 수 2,715,096 |

학습은 두 단계로 분리한다.

1. **Stage A — representation 학습:** frozen VLM의 visual feature를 복원하도록 encoder와
   decoder를 학습한다. TRLT는 tactile map도 함께 복원하며 visual/tactile loss 가중치는 각각 1이다.
   RLT original은 visual reconstruction만 학습한다. AdamW learning rate `1e-4`, weight decay
   `1e-4`, effective batch 32 / microbatch 4 / accumulation 8을 사용했다. 초기 fusion·geometry
   비교는 **15,000 updates**, 후속 sensor ID 제거 실험은 **10,000 updates**를 수행했다.
   Epoch 상한과 early stopping은 사용하지 않았다.
2. **Stage B — action probe 학습:** Stage A에서 validation으로 선택한 representation을 고정하고,
   `[flatten(z), normalized current state, visual age]`를 입력으로 받는 action MLP를 학습한다.
   Hidden layer는 `256→256`, 출력은 `6×58`이다. AdamW learning rate `1e-3`, weight decay
   `1e-4`, batch 256, 최대 100 epochs, patience 10을 사용했다.

Stage A는 RLT의 경우 validation visual loss, TRLT의 경우 validation weighted reconstruction
loss로 best checkpoint를 선택한다. Stage B는 validation normalized action micro-MSE로 선택한다.
최종 test에는 두 단계의 **validation-selected best checkpoint**를 사용했다. 모든 선택을 확정한
뒤 test 처리를 진행했으며, test 결과로 checkpoint를 선택하지 않았다.

**최종 primary metric은 normalized action episode-macro MSE**다. 각 episode 안에서 유효한
query×horizon×58D 원소의 MSE를 계산하고, 26개 episode를 동일 가중치로 평균한다.
아래 표는 모두 이 지표이며 **낮을수록 좋다**.

## 2. 구현된 Fusion Transformer

### 입력 token과 공통 구조

TRLT의 token은 다음과 같이 구성된다.

| Token | 구성 |
|---|---|
| Visual `V` | Frozen π0.5/PaliGemma의 final image feature `[768,2048]`를 512D로 projection하고 camera·patch·modality embedding을 더함 |
| Tactile `T` | 센서·시점마다 공유 CNN을 적용해 512D token 생성; 기본 설정은 sensor ID·tactile modality·상대시간 MLP embedding을 더하며, 후속 ablation에서만 encoder sensor ID를 제거 |
| State `s` | 정규화된 현재 58D state와 visual age를 MLP로 변환하고 state modality embedding을 더함 |
| Readout `q` | 각 window에서 새로 시작하는 학습 가능한 token; 최종 출력은 `z[B,1,512]` |

Tactile CNN은 3×3 stride-2 convolution 3개, 채널 `32→64→128`, LayerNorm/GELU와 spatial
flatten/Dense readout을 사용한다. 이미지 내부의 접촉 위치를 보존하도록 2D 픽셀 좌표 채널도
입력한다. 이 좌표는 아래의 **손 기준 3D geometry 좌표와 별개**다.

`U = [T, s]`로 두고, `[V, U, q]`를 fusion transformer에 입력한다. TRLT 세 구조는 모두
**512D, 8 heads, 4 stages, FFN ratio 4, dropout 0**이며, attention 연결만 다르다.
각 stage는 attention과 FFN의 residual update를 사용한다.

### Fusion 방식의 차이

아래에서 `A → B`는 **A의 query가 B의 key/value를 읽는다**는 뜻이다.

| 모델 | Stage 구성 | Visual / tactile·state 사이의 연결 |
|---|---|---|
| RLT original | 기존 RLT encoder 2 layers | Representation encoder는 visual feature만 사용; tactile/state token을 fusion하지 않음 |
| TRLT self-only | Joint self-attention ×4 | `V → V,U`, `U → V,U`; 두 stream을 함께 self-attention |
| TRLT cross-only | Bidirectional cross-attention ×4 | `V → U`, `U → V`; stream 내부 self-attention은 없음 |
| TRLT alternating | Intra → Cross → Intra → Cross | Intra stage는 `V → V`, `U → U`; Cross stage는 `V → U`, `U → V` |

모든 TRLT stage에서 `q`는 `V,U,q` 전체를 읽지만, `V/U`는 `q`를 읽지 않는다.
`U`에는 state token이 포함되므로 intra attention은 tactile·state stream 내부 attention을 뜻한다.
각 stage의 attention update는 같은 이전 hidden state에서 계산된다. 연결 제어는 dense attention의
mask로 구현했으며, sparse attention에 의한 연산량 감소를 전제하지 않는다.

세 구조의 latent와 visual decoder는 같은 형태를 사용한다. TRLT tactile decoder는 sensor·history·
relative-time query와 `z`를 이용해 tactile map을 복원한다. 이 decoder들은 Stage A 학습용이며,
Stage B에서는 고정된 `z`를 action probe에 전달한다.

### Geometry 없는 fusion 비교 결과

| 모델 | Test MSE | RLT 대비 MSE 감소율 |
|---|---:|---:|
| RLT original | 0.071770421 | 기준 |
| TRLT self-only | 0.068767073 | 4.18% |
| **TRLT cross-only** | **0.063485767** | **11.54%** |
| TRLT alternating | 0.064652117 | 9.92% |

관측된 순위는 cross-only, alternating, self-only, RLT 순이다. 다만 세 TRLT와 RLT 간 차이의
paired episode-bootstrap 95% 구간은 모두 0을 포함한다. 따라서 이번 평균값 차이를 여러 seed에
걸쳐 일관된 개선으로 해석할 수는 없다. 또한 TRLT에는 tactile뿐 아니라 representation 단계의
state 입력도 추가되므로, RLT 대비 차이를 tactile만의 효과로 분리할 수 없다.

## 3. 구현된 Geometry Embedding

### 센서 대표점과 좌표 계산

데이터에는 센서 XYZ가 직접 저장되어 있지 않다. 원본 관절 상태와 Sharpa Wave 손 모델로
FK를 계산하고, **각 끝마디에 붙은 elastomer mesh의 axis-aligned bounding-box 중심**을 센서
대표점으로 사용했다. 이는 패드 중심의 근사이며, 보정된 센서 표면 중심이나 실제 접촉점은 아니다.

각 손은 독립적인 손 모델 root frame을 사용한다. 이 원점을 wrist 대표점 `[0,0,0]`으로 정했다.
각 tactile 관측 시각에 맞는 과거 관절 상태로 XYZ를 계산하며, 현재 query의 자세를 history 전체에
반복하지 않는다. 좌표는 meter 단위이고 MLP/RoPE 공통으로 `scaled_xyz = xyz / 0.01`을 사용한다.
이 좌표는 손가락의 굽힘에 따른 위치를 표현하지만, world 좌표에서의 손목 이동이나 양손 간
실제 거리를 표현하지는 않는다.

명세는 [sharpa_wave_v1.json](../../configs/trlt/geometry/sharpa_wave_v1.json)에 기록되어 있다.
손 모델의 upstream commit은 `ba4047ce3e9f3d2013d60de600e136897e90ccb8`이며, source XML/STL hash,
관절 순서와 센서별 offset을 고정했다. 데이터에는 XYZ `[B,10,10,3]`와 유효 mask를 추가했다.

### 후보 A — MLP를 거쳐 tactile token에 더하기

모든 센서·시점이 공유하는 `Linear(3,128) → GELU → Linear(128,512)`를 사용한다.

```text
tactile token = CNN(image) + sensor ID + modality + relative time
              + MLP(scaled XYZ)
```

Fusion 전에 한 번 더하며 기존 sensor/time embedding은 유지한다. Visual token에는 geometry
MLP를 더하지 않는다. Geometry가 유효하지 않으면 추가 항은 0이다.

### 후보 B — 같은 손의 wrist/tactile 쌍에 3D RoPE 적용

각 fusion stage에서 projection된 **Q와 K**를 XYZ에 따라 회전시킨다. Head당 64차원 중
X/Y/Z에 각각 16차원씩 사용하고 나머지 16차원은 회전하지 않는다. RoPE base는 `10000`이다.
V와 reconstruction decoder에는 geometry RoPE를 적용하지 않으며, geometry MLP도 더하지 않는다.

Wrist 영상의 모든 patch에는 해당 손의 wrist 대표점 `[0,0,0]`을 부여한다. 기존 camera/patch
embedding은 유지한다. 이는 영상의 손 연관성을 나타내는 대표 좌표이며 각 patch에 보이는
물체의 3D 위치를 뜻하지 않는다.

| Attention 쌍 | Geometry 처리 |
|---|---|
| 같은 손의 wrist ↔ tactile | RoPE가 적용된 Q/K 내적 |
| 같은 손의 tactile ↔ tactile | RoPE가 적용된 Q/K 내적; 해당 fusion stage가 허용하는 연결에 한함 |
| 같은 wrist의 patch ↔ patch | 동일한 원점의 회전이므로 기존 내적과 같음 |
| 서로 다른 손의 token 쌍 | 기존 내적 유지 |
| Head camera, state, readout `q`가 포함된 쌍 | 기존 내적 유지 |

이 방식은 하나의 전역 좌표계를 모든 token에 적용하는 RoPE가 아니라 **쌍을 선택하는 axial
3D RoPE**다. 기존 fusion attention mask는 그대로 적용한다. 따라서 cross-only에도 wrist ↔
tactile 경로로 geometry가 반영되지만, tactile ↔ tactile 연결이 새로 생기지는 않는다.

### Geometry 추가 실험 결과 — encoder sensor ID 유지, Stage A 15k

Geometry 실험은 **cross-only와 alternating에 대해서만** 수행했다. 비교 기준은 위의 geometry
없는 동일 fusion 구조다. 기존 baseline은 재학습하지 않고 앞선 완료 실험의 평가를 재사용했다.
데이터 split, visual feature, 정규화와 query 계약은 공유했다.

| Fusion | Geometry | Test MSE | Geometry 없는 동일 구조 대비 변화 |
|---|---|---:|---:|
| Cross-only | MLP | 0.075159157 | +18.39% |
| Alternating | MLP | 0.065073679 | +0.65% |
| Cross-only | 3D RoPE | 0.067697001 | +6.63% |
| Alternating | 3D RoPE | 0.073425785 | +13.57% |

양수 변화율은 MSE 증가, 즉 악화를 뜻한다. 네 geometry 추가 모델 모두 이번 seed에서는
동일 구조의 geometry 없는 baseline보다 MSE가 높았다. 추가 실험 안에서는 MLP alternating이
가장 낮았으나, 기존 alternating의 `0.064652117`보다 낮지는 않았다.

차이 `geometry MSE − 동일 구조 baseline MSE`의 paired episode-bootstrap 결과는 다음과 같다.

| Fusion + Geometry | MSE 차이 | Paired episode-bootstrap 95% 구간 |
|---|---:|---|
| Cross-only + MLP | +0.011673391 | [0.001605347, 0.027877486] |
| Alternating + MLP | +0.000421561 | [-0.003293762, 0.003972811] |
| Cross-only + 3D RoPE | +0.004211235 | [-0.001468331, 0.008584147] |
| Alternating + 3D RoPE | +0.008773668 | [-0.002541287, 0.025260941] |

구간은 26개 episode를 단위로 2,000회 bootstrap한 값이다. Cross-only + MLP의 구간은 양수이고,
나머지 세 구간은 0을 포함한다. 이 구간은 **seed 0에 조건부인 episode 변동성**이며,
학습 seed 사이의 변동성은 추정하지 않는다.

## 4. Encoder Sensor ID 제거 실험 — Alternating, Stage A 10k

두 조합 모두 `use_sensor_id_embedding=false`로 설정해 fusion encoder의 학습 가능한
`sensor_embed`와 tactile token에 더하는 ID 항을 제거했다. **재구성 decoder의 `sensor_query`는
출력 센서 slot을 구분하도록 유지**했다. Modality·상대시간·camera/patch embedding, geometry
XYZ와 RoPE의 같은 손 grouping도 유지했다. 따라서 학습 시스템 전체의 모든 센서 정체성을
제거한 실험은 아니다.

Fusion은 alternating만, seed는 0만 사용했다. Stage A는 **정확히 10,000 optimizer updates**,
Stage B와 validation checkpoint 선택·최종 평가 방식은 이전과 같다. 두 모델의 checkpoint
선택은 2026-11-16 13:34:35 UTC에 확정했고, 이후 test 평가를 수행했다.

| Geometry | Encoder ID 유지·15k Test MSE | Encoder ID 제거·10k Test MSE | 이전 모델 대비 변화 |
|---|---:|---:|---:|
| MLP | 0.065073679 | 0.066686607 | +2.48% |
| 3D RoPE | 0.073425785 | 0.076711648 | +4.48% |

양수 변화율은 MSE 증가다. 새 두 조합에서는 MLP가 더 낮은 MSE를 보였으나, ID 제거·10k
모델 모두 과거의 같은 geometry·ID 유지·15k 모델을 넘지 못했다. Geometry 없는 기존
alternating의 `0.064652117`보다도 두 모델의 MSE가 높았다.

| 모델 | Stage A 실제 updates | 선택된 Stage A step | Stage B 실제 updates | 선택된 Stage B step |
|---|---:|---:|---:|---:|
| ID 제거 + MLP alternating | 10000 | 7791 | 5460 | 4060 |
| ID 제거 + 3D RoPE alternating | 10000 | 10000 | 4900 | 3500 |

차이 `ID 제거·10k MSE − ID 유지·15k MSE`의 paired episode-bootstrap 95% 구간은 다음과 같다.

| Geometry | MSE 차이 | Paired episode-bootstrap 95% 구간 |
|---|---:|---|
| MLP | +0.001612928 | [-0.005696915, 0.010201950] |
| 3D RoPE | +0.003285862 | [-0.002165708, 0.009097375] |

두 구간 모두 0을 포함하며, seed 0에서의 episode 변동성만 반영한다.
**Encoder sensor ID와 학습 budget을 동시에 변경했으므로, sensor ID 제거만의 효과로 해석할 수
없다.** Budget을 일치시킨 ID 유지·10k 대조군은 실행하지 않았다. 기존 15k 모델은 재학습하지
않고 완료된 평가를 비교 기준으로 재사용했다. 반복 사용한 기존 test set이므로 새로 확보한
untouched holdout 결과도 아니다. 각 실행의 checkpoint는 여전히 validation으로만 선택했다.

## 5. 결과의 해석 범위

이번 설정에서 가장 낮은 primary MSE는 **geometry 없는 TRLT cross-only의 0.063485767**이다.
현재의 손 기준 XYZ, 대표점 근사, MLP 및 쌍별 3D RoPE 설정에서는 geometry 추가의 개선을
관찰하지 못했다. 이 결과만으로 geometry 정보 자체가 불필요하다고 결론 내릴 수는 없다.

결과를 해석할 때 다음을 함께 고려해야 한다.

- Seed 0만 실행했으므로 여러 학습 seed에서의 순위 안정성은 확인하지 않았다.
- Geometry XYZ에는 **과거 관절 상태에서 유도한 정보**가 들어간다. Geometry 표현 효과와
  관절 history 추가 효과를 분리하려면 별도의 history 입력 대조군이 필요하다.
- MLP는 학습 파라미터를 추가하고 RoPE는 attention의 위치 의존성을 바꾼다. 두 방식은
  parameter 수가 일치하는 비교가 아니다.
- 센서 중심은 모델 기반 근사이고, wrist 영상의 patch별 scene geometry나 센서 방향은 사용하지 않았다.
- Test는 학습에서 본 object category의 새로운 demonstration이다. 새로운 object/category로의
  일반화나 online closed-loop 제어 성능을 입증하지 않는다.

## 6. 실행 근거와 구현 위치

| 실험 | Run ID | 실제 완료 시각 (UTC) |
|---|---|---|
| Geometry 없는 fusion 비교 | `squeeze_15k_seed0_wandb_20261114T081820Z` | 2026-11-15 04:17:42 |
| Geometry 추가 비교 | `squeeze_geometry_15k_seed0_20261115T074104Z` | 2026-11-16 05:37:39 |
| Alternating encoder sensor ID 제거·10k | `squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z` | 2026-11-16 13:39:08 |

아래 execution artifact는 Git에서 제외된 서버 로컬 파일이다. 새 clone에는 없을 수 있다.

- Fusion 비교: [최종 보고서](../../experiments/trlt/runs/squeeze_15k_seed0_wandb_20261114T081820Z/final_report.md),
  [실제 설정](../../experiments/trlt/runs/squeeze_15k_seed0_wandb_20261114T081820Z/configs/),
  [checkpoint 선택](../../experiments/trlt/runs/squeeze_15k_seed0_wandb_20261114T081820Z/final_selections.json).
- Geometry 비교: [최종 보고서](../../experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/final_report.md),
  [실제 설정](../../experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/configs/),
  [동일 구조 대비 차이·구간](../../experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/paired_vs_architecture.json),
  [episode·유효 원소 수 일치 확인](../../experiments/trlt/runs/squeeze_geometry_15k_seed0_20261115T074104Z/comparison_validation.json).
- Sensor ID 제거 비교: [최종 보고서](../../experiments/trlt/runs/squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z/final_report.md),
  [실제 설정](../../experiments/trlt/runs/squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z/configs/),
  [과거 모델 대비 차이·구간](../../experiments/trlt/runs/squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z/historical_comparisons.json),
  [checkpoint 선택](../../experiments/trlt/runs/squeeze_no_sensor_id_alt_10k_seed0_20261116T062348Z/final_selections.json).
- 세 run의 `status.json`과 `events.log`는 각각 전체 28/32/20단계 완료를 기록한다. 각 모델의
  `seed0/evaluation/metrics.json`, `predictions.jsonl.gz`에 상세 평가와 예측이 저장되어 있다.
- 구현: [fusion·CNN·decoder](../../src/openpi/models/trlt.py),
  [3D RoPE와 적용 쌍](../../src/openpi/models/trlt_geometry.py),
  [FK와 geometry cache](../../src/openpi/trlt/geometry.py).
  상세 계약과 CLI는 [README.md](README.md), [geometry.md](geometry.md)를 참조한다.
