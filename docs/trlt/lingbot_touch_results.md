# LingBot-Touch-Tiny 실험 결과 — 2026-11-23

실행 ID: `lingbot_sequential_seed0_20261122T230902Z`. T-Rex 학습·평가 후3DTac-Dex Assembly를 순차 실행하여
**19단계를 모두 완료**했다. 종료 시각은2026-11-23 07:27:39 UTC(16:27:39 KST),
전체 경과 시간은약8.31시간이다. 추가 학습/추출/평가 없이 저장된 결과를 분석했다.

## 주요 결과

Primary는 normalized action MSE를 각 test episode(T-Rex) 또는 원본 trajectory(Assembly)
안에서 계산한 뒤 동등 평균한 값이다. **낮을수록 좋다.** 두 데이터셋은 프로토콜이 달라
서로의 절대 score로 우열을 비교하지 않는다.

| 데이터 / fusion | 기존 scratch encoder | Frozen LingBot Tiny | 상대 오차 감소 | Test 단위 |
|---|---:|---:|---:|---|
| T-Rex / self-only4층 | CNN 0.068767073 | **0.065025158** | **5.44%** |26episodes|
| 3DTac-Dex / self-only2층 | MLP 0.059127821 | **0.056652760** | **4.19%** |16원본trajectories|

Assembly baseline은 **동일한 signed `clip(raw/30,-1,1)`을 사용하는 self-only2층**의
`assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z`다.
mean/std 전처리를 사용한 과거 결과나4층 결과와 섞어 비교하지 않았다.
T-Rex baseline은 `squeeze_15k_seed0_wandb_20261114T081820Z`의 `trlt_self_only`다.

두 비교 모두 seed0, split/normalization/query/visual-feature hash, 시간 계약,
fusion 차원, episode별 valid-element 수, Stage A update 예산 및 Stage B 설정을 확인했다.
T-Rex test는2,715,096개, Assembly test는248,112개 valid action elements를 사용했다.

## 학습과 checkpoint 선택

| 항목 | T-Rex | Assembly |
|---|---|---|
| Stage A 실제 완료 |15,000updates|8,000updates|
| Stage A batch/micro |32/4|64/64|
| Stage A 선택 best step |7,791|8,000|
| Stage B 실제 완료 |4,760updates,34epochs|1,566updates,27epochs|
| Stage B 선택 best |3,360step,24epoch|986step,17epoch|
| Stage B best validation action MSE |0.060246851|0.065398531|

Stage A는 지정 예산까지 모두 학습했으며, validation reconstruction loss로 선택한
`best`를 고정하여 latent를 export했다. Stage B는 동일한256×2 MLP와
max100epochs/patience10을 사용했다. 실제 최종 평가는 Stage A/B의 validation-selected
`best`를 사용했다. **T-Rex Stage A가7,791step에서 중단된 것은 아니다.**
선택된 params/state/metadata hash가 실제 파일과 일치하고, 두 stage 선택 봉인 후에
각 test tactile 특징 추출을 시작한 것도 events와 seal로 검증했다.

## 기존 대비 시간 pooling과 tactile token 구성의 변화

기존 CNN/MLP는 **시점×센서마다1개 token**을 생성했다. Fusion 전에 시간축을
명시적으로 합치지 않고 각 시점의 시간 embedding을 더한 뒤, Fusion Transformer가
전체 history token을 직접 처리했다. 이번 구현은 **센서당1개 token**을 전달하기 위해
LingBot 출력에 taxel mean과 시간 weighted pooling을 순서대로 적용한다.

| 데이터 | 기존 scratch 경로 | 이번 frozen LingBot 경로 | Fusion tactile token 수 |
|---|---|---|---|
| T-Rex | CNN이10시점×10센서를 각각 인코딩 |10프레임→5tubelets→센서별 taxel mean→시간 가중평균 |**100→10**|
| Assembly | MLP가3시점×8pad를 각각 인코딩 |3프레임→prefix 복제로4프레임→2tubelets→pad별 taxel mean→시간 가중평균 |**24→8**|

위 숫자는 **tactile token만** 센 것이다. Visual/state/readout token은 포함하지 않는다.
Fusion 층수는 각 baseline과 동일하게 T-Rex4층, Assembly2층을 유지했다.

### 서로 다른 두 pooling 축

1. **Taxel mean pooling:** 같은 tubelet·센서 안의 taxel feature를 평균한다.
   T-Rex는 센서당96개, PaXini는 pad당15개 feature를 평균한다. PaXini 입력은
   finger slot마다 tip15+pulp15로 구성해8pad가 공동 attention을 수행하지만,
   pooling에서는 tip과 pulp를 구분하므로8개 pad가4개 finger token으로 합쳐지지 않는다.
2. **시간 weighted pooling:** 위에서 얻은 tubelet별 센서 feature를 최근 관측에
   더 큰 비중을 주는 고정 exponential 가중치로 합쳐 센서당1개 feature로 만든다.
   이 가중치는 Stage A에서 학습하지 않는다.

센서s의 tubeletk feature를 `h[k,s]`, 실제 관측 timestamp 평균을 `u[k,s]`,
현재 query 시간을 `t`, 유효 여부를 `valid[k,s]`라 하면 다음과 같다.

```text
w[k,s] = valid[k,s] × exp((u[k,s] - t) / tau)
         / sum_j(valid[j,s] × exp((u[j,s] - t) / tau))
sensor_feature[s] = sum_k(w[k,s] × h[k,s])
```

`tau`는 T-Rex0.15초, Assembly0.20초다. 프레임 순번 대신 실제 timestamp를 사용한다.
누락·padding만으로 구성된 tubelet은 가중치에서 제외하고 유효 가중치를 다시 정규화한다.
관측이 전혀 없는 센서는0 feature와 false fusion mask를 사용한다.
0접촉 관측 자체는 누락으로 취급하지 않는다.

### 실제 tensor 흐름

아래 shape는 batch축을 생략했다. Tiny의 temporal tubelet 크기는2프레임이고,
출력 feature는768D, Fusion 입력은 학습 가능한 projection 이후512D다.

```text
T-Rex 기존:
  10프레임 × 10센서 → CNN [10,10,512] → flatten → [100,512]
T-Rex 이번:
  10프레임 → LingBot 5tubelets
  → 각 센서의96taxel mean [5,10,768]
  → 시간 가중평균 [10,768] → projection [10,512]

Assembly 기존:
  3프레임 × 8pad → MLP [3,8,512] → flatten → [24,512]
Assembly 이번:
  [t-2,t-1,t] → encoder 입력만 [t-2,t-2,t-1,t]로 prefix 복제
  → LingBot 2tubelets → 각 pad의15taxel mean [2,8,768]
  → 시간 가중평균 [8,768] → projection [8,512]
```

Assembly의 정상 window에서 첫 tubelet은 `[t-2,t-2]`, 두 번째는 `[t-1,t]`다.
복제 프레임은 새 관측으로 세지 않으므로 첫 timestamp는 원래`t-2`의 시간,
두 번째 timestamp는 원래`t-1,t`의 실제 시간 평균이다.
원본 history 범위와 Stage A reconstruction target(T-Rex10프레임, Assembly3프레임),
Stage B action horizon은 기존과 동일하게 유지했다. 단, Fusion이 시간별 특징을
직접 받던 구조에서 시간축이 압축된 센서 특징을 받는 구조로 바뀌었다.

LingBot 내부 attention은 이미 시간·taxel·센서 사이의 문맥을 섞는다.
따라서 이 시간 가중치는 출력 readout의 비중이며 원본 프레임의 직접 가중평균은 아니다.
최근 주소의 feature에도 과거 문맥이 포함될 수 있다. 시점별 sensor token에 적용되던
기존 시간 embedding은 이번에는 pooling에 사용한 가중 상대시간을 반영해 센서별로 더한다.

이 때문에 결과는 **encoder 교체 + frozen 사전학습 특징 + 시간 압축 + token 수 감소**의
결합 효과다. 동일한 token 수·시간 readout을 유지한 비교가 아니므로,
5.44%/4.19%의 관측 개선을 LingBot encoder만의 효과로 분리해 해석하지 않는다.

## 불확실성과 해석

| 데이터 | 개선된 test 단위 | LingBot−baseline 평균 차이 | Paired bootstrap95% 구간 |
|---|---:|---:|---|
| T-Rex |18/26|-0.003741915|[-0.007953746, 0.000081496]|
| Assembly |9/16|-0.002475061|[-0.006421401, 0.000530760]|

구간은 각 데이터셋의 고정된 test 단위를 paired resampling한100,000회 bootstrap,
RNG seed42 결과다. 음수 차이는 LingBot에 유리하다. **두 구간 모두0을 포함**한다.
이 구간은 학습된 seed0 모델과 해당 test 집합에 조건부이며, 학습 seed 변동성은 포함하지 않는다.
관측된 평균 오차는 개선됐지만 안정적·보편적인 성능 향상으로 단정할 수 없다.

이번 변경은 encoder뿐 아니라 pretrained/frozen 학습 조건과 시간 readout,
fusion tactile token 수(T-Rex100→10, Assembly24→8)도 바꾼다.
따라서 encoder architecture만의 효과를 분리한 ablation은 아니다.
결과는 offline action predictability이며 온라인 로봇 성공률을 의미하지 않는다.
T-Rex source-native 좌우 pixel mapping은 명시적 canonical 대응 가정이고,
PaXini 좌표는 static neutral-pose 근사 geometry다. PaXini signed Z는 사용자 지정
raw_clip30을 유지하며 upstream의 positive-Z 입력 분포와 차이가 있다.

## 실행 설정

공통: frozen LingBot-Touch-Tiny, 출력768D, tubelet별 센서 taxel mean,
실제 시간 기준 고정 exponential pooling, trainable768→512 projection, 센서당1token.
T-Rex는 native240→96points, history10/tubelet5, tau0.15초다.
Assembly는4손가락×(tip15+pulp15)를 한 번에 encode하여8pad를 공동 처리하고,
history3의 가장 오래된 관측을 prefix 복제한4프레임/tubelet2, tau0.20초다.
기존 visual cache/split/train-only normalization은 재사용했다.

## 결과 파일과 W&B

- [실행 전체 보고서](../../experiments/trlt/runs/lingbot_sequential_seed0_20261122T230902Z/final_report.md),
  [기계 판독 summary](../../experiments/trlt/runs/lingbot_sequential_seed0_20261122T230902Z/summary.json).
- [비교 검증·bootstrap 결과](../../experiments/trlt/runs/lingbot_sequential_seed0_20261122T230902Z/review/comparison.json),
  [비교 그림](../../experiments/trlt/runs/lingbot_sequential_seed0_20261122T230902Z/review/baseline_comparison.png).
- [T-Rex 상세 metrics](../../experiments/trlt/runs/lingbot_sequential_seed0_20261122T230902Z/trex/seed0/evaluation/metrics.json),
  [Assembly 상세 metrics](../../experiments/trlt/runs/lingbot_sequential_seed0_20261122T230902Z/assembly/seed0/evaluation/metrics.json).
- 각 `<dataset>/seed0/evaluation/`의 `predictions.jsonl.gz`와 `test_errors.png`,
  `<dataset>/seed0/final_selections.json`에 개별 예측/선택 근거가 있다.
- W&B T-Rex: [Stage A](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/uc6fj9e5),
  [Stage B](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/tajtxtjl).
- W&B Assembly: [Stage A](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/bckjdxwt),
  [Stage B](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/hk9mk360).

실행 산출물은 Git-ignored인 서버 로컬 파일이다. 완료된 cache나 queue를 다시 실행하지 않는다.
