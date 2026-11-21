# 3DTac-Dex Assembly: RLT / TRLT 비교 계획과 구현

이 문서는 [완료된 Assembly 변환](3dtacdex_assembly.md)을 사용하는 **별도 데이터셋 내 비교**다.
T-Rex와 같은 Stage A → frozen latent → Stage B → final test 절차를 사용한다.
2026-11-18 사용자 최종 검토·승인 후34단계 실험 큐를 시작했다.
실제 현재 단계는 아래 launch 기록의 status/progress/log 경로에서 확인한다.
별도 bounded GPU 검증 결과는 최신 검증 보고서에 기록되어 있다.

## 비교 대상

| 설정 | Stage A 입력 / 융합 | Stage A 목적함수 |
|---|---|---|
| `rlt_original` | 시각 특징만 기존 RLT encoder로 압축 | visual MSE |
| `trlt_self_only` | V(시각)와 U(촉각+state)를 합친 전체 self-attention | visual + tactile MSE |
| `trlt_cross_only` | V↔U 양방향 cross-attention, stream 내부 attention 없음 | 위와 같음 |
| `trlt_alternating` | 4 stages: 내부 self → cross → 내부 self → cross | 위와 같음 |

`self_only`는 **전체 concatenated sequence의 self-attention**이다. 모달리티끼리
차단하는 self-attention ablation이 아니다. 기존 T-Rex 구현과 동일하다.
모든 TRLT는 같은 파라미터 구조를 쓰고 fusion mask만 다르다. readout query는
각 stream을 읽지만 stream은 readout을 읽지 않는다. geometry는 끄고 encoder sensor ID는 켠다.

RLT는 기존 visual encoder/decoder를 그대로 사용한다. TRLT는 공유 pad MLP로
각 pad의 ordered 15×3 값을 flatten(45D) → Dense512 → GELU → Dense512로 인코딩한다.
가상의 이미지나 taxel 격자를 만들지 않는다. pad ID와 실제 상대 관측 시간 embedding을
더하고 현재 22D state와 visual age를 U에 넣는다. **2026-11-18 변경으로 visual/tactile/state
modality embedding은 모두 제거한다** (`use_modality_embedding=false`). camera/patch/sensor ID/시간
embedding과 decoder query는 유지한다. original RLT에는 본래 이 modality embedding이 없다.
현재 tactile decoder는 각 history/pad slot의 45개 `clip(raw/30,-1,1)` 값을 재구성한다.
아래 과거 mean/std 실험 결과는 실행 당시 config snapshot을 기준으로 해석한다.
visual decoder는 기존 구조를 공유한다. 양쪽 loss 가중치는 1이다.
Stage A에서 action label은 입력/재구성 대상이 아니다.

Stage B는 네 모델 모두 동일하게 `[frozen z, normalized current state, visual age]`를
입력받는 256×2 MLP다. representation/backbone은 고정하고 3×22 action을 예측한다.
RLT와 TRLT는 입력 정보/학습 목적도 다르므로 RLT 대비 차이를 attention만의 효과로
해석하지 않는다. 세 TRLT 간 비교가 fusion schedule 비교다.

## 고정 데이터 및 split

- 변환된 **전체 124 segments / 109 원본 trajectories / 22,069 frames**를 쓴다.
  `nouse`를 포함한 네 source version을 모두 유지한다.
- `source.episode_id` 단위로 묶고 source group/label별 약 70/15/15 비율로
  seed42 split을 한 번 저장한다. 같은 trajectory의 다른 segment가 split을 넘지 않는다.
- 실제 준비 결과:

| split | 원본 trajectories | LeRobot segments | frames / queries |
|---|---:|---:|---:|
| train | 77 | 89 | 14,819 |
| validation | 16 | 17 | 3,472 |
| test | 16 | 18 | 3,778 |

현재 state22와 action22의 mean/std만 **train 프레임으로** 계산한다.
Tactile8×15×3의 유일한 전처리는 `clip(raw/30,-1,1)`이며, encoder 입력과
reconstruction target에 동일하게 적용한다. 기준값30은 train raw 절댓값 분위수를
검토해 정한 고정값이다(99.9%:26, 99.95%:32). 각 split/batch/query에서
분위수를 다시 계산하지 않으며, 성분별 평균 제거·std 나눗셈은 하지 않는다.
Action은 element mask, tactile은 pad mask와 finite 여부를 반영한다. 0 접촉은 유효하다.
State/action std floor는1e-6이며 count/floored mask도 저장한다. 누락 hand command 10프레임은
원본 데이터에 남고 해당 16개 차원만 action loss/정규화/지표에서 제외한다.
LeRobot의 전체 데이터 통계와 기본 `train: 0:N` 표기는 실험 통계/split으로 쓰지 않는다.

`datasets/3dtacdex_assembly/protocol/`에 source inventory, split, queries,
normalization, protocol manifest를 저장한다. 각 JSON은 content hash를 검사한다.
로컬 source inventory는 변환 identity와 파일 size/mtime를 포함하며, 원본 ZIP SHA와
전체 변환 검증의 content hash 증거는 preparation report에 있다. 변환 파일 교체,
잘못된 split, 정규화 변경, 다른 시간 설정은 기존 protocol/cache 재사용 시 거부한다.
동일한 준비 명령은 기존 protocol을 재사용하고, 다른 protocol/부분 출력은 덮어쓰지 않는다.

## 시간 및 tensor 계약

T-Rex 30 Hz와 Assembly nominal5 Hz에 같은 frame 수를 적용하지 않는다.
이 계획은 native row 단위이며 interpolation이나 upsampling을 하지 않는다.

| 항목 | Assembly 설정 |
|---|---|
| query | 모든 recorded row, 약0.2초마다 |
| RGB anchor | segment 내 매3 rows, 약0.6초마다 두 외부 카메라 동시 갱신 |
| tactile history | query 포함 최근3 consecutive rows, 약0.4초 과거 범위 |
| state | query의 arm6 + hand16, 22D |
| action | query부터 현재/다음/다다음3 rows, 약0.4초 미래 범위, 22D 원본 command |
| gap 검사 | history/target 경로의 원본 frame counter 연속, 실제 인접 dt가0초 초과/0.3초 이하 |

window는 segment 경계를 넘지 않는다. 부족한 history/미래, 불연속 구간은 mask 처리한다.
실제 `source.elapsed_seconds`를 tactile 상대시간/visual age에 사용한다.
nominal video PTS는 해당 **원본 row의 RGB를 찾을 때만** 사용한다.
관측된 데이터의 인접 dt 중앙값은 약0.2초이나 nominal clock과 같다고 가정하지 않는다.
0.3초는 연속 row 간 허용 gap이며 visual context의 최대 age 제한이 아니다.
이 시간창은 T-Rex와 정확히 같은 초 단위 시간창을 재현하는 설정이 아니다.

- native tactile batch: `[B,3,8,15,1,3]`; 가운데 singleton 축은 공통 loss/mask API용이다.
- tactile mask/timestamp: `[B,3,8]`.
- frozen visual: `[B,512,2048]`, mask `[B,512]`, 각 카메라256 tokens 예상.
- state: `[B,22]`; action target와 **element mask**: `[B,3,22]`.
- camera order: `main`, `secondary`. 두 번째 카메라는 손목 카메라가 아니다.
  prefix 전용 API에 명시적 image keys를 전달하며 세 번째 카메라를 복제하지 않는다.
- 시각 특징은 동일한 local frozen pi0.5 checkpoint의 prefix-only image output이다.
  언어 token output은 제거하며 action expert와 discrete state 입력을 사용하지 않는다.
  실제 두 카메라의 prefix-only GPU 추출에서 `[512,2048]`을 확인했다. train64/validation4개
  query에 필요한24anchors만 별도 validation cache에 저장했으며 shared 전체 cache는 추출하지 않았다.

## 예산·선택·평가

초기 계획은 기존 완료 비교와 같이 **seed0 한 개**다. 설정 파일의 `training_seeds=[0]`은
계획 메타데이터이며 CLI가 seed loop를 자동 실행하지 않는다.

- Stage A: 네 모델 모두8,000 optimizer updates, epoch cap/early stopping 없음;
  AdamW1e-4, weight decay1e-4, clip1, effective batch64 / micro64 / accumulation1.
  epoch 끝과 마지막 step에 validation하고 각 모델 자신의 재구성 loss로 best를 고른다.
  서로 다른 모델의 재구성 loss 크기로 우열을 판단하지 않는다.
- train query 수 기준232 updates/epoch이며 마지막 partial batch를 유지한다.
  동일 update 수라도 T-Rex보다 데이터 반복 횟수가 많다.
- Stage A의 가중치 업데이트에는 train만 사용한다. validation은 gradient/optimizer 없이
  재구성 loss를 측정하고 checkpoint를 선택하는 용도다.8,000updates까지 계속 학습하며,
  이후 best checkpoint를 사용하므로 선택된 checkpoint step은8,000보다 작을 수 있다.
- 선택된 표현을 고정해 train/validation latent를 export한다.
- Stage B: AdamW1e-3, weight decay1e-4, batch256, hidden256×2,
  max100epochs/patience10. mask-aware validation action MSE로 best를 고른다.
- 모든 모델의 Stage A/B best hash를 먼저 봉인한다. CLI는 봉인 없이 Assembly
  test feature extraction / test latent export / evaluation 실행을 거부한다.
  봉인 후 checkpoint가 바뀌어도 거부한다. 실제 선택 step은 봉인에 기록한다.
- 최종 primary는 각 **원본 trajectory**의 valid query×horizon×joint normalized MSE를
  계산한 뒤16개 test trajectory에 동등 평균한다. 여러 segment도 같은 trajectory로 합친다.
  공통 지표 key는 `primary_normalized_episode_macro_mse`이며 여기서 episode는 원본 trajectory다.
- micro MSE, raw MSE, arm/hand/joint/offset별 MSE도 보고한다. raw command 단위가
  독립적으로 확인되지 않았으므로 radians/Newton/실제 제어 성능으로 해석하지 않는다.
- 동일 test trajectory/valid-element count를 네 모델에서 확인하고 원본 trajectory 단위
  paired bootstrap95% CI를 보고한다. source group별 결과는 object 항목에 들어가며
  실제 physical object identity를 뜻하지 않는다. seed0만으로 seed 변동성을 평가하지 않는다.

## W&B 설정 및 기록 주기 (2026-11-18)

네 Assembly config의 `tracking`은 online, project=`trlt`,
entity=`rnsrns118-seoul-national-university`, `log_every_steps=10`이다.
본 실험의8개 Stage A/B run은 plan의 run ID로 group하고,
`assembly/<model>/seed0/stage_a` / `stage_b` 이름을 사용한다. preflight는 disabled다.

- train loss/시간/진행 지표는 optimizer step10,20,30,…의 값만 W&B에 기록한다.
  validation은 매 평가 시 기록해 best 선택 근거를 남긴다.10-step 평균을 새로 계산하지 않는다.
- 로컬 `train.jsonl`은 모든 optimizer step을 보존한다. validation 빈도와 학습 계산은 바뀌지 않는다.
- W&B config에는 dataset kind=`3dtacdex_assembly`, camera2/sensor8의 순서,
  fps5/history3/horizon3, state/action22D, tactile `[3,8,15,1,3]`, visual `[512,2048]`,
  modality-disabled model config,8k/batch64/micro64/accumulation1을 기록한다.
  Stage B에도 frozen representation의 원래 model config를 별도 보존한다.
- Stage A/B metrics는 기존 `stage_a/*`, `stage_b/*`, `progress/*`, `timing/*` 그룹과
  `optimizer_step` 축을 사용한다. sensor error8개/history error3개를 펼치며 mask/count는 로컬 진단에 남긴다.
- CLI `--wandb-log-every 10`으로 주기를 지정하고, `--wandb offline`으로 외부 업로드 없이
  같은 기록 형식을 검증할 수 있다. CLI 값이 config보다 우선한다.
- tracking 설정은 optimizer/checkpoint identity 밖의 runtime metadata다.
  과거 T-Rex 명령은 별도 지정이 없으면 기존 disabled/매step 기본값을 유지한다.

## train/test만 사용하는 대안과 이번 결정 (2026-11-18)

사용자 선택에 따라 **기존 train/validation/test를 유지**한다. split/정규화는 다시 만들지 않는다.
Stage A와 B 모두 train으로만 가중치를 학습하고, validation으로 checkpoint를 선택하며,
test는 네 모델의 선택을 봉인한 뒤 최종 평가에만 사용한다.

validation을 생략한 train/test 설계도 가능하지만, 두 stage의 학습량과 하이퍼파라미터를
test를 보기 전에 고정하고 정해진 마지막 checkpoint를 사용해야 한다. Stage A8,000steps만
고정하는 것으로는 현재 Stage B의 validation 기반 patience10/best 선택을 대체할 수 없다.
Stage B에서 test를 gradient 학습, early stopping, checkpoint 선택, 하이퍼파라미터/아키텍처
수정에 사용하면 그 집합은 최종 test 역할을 하지 못한다. Stage B에서 train으로 학습하고
test로 **최종 평가**한다는 의미라면 적절하다. 정규화도 모든 stage에서 train에만 fit한다.

## 실행 명령과 저장 위치

모든 shell에서 기존 `sehwa`를 먼저 활성화한다. 아래는 **명시적 단계 예시**다.
GPU 명령 실행은 별도 실행 결정 후, 기존 작업/GPU 점유를 확인하고 진행한다.

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa

# CPU 준비 / 실제 train+validation reader 검증 (이미 완료된 출력은 재사용)
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/prepare_tacdex_experiment.py prepare
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/verify_tacdex.py

# GPU/backbone 초기화 없이 cache 크기 계획
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/cache_trex_features.py \
  --config configs/trlt/3dtacdex_assembly/trlt_alternating.json --split train --plan-only

# 새로운 이름으로 config snapshot과 단계별 argv JSON을 작성. 어떤 job도 실행하지 않음.
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/plan_tacdex_experiment.py \
  plan --run-id assembly_8k_b64_no_modality_seed0_example --device 0
```

생성된 `plan.json`에는34개 명시적 명령이 순서대로 있다:
CPU protocol 준비 → shared train/validation visual cache → 네 모델별 GPU preflight → 네 모델별
Stage A/export train/export validation/Stage B → 선택 봉인 → shared test cache →
네 모델별 export test/evaluate → summarize. 명령은 argv 배열이다.
계획 생성기는 실행기/daemon이 아니며 GPU job, tmux, W&B upload를 시작하지 않는다.
source/config SHA256도 기록한다. 실제 실행 담당자는 실행 전 이 hash를 확인하고
변경/실패/GPU 점유가 있으면 중단해야 한다. 자동 resume/retry는 없다.
GPU preflight는 각 모델에서 full-shape optimizer1update와 전체 validation을 수행한다.
`--max-steps 1`과 별도 preflight checkpoint/log 경로를 사용하므로 본 학습과 섞이지 않는다.
생성된34단계 계획은 사용자 승인으로2026-11-18에 실행을 시작했다. 별도 bounded GPU 검증의 실제 결과는 최신 보고서를 참조한다.
본 실험 Stage A/B의 W&B는 config에서 online으로 설정하며,
`rnsrns118-seoul-national-university/trlt`에 기록한다. GPU preflight는 명시적으로 W&B disabled,
GPU 검증 run은 offline으로 동작한다. 별도 검증용 run 하나에 최소 설정과 실제 probe 지표2개를
온라인 기록하고 API 재조회까지 확인했다. 실제 본 실험의 단계별 시작/완료는 run status/events에 기록한다.

기본 설정은 `configs/trlt/3dtacdex_assembly/{rlt_original,trlt_self_only,trlt_cross_only,trlt_alternating}.json`.
독립 명령은 `scripts/tacdex_offline.py`의 `train-representation`, `export-latents`,
`train-probe`, `evaluate`이며 반드시 `--config`를 전달한다. 공통 `trex_offline.py`
실행기를 사용한다. dry-run은 `--dry-run`으로 설정/명령만 검사하며 데이터/GPU를 열지 않는다.

선택 봉인은 CPU에서 실행한다:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python scripts/plan_tacdex_experiment.py \
  seal --checkpoint-root checkpoints/trlt/3dtacdex_assembly/REVIEWED_RUN_ID \
  --output experiments/trlt/3dtacdex_assembly/runs/REVIEWED_RUN_ID/final_selections.json
```

생성 계획의 경로:

- data/protocol/shared features: `datasets/3dtacdex_assembly/{lerobot_all,protocol,visual_cache}/`.
- model/seed latents: `datasets/3dtacdex_assembly/latents/<run_id>/`.
- checkpoints: `checkpoints/trlt/3dtacdex_assembly/<run_id>/`.
- logs/metrics/predictions/selections: `experiments/trlt/3dtacdex_assembly/runs/<run_id>/`.
- prepared plans/config snapshots: `experiments/trlt/3dtacdex_assembly/plans/<run_id>/`.

segment ID, 원본 trajectory ID, source group/label을 latent/예측에 보존한다.
실제 검증 결과와 미실행 항목은
[CPU 구현 검증 보고서](../../experiments/trlt/3dtacdex_assembly/implementation/report.md)를 참조한다.
기존 T-Rex 기본값/체크포인트 identity와 완료된 visual cache는 유지한다.


2026-11-18 갱신 계획: `experiments/trlt/3dtacdex_assembly/plans/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/plan.json`.
이전15k/batch32/modality-enabled 계획은 미실행 역사로 보존하고 새 실행에 사용하지 않는다.
최신 micro64/W&B10 변경과 실제 bounded GPU 검증:
`experiments/trlt/3dtacdex_assembly/revisions/20261118_full_b64_wandb10/report.md`.
이전 micro4/accumulation16의 CPU 증거는 `20261118_no_modality_8k_b64/`에 역사로 남긴다.
짧은 GPU 검증은 장기 처리량/8,000-step 학습 품질을 측정하는 benchmark가 아니다.

GPU0 RTX PRO6000에서 네 full-size 모델의 batch64/micro64/accumulation1 optimizer 각1update와
validation4queries가 통과했다. 추가 frozen latent/probe12updates 검증에서 실제 train W&B 기록은
step10에만 남았다. 별도 [온라인 로깅 검증 run](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/9c6s4a63)은
본 실험 결과가 아닌 기존 bounded 검증 지표의 수신 확인용이다.


## 승인된 본 실험 실행 시작: 2026-11-18

승인된 계획과 현재 source/config/input hash가 일치함을 확인한 후,
2026-11-18T01:23:03Z에34단계 큐를 시작했다. GPU0는 시작 전 idle이었다.
초기 관측은 protocol 재사용 완료 후 train 시각 특징 추출(stage2/34) 시작이다.
optimizer step/이후 단계 완료를 이 초기 관측으로 추정하지 않는다.

Run ID: `assembly_8k_fullb64_wandb10_seed0_20261118_prepared`.
[실행 디렉터리](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/),
[status](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/status.json),
[progress](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/progress.json).
계획의 `_prepared` run ID는 그대로 보존했으며, 원래 plan의 준비 상태 표기 대신 실제 run status를 확인한다.

Tmux socket=`trlt-runs`, session=`trlt-assembly-8k-seed0`, windows=`train`,`reports`.
별도 승인된34단계를 순차 실행하고 source/config/input 변경, 점유된 GPU, 단계 실패 시 중단한다.
자동 retry/resume는 없으며, 이미 시작한 큐를 중복 실행하지 않는다.
8개 main W&B run은 각 Stage A/B 시작 시 생성된다. 현재 단계를 확인하지 않고
feature 추출/학습/최종 평가를 별도로 다시 시작하지 않는다.


## 사용자 요청에 따른 feature 추출 최적화 및 재시작: 2026-11-18

사용자가 기존 실행 중단과 방금 생성된 cache 삭제를 명시적으로 요청했다.
이전 큐는 `cache_train` 단계에서 중단했으며, 해당 shared cache78anchors/327361813bytes를
삭제했다. 원본 변환, split/정규화, 기존 bounded validation cache와 완료된 T-Rex 결과는 보존했다.

새 feature 설정은 **입력/vision batch32, Gemma prefix microbatch1, CPU workers2,
prefetch factor2**다. 각 worker가 episode 단위로 두 영상을 한 번씩 열어 순차 디코딩한다.
원래 uint8→float32 scaling/GPU resize와 정확한 source row·timestamp를 유지한다.
마지막 불완전 배치는 마지막 입력으로 padding하고 실제 anchor 수만 저장한다.
학습의 batch64/micro64/accumulation1 설정과는 별도다.

그대로 batched Gemma를 호출하면 batch1 대비 relative L2약2.85% 차이가 나 고정 기준을
통과하지 못했다. 진단상 resize/vision/prefix 입력은 동일했고 Gemma 이후 차이가 발생했다.
따라서 image encoder는 batch 처리하되 Gemma는 JIT loop 안에서 batch1 산술을 유지한다.
이 hybrid의32-anchor 비교 relative L2는9.39e-6(약0.00094%), cosine최소0.9999999996이다.
기존 저장 batch1 cache16anchors도 각 relative L2<1e-4로 통과했다. 정확한 bit equality를
주장하지 않으며, batch/reader/prefix 방식은 새 cache identity에 기록해 구방식과 혼용하지 않는다.

Train-only 측정32anchors: CPU repeated-seek11.47초→sequential2.57초(약4.46배).
메모리에 준비된 입력의 GPU forward는 batch1약5.21→hybrid batch32약7.79anchors/s.
이 수치는 전체 추출 완료 처리량이 아니며 실제 data wait/write 포함 결과는 extraction report에 남긴다.
실제64anchor의worker2→GPU→cache 저장, tail padding, mask/key/timestamp/수치 기준이 통과했다.
CPU11tests passed(선택 회귀7deselected). 증거:
`experiments/trlt/3dtacdex_assembly/feature_optimization_20261118/report.md`.

새 run ID는 `assembly_8k_fullb64_batched_seed0_20261118`, tmux session은
`trlt-assembly-batched-seed0`(socket`trlt-runs`)이다. shared cache가 비어 있음을 확인하고
새 source/config snapshot으로34단계 큐를 시작했다. 기존 승인한 학습·최종 평가 조건은 그대로다.
이전 run은 `STOPPED_BY_USER`/`superseded.json`으로 보존한다. 현재 실행 상태:
[status](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/status.json),
[progress](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/progress.json).


재추출 완료 확인 (2026-11-18 02:10 UTC): train **4,973 anchors**, validation
**1,162 anchors**, 합계 **6,135 anchors**가 새 공유 캐시에 저장됐다. 실제 추출·저장
구간은 각각 736.35초 / 177.17초 (6.754 / 6.559 anchors/s)였다. Backbone 로딩은
각 split 보고서의 별도 총 시간 필드에 기록된다. 전체 키가 고정 split과 정확히
일치하고 test 키는 0개임을 확인했다. 캐시 해시는
`abb20dc6ecb8e7979aea11e8d6330837c9e5fc624aff3cc09b34a005765b4c1e`다.
[완료 검증](../../experiments/trlt/3dtacdex_assembly/feature_optimization_20261118/production_cache_validation.json),
[실행 보고서](../../experiments/trlt/3dtacdex_assembly/feature_optimization_20261118/report.md).
큐는 이 확인 시점에 stage 4/34 `rlt_original_gpu_preflight`로 진행했으며,
본 학습 및 최종 평가는 후속 단계다. 현재 진행은 새 run의 실제 status/log를 확인한다.


## 완료된 Assembly 실험 결과 (2026-11-18)

실험 `assembly_8k_fullb64_batched_seed0_20261118`는 **2026-11-18 16:11:22 UTC**
(한국시간 11월 19일 01:11)에 평가와 집계까지 **34/34 단계 모두 완료**했다.
네 모델 모두 Stage A를 실제 **8,000 optimizer updates**까지 학습했다.
설정은 batch 64 / microbatch 64 / accumulation 1, modality embedding 제거,
seed0이다. Stage B는 representation을 고정한 action MLP이며 validation으로
checkpoint를 선택했다. 원본 trajectory 기준 train/validation/test는 77/16/16이다.

주 지표는 train 통계로 정규화한 action MSE를 각 **원본 test trajectory 안에서**
평균한 뒤, **16개 trajectory를 동일 가중치로 평균**한 값이다. 낮을수록 좋다.
LeRobot segment 18개를 독립 표본 18개로 세지 않는다.

| 모델 | Test normalized trajectory-macro MSE ↓ | RLT 대비 오차 변화 | 선택된 Stage A step | 선택된 Stage B epoch |
|---|---:|---:|---:|---:|
| Original RLT | 0.059761 | 기준 | 8,000 | 17 |
| TRLT self attention | 0.092136 | +54.17% | 7,888 | 13 |
| TRLT cross attention | 0.090131 | +50.82% | 7,656 | 12 |
| TRLT alternating | 0.085492 | +43.06% | 8,000 | 13 |

**이번 조건에서는 Original RLT가 가장 낮은 오차를 보였고, TRLT 세 종류 중에서는
alternating이 가장 좋았다.** 이 결과에서는 tactile fusion의 action 예측 성능
이득이 관찰되지 않았다. 다만 구조, native tactile 인코딩, reconstruction 목적과
가중치 등 여러 요인이 함께 다르므로 원인을 tactile 정보 자체로 특정할 수 없다.
T-Rex의 다른 데이터/시간 계약/학습량 결과와 수치를 직접 합쳐 비교하지 않는다.

RLT와의 차이(TRLT − RLT)를 원본 trajectory 단위로 paired bootstrap한
95% 구간은 다음과 같다. 세 구간 모두 양수로, 이 seed와 test 표본에서 TRLT의
오차가 더 높은 방향이다.

| 모델 | MSE 차이 | Paired trajectory-bootstrap 95% 구간 |
|---|---:|---:|
| TRLT self attention | +0.032375 | [0.017205, 0.050393] |
| TRLT cross attention | +0.030370 | [0.015052, 0.048020] |
| TRLT alternating | +0.025730 | [0.012252, 0.041992] |

이 구간은 **학습 seed 간 변동성을 포함하지 않는다**. seed0 한 번의 결과이며,
실제 로봇 제어 성공률이나 다른 seed에서의 순위를 입증하지 않는다. 일반 집계기의
`missing_expected_seeds: [1,2]` 표시는 요청된 seed0 실험이 미완료라는 뜻이 아니다.
Stage A의 RLT visual-only total과 TRLT visual+tactile total은 목적 항이 달라
총 reconstruction loss만으로 모델 간 우열을 비교하지 않는다.

실행/평가 대조 결과: 네 모델의 원본 test trajectory ID와 각 trajectory의 valid
요소 수가 같고, 모델당 총 **248,112개 valid action 요소**를 사용했다. 동일한
split/normalization/query/feature cache 해시를 확인했고, 원본 trajectory별 MSE로
주 지표를 다시 집계해 일치함을 확인했다. 모든 checkpoint 선택은 16:04:48 UTC에
고정됐으며 test feature 추출은 그 뒤 16:04:50 UTC에 시작했다.

근거: [최종 실행 보고서](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/final_report.md),
[전체 지표와 paired bootstrap](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/summary.json),
[비교 CSV](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/summary.csv),
[평가 대상 일치 검사](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/comparison_validation.json),
[이번 결과 대조 기록](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/results_review.json),
[고정된 checkpoint 선택](../../experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_batched_seed0_20261118/final_selections.json).
각 모델의 `<model>/seed0/evaluation/metrics.json`과 `predictions.jsonl.gz`에
세부 지표와 예측이 있다. 이 결과 정리에서는 학습·추출·평가를 재실행하지 않았다.


## Self attention fusion 2층 후속 실험 (2026-11-19)

사용자 요청으로 `trlt_self_only`를 fusion **2 layers**로 추가 실행한다.
[2층 config](../../configs/trlt/3dtacdex_assembly/depth/trlt_self_only_2layer.json)는
직전 실제 4층 실행 snapshot에서 `model.num_stages: 4 → 2`만 변경했다.
Tactile history 3 frames, modality embedding 제거, seed0, Stage A 8000 updates,
batch64/micro64/accumulation1, Stage B 기본값, W&B 10-step 기록을 유지한다.
기존 split/정규화/7402-anchor visual cache를 재사용하며 feature 재추출은 없다.

새 run: `assembly_self2_8k_fullb64_seed0_20261119T002037Z`.
2026-11-19 00:22:58 UTC에 GPU0에서 독립 9단계 큐를 시작했다.
1-update/full-validation GPU preflight → Stage A → train/validation latent export
→ Stage B → checkpoint seal → test latent export → test 평가 → 비교 보고서 순서다.
현재 완료 여부는 [실제 status](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/status.json)와
[progress](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/progress.json)를 확인한다.
체크포인트/latent/log는 새 run ID로 격리하고 기존 완료 결과는 보존한다.

한 모델만 새로 학습하므로 `plan_tacdex_experiment.py seal`에
`--base-selection-seal <기존 final_selections.json> --replace-models trlt_self_only`
옵션을 추가했다. 기존 나머지 모델의 검증된 선택 기록을 그대로 보존하고 새 모델의
Stage A/B best만 교체·검증해 별도 seal을 만든다. 기존 네 모델의 선택 확정 전 test를
차단하는 기본 동작은 유지한다. CPU 회귀 2 tests와 7 CLI dry runs가 통과했고,
기존 checkpoint 해시 및 격리된 과거 지표 replay로 보고서 생성도 검증했다.
근거: `experiments/trlt/3dtacdex_assembly/depth_2layer_validation/` 및 새 run의
`approved_plan.json`, `commands.json`, `launch_manifest.json`, `source_snapshot/`.

최종 보고서는 기존 Original RLT, self-only 4층과 새 self-only 2층을 구분해
비교하고 2층−4층 paired trajectory-bootstrap도 기록한다. 기존 test 결과를 본 뒤
정한 후속 실험임을 명시하며, 이번 실행의 checkpoint 선택은 validation만 사용한다.
단일 seed와 offline action prediction의 해석 한계도 유지한다.


## Self attention 2층 후속 실험 완료 결과 (2026-11-19)

추가 실험은 **2026-11-19 03:26:42 UTC / 12:26 KST**에 test 평가와 비교 보고서까지
**9/9 단계 모두 완료**했다. 실제 Stage A는 8,000 optimizer updates이며, validation으로
선택한 best는 **7,888 step**이다. Stage B best는 **epoch 12**, 마지막은 epoch 22
(patience 10 종료)다. Seed0 / batch64 / micro64 / accumulation1 / tactile history3 /
modality embedding 없음 등의 설정은 기존과 같다. 모델 설정에서 바뀐 값이
`num_stages: 4 → 2` 하나뿐임을 실행 provenance로 대조했다.

주 지표는 동일한 **원본 test trajectory 16개**의 normalized action MSE를 각
trajectory 안에서 평균하고, trajectory 간 동일 가중치로 평균한 값이다. 낮을수록 좋다.

| 모델 | Fusion layers | Test normalized trajectory-macro MSE ↓ | RLT 대비 오차 변화 |
|---|---:|---:|---:|
| Original RLT | — | 0.059761 | 기준 |
| TRLT self attention | 2 | 0.081194 | +35.86% |
| TRLT alternating | 4 | 0.085492 | +43.06% |
| TRLT cross attention | 4 | 0.090131 | +50.82% |
| TRLT self attention | 4 | 0.092136 | +54.17% |

**Self attention의 fusion을 4층에서 2층으로 줄이면 이번 seed0의 test MSE가
0.092136 → 0.081194로 11.88% 감소**했다. 실행된 TRLT 설정 중 가장 낮지만,
Original RLT 0.059761보다는 여전히 35.86% 높다.

2층−4층 MSE 차이는 **−0.010942**, paired original-trajectory bootstrap 95% 구간은
**[−0.016479, −0.005356]**이다. 이 seed와 test 표본에서는 개선 방향이다.
2층−RLT 차이의 95% 구간은 **[+0.009639, +0.034084]**로 RLT보다 높은 오차다.
이 구간들은 training seed 간 변동성을 포함하지 않는다.

Stage A best total reconstruction validation loss는 4층 1.496196 / 2층 1.497788로
비슷한 반면, Stage B best validation MSE는 0.112642 → 0.090179로 낮아졌다.
재구성 total loss 하나만으로 action probe의 성능 순위를 판단할 수 없음을 보여준다.
단일 실험만으로 개선 원인을 과적합 완화 등으로 확정하지 않는다.

검증: 기존 모델과 원본 test trajectory ID 및 trajectory별 valid 요소 수가 같고,
모델당 총 248,112 valid action 요소를 사용했다. Split/query/normalization/feature
cache provenance가 같고, trajectory별 MSE로 주 지표를 재집계해 일치함을 확인했다.
새 Stage A/B 선택을 고정한 뒤 새 test latent export와 평가를 진행했다. 기존
Original RLT / self4 / cross4 / alternating4 결과는 재학습·재평가하지 않고 재사용했다.

이는 이전 test 결과를 본 뒤 정한 **후속 탐색 실험**이다. 동일 조건의 깊이 비교지만,
여러 seed에서의 재현성이나 실제 로봇 제어 성공률을 입증하지 않는다.

근거: [최종 보고서](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/final_report.md), [집계 JSON](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/summary.json),
[비교 CSV](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/summary.csv), [실행 결과 대조](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/results_review.json),
[원본 2층 test 지표](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/trlt_self_only/seed0/evaluation/metrics.json),
[checkpoint 선택](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_8k_fullb64_seed0_20261119T002037Z/final_selections.json).
학습 곡선: [Stage A W&B](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/tjyjid6l),
[Stage B W&B](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/46v2tn4l).
이 결과 정리에서는 학습·feature 추출·평가를 새로 실행하지 않았다.


## Self attention 2층 + modality embedding 후속 실험 (2026-11-19)

사용자 요청으로 직전 self-only2 설정에 **modality embedding을 다시 활성화**한다.
새 [config](../../configs/trlt/3dtacdex_assembly/modality/trlt_self_only_2layer_modality.json)는
직전 실제 2층 run의 config에서 `model.use_modality_embedding: false → true`만
변경했다. Visual / tactile / state용 학습 embedding 3개, 각 512차원이다.
Sensor ID, camera/patch, 상대 시간 embedding과 readout query는 기존대로 유지한다.

유지 설정: fusion2, seed0, tactile3frames, Stage A8000updates,
batch64/micro64/accumulation1, Stage B 기본값 및 validation checkpoint 선택,
W&B online 10-step 기록. 기존 split/train-only 정규화/7402-anchor feature cache를
재사용하고 backbone feature를 다시 추출하지 않는다. 기존 완료 실험도 보존한다.

Run `assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z`가
**2026-11-19T03:50:26.710345+00:00**에 GPU0 큐로 시작됐다.
GPU 1-update/full-validation preflight → Stage A → train/validation latent export
→ Stage B → 새 선택 seal → test latent export → 평가 → 비교 보고서의 9단계다.
실제 현재 단계/완료는 [status](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/status.json),
[progress](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/progress.json), stage logs 및 checkpoint에서 확인한다.
Tmux socket `trlt-runs`, session `trlt-assembly-self2-modality-seed0`.

CPU 검증: 활성화 시 기존 config 직렬화 호환성 회귀 1 test 통과(무관한 8개 제외),
새 CLI 7개 dry-run 통과, 기존 checkpoint 해시 및 격리된 과거 지표 replay 통과.
근거는 `experiments/trlt/3dtacdex_assembly/modality_2layer_validation/`의
`pytest.xml`, `cpu_preflight.json`, `dry_run_*.json`, `report_replay.log`다.
과거 지표 replay는 새 모델의 평가 결과가 아니다.

기존 serializer는 기본값 true를 생략할 수 있으므로, 실행 config의 명시적 true와
실제 checkpoint의 `encoder/modality_embed` **[3,512]** 존재·유한값도 확인한다.
큐의 modality 검사는 이 기본값 생략 규칙을 반영한다. 원래 four-model seal
계약은 유지하고 직전 2층 modality-off seal에서 self-only 선택만 새 best로 교체한다.

최종 보고서는 Original RLT / self4 modality-off / self2 modality-off / 새 self2
modality-on을 구분하고, **self2 on−off paired trajectory-bootstrap**도 기록한다.
이전 test 결과를 확인한 뒤 요청한 후속 탐색 실험이라는 점을 명시한다. 새 실험의
checkpoint 선택은 validation만 사용하며 test는 선택을 고정한 뒤 평가한다.


## Self attention 2층 modality embedding ON/OFF 완료 결과 (2026-11-19)

Modality-on 후속 실험은 **2026-11-19 07:06:42 UTC / 16:06 KST**에 최종 test
평가와 집계까지 **9/9 단계 모두 완료**했다. Stage A는 실제 **8,000 optimizer
updates**를 완료했고 validation best는 **7,192 step**, Stage B best는
**epoch 13**이다(마지막 epoch 23, patience 10). 직전 modality-off2와 비교해
model 설정은 `use_modality_embedding:false→true`만 바뀌었다. Fusion2, seed0,
tactile3frames, batch64/micro64/accumulation1 및 나머지 설정은 동일하다.

동일한 원본 test trajectory 16개에서 normalized action MSE를 trajectory별로
평균한 뒤 동일 가중치로 집계한 주 지표다. 낮을수록 좋다.

| 모델 | Test normalized trajectory-macro MSE ↓ | RLT 대비 오차 변화 |
|---|---:|---:|
| Original RLT | 0.059761 | 기준 |
| TRLT self2 / modality OFF | 0.081194 | +35.86% |
| TRLT self2 / modality ON | 0.080930 | +35.42% |

**Modality ON의 오차는 OFF보다 0.33% 낮았지만, 이번 결과로 명확한 개선 효과를
확인하기 어렵다.** ON−OFF MSE 차이는 **−0.000264**이고, 원본 trajectory 단위
paired bootstrap 95% 구간은 **[−0.005464, +0.005207]**로 0을 포함한다.
이 구간은 두 설정의 동등성을 입증하는 검정이 아니며, 단일 seed 결과에서 계산했다.
Original RLT보다 ON의 오차가 여전히 35.42% 높다. ON−RLT 차이의 95% 구간은
[+0.011010, +0.032884]로 이 seed/test 표본에서 RLT가 낮은 오차를 보인다.

검증: 선택된 best checkpoint에 `encoder/modality_embed` [3,512] 파라미터가
실제로 존재하고 유한함을 CPU에서 확인했다. 켜짐은 기존 직렬화의 기본값이므로
model-config dict에 true 항목이 생략될 수 있다. ModelConfig로 복원한 on/off 설정을
대조해 해당 flag 외의 값이 같음을 확인했다. 각 모델의 원본 test trajectory ID,
trajectory별 valid 요소 수 및 총 248,112 valid action 요소가 동일하고, split/query/
normalization/feature cache provenance도 같다. 원본 trajectory별 지표로 주 MSE를
재집계해 일치함을 확인했다. 새 checkpoint 선택을 고정한 뒤 새 test export/평가를
진행했으며, 기존 RLT/self4/self2-off 결과는 재학습·재평가하지 않고 재사용했다.

단일 seed이며 이전 test 결과를 본 뒤 정한 후속 탐색 실험이다. Trajectory
bootstrap은 학습 seed 간 변동성을 포함하지 않으며 실제 제어 성공률을 뜻하지 않는다.

근거: [최종 보고서](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/final_report.md), [집계/신뢰구간](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/summary.json),
[CSV](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/summary.csv), [결과 대조](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/results_review.json),
[원본 test 지표](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/trlt_self_only/seed0/evaluation/metrics.json),
[고정된 checkpoint 선택](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_modality_8k_fullb64_seed0_20261119T034907Z/final_selections.json).
[Stage A W&B](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/1qqlmlzz),
[Stage B W&B](https://wandb.ai/rnsrns118-seoul-national-university/trlt/runs/fn1lncp3).
이 결과 정리에서는 학습·추출·평가를 새로 실행하지 않았다.

## Self-only 2-layer raw clipping follow-up (2026-11-19)

User selected `clip(raw / 30, -1, 1)` for both Assembly tactile encoder input
and reconstruction target, with `lambda_tactile=1`. Config:
`configs/trlt/3dtacdex_assembly/normalization/trlt_self_only_2layer_rawclip30.json`.
Relative to the executed modality-off self2 experiment, this changes only tactile
preprocessing. Seed0, two fusion layers, tactile history3, Stage A8000updates,
batch64/micro64, Stage B defaults and online W&B10 remain unchanged. No episode
or taxel is removed. State/action train statistics, fixed splits and the completed
visual cache are reused; tactile no longer uses the saved coordinate mean/std.

At launch, optional top-level `tactile_preprocessing={"mode":"raw_clip","scale":30.0}`
was introduced beside the then-existing standardization default. The 2026-11-20
policy below supersedes that default: raw clipping is now the only supported path.
It is recorded separately in dataset, checkpoint and latent provenance. Mixing
legacy/raw-clip representations, caches or different scales is rejected both ways. Invalid raw pads remain masked even if clipping could
turn infinity into a finite bound. Decoder target is the transformed batch tactile.
A bound on targets does not constrain the linear decoder's predictions to [-1,1].

This normalization also changes reconstruction-loss scale; keeping lambda1 does
not preserve the old visual/tactile balance. Compare downstream action metrics,
and do not interpret a smaller total reconstruction loss as improvement. This is
an exploratory follow-up after prior test inspection, with one seed only.

CPU evidence: `experiments/trlt/3dtacdex_assembly/rawclip30_validation/`.
Twenty regression cases and Ruff passed;10real train/validation queries match
raw/30 exactly,8queries match synchronous/spawned-worker loading exactly, and
seven CLI dry runs passed. Historical report replay is explicitly diagnostic,
not a new experiment result.

Prepared nine-stage queue:
`experiments/trlt/3dtacdex_assembly/runs/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/`.
Stages: one-update GPU preflight with validation; fresh Stage A; train/validation
latent export; Stage B; seal new validation selections using the completed base
seal; test export/evaluation; comparison against historical RLT and standardized
self2. Consult actual status/events/checkpoints before stating current progress.
No cache re-extraction, automatic retry/resume, extra seed or baseline retraining.

The rawclip30 queue actually launched at2026-11-19T10:51:10Z (19:51KST),
GPU0, tmux socket `trlt-runs`, session `trlt-assembly-self2-rawclip30-seed0`.
Active index: `experiments/trlt/3dtacdex_assembly/rawclip_active_run.json`
(also Assembly/global active indices). First preflight model initialization
was observed; later optimizer/preflight/benchmark completion must be read from
actual logs and checkpoints rather than inferred from this launch record.


## Rawclip30 완료 결과 및 tactile 전처리 기본 정책 (2026-11-20 확인)

실험 `assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z`는
**2026-11-19 23:02:41 KST (14:02:41 UTC)**에 평가·보고까지 **9/9단계 완료**했다.
실제 Stage A는8000updates까지 학습했고 best도8000이다. Stage B는 epoch35까지
진행한 뒤 patience10으로 종료했고 epoch25 checkpoint가 선택됐다. 설정은
modality-off self-only2layers,seed0,batch64/micro64/accumulation1,lambda_tactile1이며,
같은 self2 mean/std 실행 대비 변경한 실험 설정은 tactile 입력·재구성 target 전처리뿐이다.
Visual cache와 split, state/action 통계는 동일하게 유지했다. 기존 결과는 재학습하지 않았다.

아래는 지금까지 완료한 Assembly 설정의 통합표다. 새 기본 config 파일은 clipping으로
변경됐으므로, 과거 결과 재현에는 해당 run의 실제 config/source snapshot을 사용한다.
주 지표는 동일한16원본 test trajectory별 normalized action MSE를 동일 가중치로 평균한다.

| 설정 | Tactile 전처리 | Fusion layers | Test 주 MSE ↓ | RLT 대비 오차 변화 |
|---|---|---:|---:|---:|
| RLT | tactile 미사용 | 기존 RLT2 | 0.059761459 | 기준 |
| Self-only | mean/std (과거) | 4 | 0.092136257 | +54.17% |
| Cross-only | mean/std (과거) | 4 | 0.090131 | +50.82% |
| Alternating | mean/std (과거) | 4 | 0.085492 | +43.06% |
| Self-only | mean/std (과거) | 2 | 0.081193840 | +35.86% |
| Self-only, modality-on | mean/std (과거) | 2 | 0.080929551 | +35.42% |
| **Self-only, rawclip30** | **clip(raw/30,-1,1)** | **2** | **0.059127821** | **−1.06%** |
| Self-only, rawclip30 | clip(raw/30,-1,1) | 4 | 0.061742904 | +3.32% |

Rawclip30은 같은 modality-off self2 mean/std보다 **27.18% 낮은 MSE**를 보였다.
Paired 원본 trajectory-bootstrap95% 구간은 MSE차이(rawclip−mean/std)
**[-0.033622,-0.011102]**로0을 포함하지 않는다. RLT와의 차이는−0.000633638,
95% 구간 **[-0.003001,+0.001725]**로0을 포함하므로 RLT보다 우수하다고 확정하지 않는다.
Seed0 한 번의 결과이며 seed 변동성은 측정하지 않았다. 이전 test 결과를 본 후 진행한
탐색적 후속 비교이고 online control 성능을 측정한 실험이 아니다.

모든 비교 결과의16trajectory별 유효 target 수가 같고 총248112elements이다.
주 MSE를 trajectory별 값으로 재집계해 일치함을 확인했다. 새 checkpoint 선택 seal은
14:01:37 UTC에 완료됐고 test export는14:01:38 UTC에 시작됐다.
전처리가 입력 표현과 reconstruction loss 스케일을 함께 바꾸므로 개선을 clipping,
성분별 재가중치 제거, visual/tactile loss 균형 중 하나의 효과로 특정할 수 없다.

### 앞으로의 코드 정책

- 모든 Assembly config는 `tactile_preprocessing={"mode":"raw_clip","scale":30.0}`를 명시한다.
  생략해도 동일하게 동작한다. `standardize` 모드는 거부하며 tactile mean/std 입력
  경로는 제거했다. Encoder 입력과 decoder target은 같은 변환된 batch field다.
- 새 protocol 준비는 state/action 통계만 계산한다. 기존 protocol의 역사적 tactile
  통계 필드는 보존하지만 사용하지 않으며, 기존 데이터·split·cache를 재생성하지 않는다.
- 전처리 metadata는 checkpoint/latent identity에 항상 기록한다. 과거 mean/std
  checkpoint로 새 입력의 latent를 export/resume하는 것은 mismatch로 차단한다.
  과거 실행의 artifact와 source/config snapshot은 보존한다.
- 이 정책 변경은 새로운 학습·GPU 실행·test 평가를 시작하지 않았다. 현재 T-Rex
  deformation `/255` 처리에는 적용하지 않는다.

실행 근거(서버 로컬, Git-ignored):

- [최종 보고서](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/final_report.md)
- [최종 집계·paired bootstrap](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/summary.json)
- [실제 완료 상태](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/status.json)
- [선택 checkpoint seal](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z/final_selections.json)
- [새 기본 정책 CPU 검증](../../experiments/trlt/3dtacdex_assembly/clipping_default_validation/summary.json)

## Rawclip30 self-only 4-layer 후속 실험 (2026-11-20)

사용자가 완료된 rawclip30 self2의 fusion을2층에서4층으로 바꾸도록 요청했다.
기존 `configs/trlt/3dtacdex_assembly/trlt_self_only.json`이 요청과 정확히 일치하며,
실제 self2 run config snapshot과 대조해 `model.num_stages:2→4` 외 차이가 없음을
확인했다. Tactile 입력·target clip(raw/30,-1,1),lambda_tactile1,modality-off,
seed0,Stage A8000updates,batch64/micro64,Stage B defaults,online W&B10을 유지한다.
기존 데이터/split/통계/7402-anchor visual cache를 재사용한다.

새 run: `assembly_self4_rawclip30_8k_fullb64_seed0_20261120T051945Z`.
실행 폴더:
`experiments/trlt/3dtacdex_assembly/runs/assembly_self4_rawclip30_8k_fullb64_seed0_20261120T051945Z/`.
Tmux socket `trlt-runs`, session `trlt-assembly-self4-rawclip30-seed0`.
독립된 checkpoint/latent 경로를 사용하며 resume하지 않는다. 명시적9단계는 GPU
1-update preflight+validation,Stage A,train/validation exports,Stage B,selection
seal,test export/evaluation,기존 rawclip30 self2 및 RLT와의 비교 보고다.

CPU CLI dry-run,실행 guard와 historical-only 보고 replay 근거:
`experiments/trlt/3dtacdex_assembly/self4_rawclip30_validation/`.
실제 GPU update/학습/평가 상태는 run의status.json,events.log,stage_logs와
checkpoint를 확인한다. 과거 test 결과를 확인한 뒤의 seed0 탐색적 후속 비교다.
Source/config/input 변경,점유된 GPU,실패 단계는 queue를 멈추며 자동 retry/resume하지 않는다.

실제 queue 시작은2026-11-20 14:21:19KST (05:21:19UTC),GPU0이다.
CPU CLI7개 dry-run,기존 지표 복사본으로 보고 코드 replay,실행 직전 source/config/
input/GPU-idle guard가 통과했다. Active index는
`experiments/trlt/3dtacdex_assembly/self4_rawclip_active_run.json`이며,
첫 GPU preflight가 실제 모델 초기화를 시작한 것을 확인했다. 이 관측은 benchmark
완료가 아니며 이후 상태는 실제events/checkpoint를 기준으로 판단한다.


### Rawclip30 self4 완료 결과 (2026-11-20 확인)

위 self4 run은 **2026-11-20 17:50:42KST (08:50:42UTC)**에 평가·보고까지
**9/9단계 완료**했다. 실제 Stage A8000updates, validation-selected best7888;
Stage B best epoch14,last24(patience10)이다. 새 checkpoint hash와 raw_clip30
provenance를 확인했고,16원본 trajectory/248112valid target elements가 RLT·rawclip30
self2와 동일함을 검증했다. 원본 trajectory별 값에서 주 MSE를 재집계해 일치했다.
Seal 완료08:49:43UTC 이후08:49:44UTC에 test export를 시작했다.

| 설정 | 주 Test normalized trajectory-macro action MSE ↓ | RLT 대비 |
|---|---:|---:|
| RLT | 0.059761459 | 기준 |
| Rawclip30 self2 | 0.059127821 | −1.06% |
| Rawclip30 self4 | 0.061742904 | +3.32% |

Self4는 동일 rawclip30 self2보다 MSE가 **4.42% 높았다**.
Paired 원본 trajectory-bootstrap95%구간(self4−self2)은
**[-0.001400,+0.007499]**로0을 포함한다. RLT 대비 구간도
**[-0.001875,+0.007694]**로0을 포함한다. 이번 seed의 관측상 self2가 가장 낮지만,
4층의 일관된 악화나 세 설정의 확정적 우열을 주장하지 않는다. 두 실행은 실제
config snapshot상 fusion depth2→4만 다르며, seed0 한 번의 test-inspected 후속
실험이다. 과거 mean/std self4의0.092136257보다는32.99% 낮지만 전처리는 입력과
loss의 수치적 스케일을 함께 바꾸므로 어느 하나의 효과로 특정하지 않는다.

실제 근거:
[최종 보고서](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self4_rawclip30_8k_fullb64_seed0_20261120T051945Z/final_report.md),
[비교·bootstrap 집계](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self4_rawclip30_8k_fullb64_seed0_20261120T051945Z/summary.json),
[검증 기록](../../experiments/trlt/3dtacdex_assembly/runs/assembly_self4_rawclip30_8k_fullb64_seed0_20261120T051945Z/results_review.json).
이 결과 검토는 새 학습·추론·평가를 실행하지 않았다. 위 launch/preflight-only
관측을 완료 상태로 대체하며, 완료 queue를 재시작하지 않는다.
