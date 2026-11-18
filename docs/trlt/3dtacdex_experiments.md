# 3DTac-Dex Assembly: RLT / TRLT 비교 계획과 구현

이 문서는 [완료된 Assembly 변환](3dtacdex_assembly.md)을 사용하는 **별도 데이터셋 내 비교**다.
T-Rex와 같은 Stage A → frozen latent → Stage B → final test 절차를 사용한다.
본 학습·최종 test 평가는 아직 시작하지 않았다.2026-11-18 사용자 승인으로 실제 train/validation
일부 샘플의 bounded GPU 검증을 완료했으며, 결과는 아래 최신 검증 보고서에 기록한다.

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
기존 tactile decoder가 각 history/pad slot의 45개 표준화 taxel 값을 재구성한다.
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

state22, action22, tactile8×15×3 각각의 mean/std는 **train 프레임만** 사용한다.
action은 element mask, tactile은 pad mask와 finite 여부를 반영한다. 0 접촉은 유효하다.
std floor는 1e-6이며 count/floored mask도 저장한다. 누락 hand command 10프레임은
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
생성 계획 전체와 본 학습은 아직 미실행이다. 별도 bounded GPU 검증의 실제 결과는 최신 보고서를 참조한다.
본 실험 Stage A/B의 W&B는 config에서 online으로 설정하며,
`rnsrns118-seoul-national-university/trlt`에 기록한다. GPU preflight는 명시적으로 W&B disabled,
GPU 검증 run은 offline으로 동작한다. 별도 검증용 run 하나에 최소 설정과 실제 probe 지표2개를
온라인 기록하고 API 재조회까지 확인했다. 본 실험의8,000-step 학습은 시작하지 않았다.

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
