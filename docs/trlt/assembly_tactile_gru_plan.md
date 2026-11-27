# Assembly 촉각 MLP–GRU 오프라인 비교: 확정 계획 및 실행 인계

- 확정일: 2026-11-27 (Asia/Seoul)
- 브랜치: `experiment/assembly-tactile-gru`
- 분기 기준: `attention-actor-3dtacdex@0d4c06146debd761b692924f04fc3c11384a6ea7`
- 상태: **계획 확정. 이 문서 등록 시점에는 새 모델 구현·학습·평가를 실행하지 않았다.**
- 이번 인계의 실행 범위: 서버 자산 조사, 구현, CPU 검증, 실행 준비 및 문서화.
- GPU 사용, feature 추출 실행, 학습, 최종 test 평가는 별도 사용자 승인 후 진행한다. 이 계획의 확정이나 브랜치 생성 자체를 학습 승인으로 간주하지 않는다.

## 1. 목적과 범위

같은 촉각 이력을 사용할 때, Actor의 시간축 연결 MLP보다 GRU가 행동 예측에 유용한 특징을 만드는지 검증한다.

최종 행동 출력까지 학습·평가하는 오프라인 Stage B 비교다. Stage A 재학습, 실물 제어, 온라인 RL, RDP action tokenizer, VLA reference action 보정은 범위 밖이다. GRU는 촉각 encoder이며 Actor 전체를 대체하지 않는다.

질문:
1. 별도 촉각 경로가 도움이 되는가?
2. 현재 촉각보다 과거 이력이 도움이 되는가?
3. 같은 이력에서 GRU의 순차 처리가 MLP보다 유리한가?

## 2. 기존 구현 및 근거의 상태

- 기존 H3는 taxel별 과거 xyz를 연결해 학습형 MLP로 처리하고, 손가락별 cross-attention으로 Actor에 전달한다.
- 기본값은 tip-only, 2프레임, d128, dropout 0이다.
- 본 실험의 M5는 5프레임 및 dropout 0.1을 적용하는 **H3 기반 새 조건**이며 기존 H3와 동일한 실행으로 표기하지 않는다.
- 분기 시점의 AGENTS.md는 H0–H3가 구현됐지만 Assembly cache·GPU 학습·test 평가가 미실행이라고 기록한다.
- 서버에서 이후 실행됐는지는 직접 확인해야 한다. 문서 기록, 직접 확인한 산출물, 새 실행 결과를 구분한다.
- 기존 actor 결과에는 hierarchical d128/dropout 0.1의 3-seed 평균 MSE 0.051502가 기록돼 있다. 이는 별도 촉각 H3 결과가 아니다.

참고:
- [기존 H0–H3 계약](lingbot_3dtacdex.md)
- [Assembly actor 결과](attention_actor_3dtacdex_results_2026-11-21.md)
- [Stage A Assembly 실험](3dtacdex_experiments.md)

기존 설정, 데이터, 결과, checkpoint를 덮어쓰지 않는다.

## 3. 비교 조건

| ID | Actor의 별도 촉각 경로 | 목적 |
|---|---|---|
| B0 | 없음 | 별도 경로 없는 기준선 |
| M1 | 현재 프레임 → MLP | 직접 촉각 입력 효과 |
| M5 | 최근 5프레임 연결 → MLP | 이력 효과 |
| G5 | 동일한 5프레임 → GRU | 순차 처리 효과 |

주 비교는 G5−M5, 보조 비교는 M5−M1 및 각 조건−B0다.
B0도 Stage A latent를 통한 촉각 정보가 있으므로 '촉각 없는 모델'이라고 부르지 않는다.

## 4. 공통 조건

| 항목 | 설정 |
|---|---|
| 데이터 | 3DTacDex Assembly |
| 분할 | 기존 원본 trajectory 기준 train77 / validation16 / test16 |
| Stage A | 완료된 self-only2 rawclip30 seed0 모델 및 latent 동결 |
| 추가 촉각 | 네 손가락 tip pad, pad당15 taxel |
| 관측 주기 | 5 Hz |
| 5프레임 | t−0.8초부터 t까지 |
| Actor | hierarchical, d128, 2층, 4 attention heads |
| Dropout | 모든 조건 0.1, 평가·추론에서는 비활성화 |
| 출력 | 기존 3-step ×22D 행동 및 validity mask |
| Stage B seed | 0,1,2 |

Stage A 예산·시드·표현을 변경하지 않는다. 3개 Stage B 시드는 Stage A 학습 변동성을 측정하지 않는다.

추가 촉각의 전처리·위치 정보는 기존 H3 계약을 명시적으로 유지한다:
- XY: clip(raw/30,−1,1), Z: clip(raw/30,0,1).
- 이는 Stage A xyz signed rawclip30과 다른 경로다.
- 공개 데이터의 scale30을 실제 장비의 물리 단위로 해석하지 않는다.
- 기존 근사 geometry와 finger 순서, mask를 기록한다.
- 이번 비교에서 전체8pad 확대, 합력 표현 변경, geometry 변경을 함께 하지 않는다.

## 5. 아키텍처와 학습 범위

공통 경로:
고정 Stage A z + visual age로 조건화한 부위별 Actor.
현재22D proprioception → 팔 및4개 손가락 token → hierarchical block → 손가락별 촉각 cross-attention.
이 블록을2층 사용하고 부위별 head에서3×22D 행동을 출력한다.

M5:
- taxel별 시간순 xyz5개를15차원 벡터로 연결.
- 학습형 MLP로 token 생성.

G5:
- 같은 taxel xyz5개를 과거부터 현재 순서로 GRU에 입력.
- 마지막 hidden state를 projection하여 M5와 동일한 token 차원 생성.
- GRU 가중치는 taxel 간 공유.
- 각 window에서 hidden state를0으로 초기화.
- window 간 state 전달 없음.
- 기존 H3의 층별 촉각 처리 배치를 유지.

두 조건의 token 수·차원, 위치 정보, attention 연결과 head 구조를 동일하게 둔다. Actor는 각 조건별로 따로 학습한다. Stage A만 동결하며 촉각 MLP/GRU와 Actor는 학습한다.

GRU hidden dimension32를 초기 구현 후보로 둔다. 촉각 모듈과 전체 파라미터 수를 각각 집계하고 test 전에 최종 구성을 확정한다. 용량 차이가 크면 GRU와 촉각 모듈 용량을 맞춘 MLP 대조군을 추가하고 실행 수를 명시한다. 총 모델 크기만으로 용량 일치를 주장하지 않는다.

현재 hierarchical 연결에서 팔 token은 손가락 token을 읽지 않고 촉각 cross-attention도 손가락만 갱신한다. 별도 촉각이 팔에 직접 전달되는 구조로 바꾸지 않는다. 팔은 z를 통해 간접 촉각 정보를 받을 수 있다.

## 6. 서버 사전 조사 및 자산 재사용

실제 서버 checkout과 AGENTS.md를 확인한다. 공유 환경의 다른 사용자 인증이나 작업을 변경하지 않는다. 이 repo의 명령은 기존 sehwa 환경에서 실행한다:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
```

작업 중인 변경을 덮어쓰거나 강제 checkout하지 않는다. 필요하면 독립 worktree를 사용한다. 기존 uv 환경을 재설치하지 않는다.

확인 항목:
- source commit 및 working tree 상태.
- 기존 H3 cache/checkpoint/평가 산출물의 실제 존재와 실행 상태.
- split, normalization, query 목록, Stage A checkpoint 및 latent의 provenance/hash.
- 완료된 self2 rawclip30 seed0 실행:
  `assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z`.
- B0와 정확히 같은 d128/dropout0.1 설정의 기존 seed0/1/2 checkpoint와 개별 예측.

동일 계약이 확인된 visual cache·Stage A latent는 재사용하고 재추출·재학습하지 않는다. B0는 동일 계약이 검증되면 재사용하고, 아니면 새로 학습한다. 과거 집계 숫자만으로 동일 조건이라고 간주하지 않는다.

학습 규모: M1/M5/G5 ×3seed =9회. B0 재학습 시12회. 용량 대조군은 별도 산정.

## 7. 구현·CPU 검증

- query 시각 이하 촉각만 사용하고 segment 경계를 넘지 않는다.
- 시작 구간의 부족한 이력과 invalid pad를 모든 조건에서 동일 규칙으로 처리한다.
- 유효성 계약을 확인하여 실제 영접촉과 결측을 구분한다.
- 이력 길이 변경으로 조건별 query 집합이 달라지지 않게 한다.
- 1/5프레임 cache에는 기존2프레임과 다른 identity를 부여한다.
- raw 경로의1/5프레임 지원을 확인하고 LingBot의 짝수 tubelet 제약을 raw/GRU에 불필요하게 적용하지 않는다.
- finger/taxel/time 순서와 (segment_id,query_frame_index) join을 검증한다.
- 현재 stop_gradient는 raw cache 입력을 위한 것이다. trainable MLP/GRU 출력의 gradient를 차단하지 않도록 검증한다.
- CPU synthetic update에서 MLP/GRU와 Actor의 gradient·파라미터 갱신을 확인하고 Stage A 불변을 확인한다.
- dropout off 평가의 결정성, output shape, mask, 경계·결측 조건을 검증한다.
- 기존 H3와 기존 설정의 회귀 동작을 보존한다.

CPU 검증에는 CUDA_VISIBLE_DEVICES="" 및 JAX_PLATFORMS=cpu를 명시한다. CPU 검증을 이유로 GPU를 사용하지 않는다.

## 8. 학습·선택 계약

기존 Stage B의 optimizer, learning-rate schedule, batch, 최대 epoch, patience, checkpoint 선택 지표를 코드와 실제 설정에서 확인하여 모든 조건에 동일하게 적용한다. 변경사항은 실행 전에 문서화한다.

조건·용량·하이퍼파라미터를 test를 보며 선택하지 않는다. validation으로 checkpoint를 선택하고 기존 selection seal/provenance 검사를 보존한다. test cache와 평가도 기존 gate를 준수한다.

학습 실행은 별도 사용자 승인 후 짧은 실행 점검, 본 학습 순으로 진행한다. 자동으로 추가 seed, 재시도, 긴 이력 실험을 시작하지 않는다.

## 9. 평가와 판정

주 지표: normalized action trajectory-macro MSE.

보조:
- 첫 행동 오차.
- 손 전체·손가락별 및 팔 오차.
- 접촉 변화 구간 오차.

접촉 변화 구간 기준은 train/validation에서 정해 test 전에 고정한다. 입력에 미래 촉각을 사용하지 않는다. threshold와 metric 단위·집계 방식을 보고한다.

각 seed 및3seed 평균·표준편차를 보고한다. G5−M5는 원본 trajectory를 paired resampling하여 bootstrap 구간을 산출한다. 시드 평균 후 trajectory bootstrap을 쓰면 그 절차를 명시하고, trajectory 변동과 학습 시드 변동을 혼동하지 않는다. frame들을 독립 표본으로 취급하지 않는다.

판정은 평균 개선, seed별 방향, paired 불확실성, 첫 행동·손·접촉 구간의 개선과 추론 비용을 함께 본다. 3seed만으로 보편적인 우위를 주장하지 않는다.

- G5<M5: 동일 이력에서 GRU 처리의 이득을 지지.
- M5≈G5<M1: 이력은 유용하나 GRU 필요성은 약함.
- M1≈M5≈G5<B0: 직접 촉각 경로 효과가 중심.
- 모두 B0와 유사: Stage A와의 정보 중복 또는 추가 경로 활용 부족을 검토.

낮은 값이 좋은 MSE 기준이다. 기존 test는 이미 후속 실험에 사용됐으므로 탐색적 비교임을 명시한다. 결과는 실물 성공률, 빠른 폐루프 반응성, 온라인 RL 표본 효율을 입증하지 않는다.

## 10. 지연 평가

동일 장치·정밀도·입력 조건에서:
- batch1, dropout off.
- 초기 컴파일과 warm inference를 분리.
- 장치 실행 완료를 동기화하여 측정.
- 촉각 모듈 및 전체 Actor를 각각 측정.
- median/p95와 반복·warmup 횟수, 장치·소프트웨어 정보를 보고.
- 전체 Actor 측정에 촉각 처리 시간을 포함.

G5는 매 query에서5프레임 window를 처리하는 실제 구현을 측정한다. 지속 hidden state를 유지하는 streaming GRU의1-step 속도로 대체하지 않는다.

과거 관측은 버퍼에 있으므로0.8초 이력을 사용한다고 매번0.8초를 기다리지는 않는다. 다만 초기 buffer와 결측 처리는 별도 계약이다. 모델 추론 지연과 센서·통신·로봇 제어 포함 end-to-end 지연을 구분한다. GPU latency benchmark도 별도 GPU 승인 범위에 포함한다.

## 11. 진행 단계 및 완료 산출물

1. 서버 자산·실행 이력 조사.
2. 모델/config/cache/launcher 구현.
3. CPU 계약·gradient·회귀 검증.
4. 파라미터 수, 최종 조건, 실행 명령, 예상 실행 수를 실행 준비 문서에 정리.
5. 구현 commit/push 후 사용자에게 준비 상태와 미실행 항목 보고.
6. 별도 학습 승인 후 실행 점검·본 학습.
7. validation 선택 봉인 후 test 평가 및 지연 평가.
8. 결과 보고.

문서에는 실제 존재하는 CLI만 기록한다. 구현되지 않은 명령을 실행 가능한 것으로 제시하지 않는다.

산출물:
- 확정 설정과 source commit.
- cache/checkpoint provenance.
- CPU 검증 결과와 미검증 범위.
- 시드별 지표 및 trajectory별 예측.
- 파라미터 수·지연.
- 비교 보고서와 실행 상태.

데이터는 datasets/, checkpoint는 checkpoints/, 실행 산출물은 experiments/ 아래에 저장한다. Git-ignored 산출물은 강제 추가하지 않고 문서에서 경로를 연결한다. 이 계획과 실행 결과는 별도 문서로 유지하며 미실행을 완료로 바꾸어 쓰지 않는다.

## 12. 후속 범위

첫 비교 후 필요하면 MLP와 GRU 모두 동일하게2/5/10프레임으로 확장한다. 전체8pad, 합력 표현, 팔 촉각 경로, 지속 hidden state, RDP식 실행 중 행동 갱신, 온라인 RL은 별도 설계·승인 대상이다.
