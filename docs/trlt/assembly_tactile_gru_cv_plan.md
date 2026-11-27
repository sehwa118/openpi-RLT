# Assembly 촉각 MLP–GRU: trajectory 교차검증 계획

- 선행: [assembly_tactile_gru_results.md](assembly_tactile_gru_results.md), [assembly_tactile_gru_long_history_results.md](assembly_tactile_gru_long_history_results.md).
  test(16 trajectory)에서 G20 − M20 = −2.8%(3/3 seed)였지만 CI가 0을 포함했다. test는 이미 두 번 사용됐다.
- 작성: 2026-11-27, hamin, 브랜치 `experiment/assembly-tactile-gru`. 학습 전에 commit하며, 이후 조건·분석·판정 규칙을 바꾸지 않는다.

## 1. 목적

test를 다시 쓰지 않고 표본을 늘려 확인한다. **Q1 (주): G20 − M20.** 보조로 G20 − M5(validation 1위), M5 − B0, G20 − B0, M20 − M5를 본다.

## 2. 설계

- 데이터: 고정 Stage A latent의 **train(77 trajectory) + validation(16)** = 93 trajectory, 18,291 query. **test split은 읽지 않는다.**
  `cv_folds.build_cv_datasets`는 split이 정확히 train과 validation인 source만 받는다(테스트로 고정). cache는 복사하지 않고 행을 그대로 참조한다.
- fold: source trajectory(`episode_id`) 단위 5-fold, `source_group` 층화, fold seed 0. fold당 18–19 trajectory이며, 모든 조건에 같은 분할을 쓴다
  (`experiments/cv/folds/fold{k}.json`).
- outer fold k마다 다음과 같이 나눈다.
  - 평가(held-out): fold k
  - early stop·best 선택: fold (k+1) mod 5. Stage B와 같이 validation `normalized_action_micro_mse`로 고른다.
  - 학습: 나머지 3 fold(55–57 trajectory)
- 각 trajectory는 정확히 한 번, 자신을 학습에도 선택에도 쓰지 않은 모델로 평가된다.
- 조건: B0, M5, M20, G20(config는 선행 실험과 동일). 학습 seed 0. 4조건 × 5 fold = **20회**.
- 정규화: protocol train 정규화(기존과 동일, 고정). 접촉 변화 정의도 기존 `contact_change_definition.json`을 그대로 쓴다.

## 3. 분석 (`scripts/analyze_tactile_gru_cv.py`)

- 조건별로 93 trajectory의 held-out 오차를 모아 trajectory macro 평균을 낸다. 지표는 주 지표, 첫 행동, 팔, 손, 접촉 변화다.
  다시 계산한 주 지표는 fold별 evaluator `metrics.json`과 일치해야 한다.
- 짝지은 차이: trajectory별 차이를 93개 trajectory 단위로 bootstrap(10,000회)해 95% CI를 낸다. fold별 평균 차이의 방향도 따로 보고한다.

## 4. 판정 규칙 (사전 고정)

- "G20이 M20보다 낫다"는 G20 − M20 주 지표 차이의 **95% CI가 0 미만이고, 5개 fold 중 4개 이상이 같은 방향**일 때만 말한다.
  그렇지 않으면 "차이 불확실" 또는 "차이 없음"으로 보고한다.
- 보조 비교는 설명용이며 다중 비교 보정은 하지 않는다.

## 5. 한계

- Stage A가 train trajectory를 학습했으므로, train 출신 trajectory의 절대 오차는 낙관적이다. 이 편향은 모든 조건에 공통이라 조건 간 비교에는
  영향이 작지만, 수치를 test 결과와 직접 비교하지 않는다.
- 학습 seed가 1개다. fold 사이의 변동이 seed 변동을 일부 포함한다.

## 6. 하지 않는 것

test 접근, 실물 제어, 기존 `stage_b`·`stage_b_long` 산출물 변경, 결과를 보고 하는 추가 조건·튜닝.
출력은 `/workspace/users/hamin/data/assembly_tactile_gru/experiments/cv` 아래에만 쓴다. 점검용 `cv_smoke`(30 step)는 본 결과가 아니다.
