# Assembly 촉각 MLP–GRU 후속: 긴 이력(10·20프레임) 계획

- 선행: [assembly_tactile_gru_results.md](assembly_tactile_gru_results.md) (`acf7997`). 5프레임(1 s)에서 G5 ≈ M5, 5프레임 이력은
  B0 대비 약 3% 개선(탐색적).
- 작성: 2026-11-27, hamin, 브랜치 `experiment/assembly-tactile-gru`. 이 문서는 학습 전에 commit하며, 이후 결과를 보고 조건·
  하이퍼파라미터·분석 방법을 바꾸지 않는다.

## 1. 질문

1. **Q1 (주):** 이력이 길어지면(10프레임 = 2 s, 20프레임 = 4 s, 5 fps) GRU가 연결-MLP보다 나아지는가?
   주 비교 G10 − M10, G20 − M20. 보조 G10 − M10s, G20 − M20s(용량 대조).
2. **Q2:** 이력 길이를 늘리면 계속 좋아지는가? M1 → M5 → M10 → M20, G5 → G10 → G20.

## 2. 조건 (신규 6조건 × seed 0/1/2 = 18회)

| ID | config | encoder (층별, taxel·손가락 공유) | encoder 파라미터 (2층) | cache |
|---|---|---|---|---|
| M10 | `tactile_gru/probe_tgru_m10.json` | 10프레임 연결 xyz(30) → Dense128 → GELU → Dense128 | 40,960 | `raw_t10` |
| M10s | `probe_tgru_m10s.json` | 같은 입력 → Dense48 → GELU → Dense128 | 15,520 | `raw_t10` |
| G10 | `probe_tgru_g10.json` | GRU(32) 10단계, window별 0 초기화, 마지막 hidden → Dense128 | 15,424 | `raw_t10` |
| M20 | `probe_tgru_m20.json` | 20프레임 연결 xyz(60) → Dense128 → GELU → Dense128 | 48,640 | `raw_t20` |
| M20s | `probe_tgru_m20s.json` | 같은 입력 → Dense40 → GELU → Dense128 | 15,376 | `raw_t20` |
| G20 | `probe_tgru_g20.json` | GRU(32) 20단계 | 15,424 | `raw_t20` |

- 용량 대조 폭은 G encoder 파라미터에 가장 가까운 정수 폭으로 정했다(M10s +0.6%, M20s −0.3%). 원 계획 §5의 규칙("용량 차이가
  크면 대조 추가")을 따랐다. M 계열은 G의 2.7배(T10), 3.2배(T20)다.
- 나머지는 모두 M5/G5와 같다(hierarchical d128, 2층, 4 heads, dropout 0.1, 학습 설정 동일). 테스트로 고정했다:
  `tactile_gru_test.py`의 config 차이, 파라미터 수, 10/20프레임 cache 인과성, segment 시작부 복제, cache–config 프레임 불일치 거부.
- 재사용(재학습 없음): B0(jiwoo), M1/M5/M5s/G5(`experiments/stage_b`, 이미 test 평가됨).
- 이력 규칙은 기존과 동일하다. segment 시작부에서 부족한 이력은 segment의 첫 행을 복제한다. 20프레임에서는 복제되는 query 비율이 늘어난다(manifest에 기록).

## 3. 절차

출력: `/workspace/users/hamin/data/assembly_tactile_gru/experiments/stage_b_long` (기존 `stage_b`와 seal은 건드리지 않음),
cache: `datasets/touch_cache/raw_t{10,20}/{train,validation,test}`.

1. `caches` (CPU) → `train` (GPU, 18회, 자동 재시도·추가 seed 없음).
2. 선택: checkpoint는 validation `normalized_action_micro_mse` best(기존과 동일). 조건별 validation 3-seed 평균을 먼저 기록한다.
3. `seal` (b0 + 신규 6조건, seed별) → `test-caches` → `evaluate`(신규 6조건 + b0 재추론, 조건·seed당 1회).
4. 분석: `analyze_tactile_gru_results.py --logs stage_b/logs stage_b_long/logs`로 기존 평가와 합쳐 계산한다. 지표, 접촉 변화 정의
   (`stage_b/contact_change_definition.json`, 변경 없음), bootstrap 방법은 선행 실험과 같다.
   pairs: `g10:m10 g20:m20 g10:m10s g20:m20s m10:m5 m20:m10 g10:g5 g20:g10 m10:b0 m20:b0 g10:b0 g20:b0`.
5. 지연: M10/G10/M20/G20 seed0 best, batch1 GPU, 조건별 별도 프로세스.

## 4. 판정 규칙 (사전 고정)

- "GRU가 낫다"는 G10 − M10 또는 G20 − M20 주 지표 차이의 95% CI가 0 미만이고, 3/3 seed가 같은 방향일 때만 말한다.
  그렇지 않으면 "차이 없음/불확실"로 보고한다. 보조 비교와 다른 지표는 설명용으로만 쓴다.
- 추천 모델(다음 단계용)은 test가 아니라 **validation 3-seed 평균**으로 고른다.
- test split은 이미 사용됐다. 이 비교도 탐색적이며, 다중 비교 보정은 하지 않는다(한계로 명시).

## 5. 하지 않는 것

실물 제어, 다른 사용자 산출물 변경, 기존 `stage_b` 결과·seal 덮어쓰기, 결과를 보고 하는 추가 조건·튜닝, streaming GRU.
