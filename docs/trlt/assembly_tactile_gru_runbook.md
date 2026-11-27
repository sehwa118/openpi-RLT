# Assembly 촉각 MLP–GRU 비교: 실행 준비 문서

- 계획: [assembly_tactile_gru_plan.md](assembly_tactile_gru_plan.md) (등록 커밋 `6a5e0e9`)
- 작성: 2026-11-27, 브랜치 `experiment/assembly-tactile-gru`
- 이 문서는 실제로 구현·확인된 CLI와 설정만 기록한다. 실행 결과는 별도 결과 문서에 기록한다.

## 1. 확정 조건

| ID | config | 별도 촉각 encoder (층별, taxel·손가락 간 가중치 공유) | raw cache | 실행 |
|---|---|---|---|---|
| B0 | `attention_actor/probe_part_attn_hierarchical_drop01.json` | 없음 | – | 기존 checkpoint 재사용 (§3) |
| M1 | `tactile_gru/probe_tgru_m1.json` | 현재 xyz(3) → Dense128 → GELU → Dense128 | `raw_t1` | 신규 seed0/1/2 |
| M5 | `tactile_gru/probe_tgru_m5.json` | 5프레임 연결 xyz(15) → Dense128 → GELU → Dense128 | `raw_t5` | 신규 seed0/1/2 |
| M5s | `tactile_gru/probe_tgru_m5s.json` | 5프레임 연결 xyz(15) → Dense53 → GELU → Dense128 (G5 용량 대조) | `raw_t5` | 신규 seed0/1/2 |
| G5 | `tactile_gru/probe_tgru_g5.json` | 5프레임 xyz를 과거→현재 순서로 GRU(32), window마다 0 초기화, 마지막 hidden → Dense128 | `raw_t5` | 신규 seed0/1/2 |

공통: hierarchical d128, 2층, 4 heads, dropout 0.1 (평가 시 off), 출력 3×22D, touch cross-attention 4 heads,
`touch_use_pos_xyz=true`, AdaLN-Zero gate(초기 출력 = B0 함수). 학습: AdamW lr 5e-4 cosine(warmup 200, min 0.1),
wd 1e-4, clip 1.0, batch 256, 최대 150 epoch, patience 20, validation `normalized_action_micro_mse`로 best 선택.
이는 B0 checkpoint의 실제 `train_config`와 동일하다(촉각 필드 제외).

기존 H3(`lingbot_touch/probe_touch_raw_taxel.json`: 2프레임, dropout 0, `raw_mlp`)와는 다른 새 조건이다. H3 config와
identity는 바뀌지 않았다(테스트로 고정).

파라미터 수 (`tactile_gru_test.py::test_parameter_counts`로 고정):

| 조건 | 전체 | touch cross-attention 2층 | 그중 encoder |
|---|---|---|---|
| B0 | 721,602 | 0 | 0 |
| M1 | 987,842 | 266,240 | 34,048 |
| M5 | 990,914 | 269,312 | 37,120 |
| M5s | 969,314 | 247,712 | 15,520 |
| G5 | 969,218 | 247,616 | 15,424 |

M5 encoder가 G5의 2.4배이므로 계획 §5에 따라 M5s를 test 전에 추가했다. 실행 수: 신규 4조건×3seed = **12회**, B0 재사용.

## 2. 구현

- `src/openpi/models/tactile_readout.py`: `touch_kv_mode="raw_gru"`, `raw_hidden_dim`(raw_mlp 폭), `raw_frames`(cache 채널
  = 3×frames 강제). 입력 cache만 `stop_gradient`이며 MLP/GRU는 학습된다.
- `src/openpi/trlt/config.py`: `touch_gru_hidden_dim`, `touch_raw_hidden_dim`, `touch_raw_frames` (None이면 직렬화 생략 →
  B0/H1–H3 identity 불변). 잘못된 조합은 거부.
- raw touch cache(`touch_tokens.py`)는 변경 없음: raw 경로는 이미 `frames` 임의 정수를 지원하고 LingBot 짝수 tubelet 검사는
  lingbot/random source에만 적용된다. identity에 `adapter.frames`가 들어가 1/2/5프레임 cache는 서로 구분된다.
- 이력 규칙(기존 H3와 동일, 전 조건 공통): query 행 이하의 연속(contiguous) 행만 사용, 부족한 앞부분은 segment 내 가장
  이른 연속 행을 복제(`replicated_history_queries`), 과거 프레임의 invalid pad는 0, token 유효성은 query 시점 pad 유효성.
- `scripts/run_tactile_gru_stage_b.sh`: 새 launcher(기존 H0–H3 launcher는 무변경).
- `scripts/benchmark_tactile_gru_latency.py`: batch1 지연(촉각 encoder 단독 / Actor 전체, 컴파일·warm 분리).

## 3. 자산 확인 (직접 확인, 2026-11-27)

- Stage A: `assembly_self2_rawclip30_8k_fullb64_seed0_20261119T104744Z` latent `trlt_self_only/seed0/{train,validation,test}`.
  representation checkpoint hash `b3787bea…`, train latent cache `13f14edb…`, validation `79848a46…`, split `b9457d3b…`,
  query manifest `68e484de…`. 재학습·재추출하지 않음.
- B0 재사용: `/workspace/users/jiwoo/trlt_attention_actor_3dtacdex_self2_rawclip30_d64_mseed/checkpoints/seed{0,1,2}/probe_part_attention_hierarchical_drop01`
  (읽기 전용). 세 seed 모두 `train_config`가 위 B0 config와 동일, 721,602 params, git `d6b0700`(clean, HEAD 조상), 위 Stage A
  representation/latent hash와 일치. `d6b0700..HEAD` 변경은 `touch_part_attention` 추가와 `apply_probe` 리팩터링뿐으로
  `part_attention` 계산은 동일. jiwoo가 이 B0의 test 평가를 이미 수행했으므로 test는 이미 사용된 상태 → 탐색적 비교.
- 기존 H3 Assembly cache/checkpoint: 서버에서 발견되지 않음(문서 기록대로 미실행).
- 새 raw cache (`/workspace/users/hamin/data/assembly_tactile_gru/datasets/touch_cache/raw_t{1,5}/{train,validation}`):
  14,819 / 3,472 query, latent 행 100% join, 1/5프레임 key 동일, T5 마지막 slot = T1, tip pad 유효율 1.0,
  T5 시작부 복제 query train 356 / validation 68. 증거: `…/experiments/cpu_validation/real_data_join.json`.

## 4. CPU 검증

`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $GRU_PY -m pytest src/openpi/trlt/tactile_gru_test.py src/openpi/trlt/touch_tokens_test.py src/openpi/trlt/runner_test.py src/openpi/trlt/tacdex_experiment_test.py`
→ 전부 통과. 확인 내용: config 차이가 촉각 필드뿐, B0/H3 identity·H3 파라미터 트리 불변, 잘못된 조합 거부, 파라미터 수,
GRU 가중치 공유·window별 0 초기화·시간 순서 민감성·수동 GRU 재귀와 일치, cache/config 프레임 불일치 거부, 초기 출력 = B0,
gate 개방 시 encoder gradient 존재·z/touch 입력 gradient 0, optimizer update 후 Actor·encoder 갱신, dropout off 결정성,
invalid finger 마스킹, 1/2/5프레임 cache causal·segment 경계·key 정렬, G5 학습→평가 end-to-end.
초기 gate가 0이므로 정확한 init 시점 encoder gradient는 0이다(첫 update 후 gate가 열리면 전달).
`tacdex_lingbot_test.py`의 2개 실패(zarr `create_dataset … 'shape'`)는 HEAD 원본에서도 동일하게 실패하는 환경 문제다.

## 5. 실행 명령

환경 (hamin): `source /workspace/users/hamin/data/assembly_tactile_gru/env.sh` 후 아래 변수 사용.

```bash
D=$GRU_WT/datasets/3dtacdex_assembly
COMMON=(--protocol-dir $D/protocol --data-root $D/lerobot_all
  --latent-root $D/latents/$GRU_STAGE_A_RUN/trlt_self_only
  --out-root $GRU_OUT/experiments/stage_b --touch-root $GRU_OUT/datasets/touch_cache
  --b0-root $GRU_B0_CANDIDATE --python $GRU_PY --seeds "0 1 2" --variants "b0 m1 m5 m5s g5")
scripts/run_tactile_gru_stage_b.sh --stage caches "${COMMON[@]}"        # CPU, train/val raw_t1/raw_t5
scripts/run_tactile_gru_stage_b.sh --stage train "${COMMON[@]}"         # GPU, 12회 (b0는 건너뜀)
scripts/run_tactile_gru_stage_b.sh --stage seal "${COMMON[@]}"          # CPU, seed별 b0+4조건 봉인
scripts/run_tactile_gru_stage_b.sh --stage test-caches "${COMMON[@]}"   # CPU, seal 필요
scripts/run_tactile_gru_stage_b.sh --stage evaluate "${COMMON[@]}" --normalization $D/protocol/normalization.json  # GPU
scripts/run_tactile_gru_stage_b.sh --stage summarize "${COMMON[@]}"     # vs b0 / m5 / m1
```

GPU 단계는 `--mem-fraction 0.05`(약 4.8 GB), 무선할당, 여유 10 GB 미만이면 시작하지 않는다. 이미 checkpoint/평가가 있는
경로는 거부한다. 모든 출력은 `/workspace/users/hamin/…` 아래로 제한된다.
짧은 점검은 `--out-root $GRU_OUT/experiments/stage_b_smoke --max-steps 50`으로 별도 경로에서 실행한다.

지연(batch1, 조건별 별도 프로세스):

```bash
$GRU_PY scripts/benchmark_tactile_gru_latency.py --device 0 \
  --checkpoint <ckpt>/best --latents $D/latents/$GRU_STAGE_A_RUN/trlt_self_only/seed0/validation \
  --touch $GRU_OUT/datasets/touch_cache/raw_t5/validation --output <out>.json   # B0는 --touch 생략
```

## 6. GPU 사용량 근거

- synthetic 측정(GPU0, PREALLOCATE=false, 조건·모드별 별도 프로세스): 학습 b256 JAX peak 118–287 MiB, 프로세스 786–1,720 MiB;
  추론 b1 JAX peak 약 155 MiB, 프로세스 약 0.9–1.2 GiB. G5 수치는 당시 임시 프로토타입 기준이다.
- 실제 파이프라인 50-step 점검(`stage_b_smoke`, seed0, 4조건): 모두 정상 종료, train loss 1.90→0.47, update 약 0.02 s/step,
  checkpoint provenance에 touch identity·Stage A hash 기록 확인. 점검은 commit 전 코드(`git_dirty=true`)라 본 결과가 아니다.
