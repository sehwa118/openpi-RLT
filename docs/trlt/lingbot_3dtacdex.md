# 3DTacDex(PaXini)에서 LingBot-Touch 테스트

frozen LingBot-Touch target encoder에 3DTacDex의 PaXini 촉각을 넣어 feature를 추출하고, CPU 진단을 돌리는 경로입니다. 학습, action head, online RL은 범위 밖입니다.

## 구성 요소

| 파일 | 역할 |
|---|---|
| `src/openpi/trlt/tacdex_lingbot.py` | PaXini → LingBot `magnetic` 입력 어댑터(NumPy). OXT Zarr reader(zarr 2/3), causal window, trajectory 단위 split |
| `src/openpi/models/lingbot_touch_flax.py` | `load_flax_bundle`(SHA 검증), `encode_single_hand`(한 손·부분 coverage), `init_random_params`(대조군) |
| `scripts/lingbot_tacdex.py` | `inspect` / `extract` / `probe` CLI. 명령은 서로를 자동으로 실행하지 않습니다 |
| `src/openpi/trlt/tacdex_lingbot_test.py` | CPU 테스트. 실제 Tiny 번들이 있으면 그것도 검사합니다 |

## 입력 계약

- **손가락 슬롯**: LingBot 순서는 (thumb, index, middle, ring, pinky)입니다. 3DTacDex 채널 `[thumb_tip, thumb_pulp, index_tip, …, ring_pulp]`를 앞의 네 슬롯에 넣습니다. LEAP에는 pinky가 없으므로 slot 4는 0으로 채우고 `coverage=False`로 둡니다(공식 가이드의 partial-hand 방식).
- **pad 선택**: 기본값 `--pads tip`이면 손가락당 M=15입니다. `tip_pulp`이면 M=30이고, pulp를 tip보다 근위쪽(−y)에 둡니다. pulp는 중간마디에 있어 LingBot 사전학습(fingertip)에는 없는 위치이므로 실험용입니다.
- **신호**: PaXini 축 0/1은 shear(X/Y), 축 2는 normal(Z, 접촉 시 항상 양수)로 봅니다. `X=clip(X/30,−1,1)`, `Y=clip(Y/30,−1,1)`, `Z=clip(Z/30,0,1)`입니다(`--scales-xyz`로 변경 가능). 스케일 30은 프로젝트의 Assembly rawclip30 정책과 맞춘 값입니다. 3DTacDex 원 코드는 50을 썼습니다. 축 부호(`--axis-signs`)와 LingBot magnetic 축의 대응은 **검증되지 않았습니다**.
- **geometry**: 3DTacDex 원 코드(`fk/tactile.py`, commit a4a3675)의 PaXini taxel 중심 좌표(mm)를 씁니다. 센서 좌표계는 x 좌우, y 손끝 방향, z 바깥 법선이고, taxel index는 `k = 5·column + row`(column-major)입니다. 변환은 `p = (p_mm − origin) / radius`이며 기본값은 `radius=14 mm`, `origin=(0, 13.25, 3.0) mm`입니다. 이 값은 LingBot `magnetic_096` 기준 fingertip의 pad와 tip 영역(part 0/1)에 양방향 Chamfer로 맞춘 **근사 정렬**이고, 실측 calibration이 아닙니다. thumb 기준으로 맞추면 r=18.5가 나와 해가 하나로 정해지지 않으므로, non-thumb 기준값을 네 손가락에 공통으로 씁니다. 회전은 identity입니다. pad가 좌우 대칭이라 위치에는 x 부호 모호성이 영향을 주지 않지만, shear X 부호에는 영향을 줍니다.
- **시간**: window는 `[t−T+1 … t]`이며 segment 경계를 넘지 않습니다. segment 시작 부분은 첫 frame을 복제하고 `replicated_history`에 기록합니다. Tiny의 tubelet은 2이므로 T는 짝수여야 합니다. 데이터는 5 Hz이고 LingBot 사전학습의 Hz는 공개되지 않았습니다.
- **출력**: tip/T=2에서 window당 60 token × 768D입니다. 저장하는 값은 `finger_mean [N,4,768]`(tubelet·taxel 평균)이고, 선택적으로 `last_tubelet_tokens`(float16)도 저장합니다. 공식 권장은 attention readout이며, 평균 pooling은 진단용입니다.

## 실행 (CPU 전용)

```bash
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
# 1) 가중치: HF robbyant/lingbot-touch-tiny @ f23969cc (encoder.pt SHA256 ddb8b435…) → checkpoints/lingbot_touch/tiny_hf_<rev>/
python scripts/convert_lingbot_touch.py --bundle-dir checkpoints/lingbot_touch/tiny_hf_<rev> \
    --output-dir checkpoints/lingbot_touch/tiny_flax_<rev>          # PyTorch parity 검증 포함 (torch 필요)
# 2) 어댑터 점검 (모델 불필요)
python scripts/lingbot_tacdex.py inspect --zarr datasets/3dtacdex/openbox/openbox.zarr.zip
# 3) feature 추출: 사전학습 모델 / 무작위 초기화 대조군
python scripts/lingbot_tacdex.py extract --zarr <zip> --bundle checkpoints/lingbot_touch/tiny_flax_<rev> --tag tiny_tip_t2
python scripts/lingbot_tacdex.py extract --zarr <zip> --random-init-seed 0 --tag random0_tip_t2
# 4) 진단 (원본 trajectory 단위 0.6/0.2/0.2 split, ridge)
python scripts/lingbot_tacdex.py probe --features datasets/3dtacdex/lingbot_features/<task>/{tiny_tip_t2,random0_tip_t2}
```

출력 위치는 feature가 `datasets/3dtacdex/lingbot_features/<task>/<tag>/`, 보고서가 `experiments/3dtacdex_analysis/lingbot/`입니다. Assembly ZIP에도 같은 명령을 그대로 쓸 수 있습니다. probe의 split은 진단용 자체 split이며, Assembly 실험의 77/16/16 protocol과는 별개입니다. 이 split을 test 평가에 쓰면 안 됩니다.

## 실행 기록

실행 결과와 명령은 `experiments/3dtacdex_analysis/lingbot/README.md`에 있습니다(2026-11-21, openbox, CPU).

## Stage B 비교 실험 (H0–H3)

같은 Stage A 동결 latent(`self2 rawclip30`, seed0)와 같은 split, 정규화, optimizer 설정을 쓰고, 촉각 token 입력만 바꿉니다.

| ID | config | 입력 |
|---|---|---|
| H0 | `attention_actor/probe_part_attn_hierarchical.json` | z + proprio (기존 baseline) |
| H1 | `lingbot_touch/probe_touch_lingbot_tiny.json` | + 사전학습 LingBot Tiny token |
| H2 | `lingbot_touch/probe_touch_lingbot_random.json` | + 무작위 초기화 LingBot token (구조 대조군) |
| H3 | `lingbot_touch/probe_touch_raw_taxel.json` | + raw PaXini taxel token, 학습형 MLP tokenizer (`touch_kv_mode=raw_mlp`) |

- **모델**: `touch_part_attention`은 `TouchPartAttentionActionProbe`입니다. hierarchical part attention의 각 층 뒤에 손가락별 cross-attention을 둡니다. index, middle, ring, thumb part가 각자 자기 손가락의 token만 읽습니다. gate가 0으로 초기화되므로 학습 시작 시점에는 H0와 같은 함수입니다(테스트로 확인).
- **cache**: `scripts/build_touch_tokens.py`는 CPU 전용입니다. 시작할 때 GPU를 숨기도록 강제합니다. AssemblyDataset과 같은 query와 `contiguous` 규칙으로 window를 만들고, latent와는 `(segment_id, query_frame_index)`로 join합니다. query frame에서 pad가 invalid이면 해당 token을 mask합니다. history 프레임에서 invalid인 pad는 0으로 채웁니다. test cache는 `--selection-seal`(seal-probes 결과)이 있어야 만들 수 있습니다.
- **provenance**: `touch_identity`(어댑터 설정, 모델 SHA 또는 seed, finger 순서, K, C)가 checkpoint와 test 데이터에서 일치해야 평가됩니다.
- **실행**: `scripts/run_lingbot_touch_stage_b.sh --stage {caches|train|seal|test-caches|evaluate|summarize}`. 한 번에 한 단계만 실행하고 다음 단계를 자동으로 시작하지 않습니다. GPU 단계는 시작 전에 `nvidia-smi`로 여유 메모리를 확인하고(기본 20 GB 이상), `XLA_PYTHON_CLIENT_PREALLOCATE=false`와 `MEM_FRACTION=0.20`을 설정합니다.
- **판정 기준 (사전 고정)**: H1이 H0보다 낮고 paired trajectory-bootstrap 95% CI가 0을 포함하지 않으며, 동시에 H1이 H2보다 낮을 때 "LingBot 사전학습이 도움이 된다"고 봅니다. H3와의 비교는 같은 readout 구조에서 raw 촉각 대비 이득이 있는지를 봅니다.
