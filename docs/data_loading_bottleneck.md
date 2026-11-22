# 3DTacDex 데이터 로딩 병목: 원인과 해결 기록

작성일: 2026-11-22

`pi05_tacdex_lora` 학습에서 GPU가 데이터를 기다리는 현상을 CPU 데이터 로더로
재현했다. 직접 원인은 데이터 변환 시 키프레임 간격을 지정하지 않아 생성된
긴 GOP(Group of Pictures)와, 랜덤 샘플마다 중간 프레임을 반복 디코딩하는
로딩 경로였다. LeRobot 기본 인코더로 학습용 복사본을 생성하고 config를
전환했으며, 동일 랜덤 샘플의 평균 로딩 시간은 약 **29배** 개선됐다.

이 문서의 데이터 로딩 성능은 CPU에서 측정했다. 수정 후 GPU 학습 처리량과
GPU utilization은 아직 측정하지 않았다.

## 1. 나타난 문제

문제가 보고된 실행 조건:

```bash
CONFIG=pi05_tacdex_lora
RUN=lora_batch32_step8k_retry
STEPS=10000
BATCH=32
NUM_WORKERS=2

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name "$RUN" \
  --batch-size "$BATCH" --num-train-steps "$STEPS" \
  --profile-dir "experiments/pi05_tacdex/debug/$RUN" \
  --num_workers "$NUM_WORKERS"
```

- GPU utilization이 대부분 0%로 보이고, 배치 사이에 긴 대기가 발생했다.
- 실제 프로젝트 데이터셋과 transform을 사용하는 CPU 로더에서도 재현됐다.
- 배치 32·워커 2로 연속 네 배치를 받을 때 대기는
  **84.62 → 1.36 → 92.28 → 13.55초**였다. 시작 비용 이후에도 대기가 반복됐다.
- 기존 GPU 실행 기록의 컴파일 이후 학습 연산은 약 1.71초/step이었다.
  데이터 공급 시간이 연산 시간보다 훨씬 길었다.

학습에는 첫 50개 에피소드, 12,235개 시점, 외부 RGB 카메라 2개를 사용한다.
전체 원본 데이터는 124개 에피소드, 22,069개 시점, 영상 248개다.

## 2. 확인된 원인

### 2.1 직접 인코딩하면서 짧은 GOP 설정을 누락

당시 데이터 변환 코드의 실험 스냅샷을 찾았고, 파일 SHA-256이
`meta/assembly_conversion.json`의 `implementation_sha256`과 일치함을 확인했다.
해당 코드는 LeRobot의 기본 인코딩 함수 대신 PyAV를 직접 호출했다.

```python
stream = container.add_stream(
    "libx264rgb",
    rate=FPS,
    options={"crf": "0", "preset": "ultrafast"},
)
```

원본 RGB 무손실 보존과 빠른 변환을 위한 설정이지만, `g` 또는 `keyint`를
지정하지 않았다. 실제 MP4 내부에 기록된 설정은 다음과 같았다.

```text
keyint=250
scenecut=0
qp=0
```

- 최대 키프레임 간격 250프레임이 적용됐다.
- 데이터가 5fps이므로 250프레임은 **50초**다.
- 장면 변화에 따른 추가 키프레임 삽입도 비활성화됐다.
- 실제 episode 0은 41초·205프레임 전체에서 첫 프레임만 키프레임이었다.
  episode 40의 키프레임 시점은 0초와 50초였다.

무손실 저장이 긴 GOP를 요구하는 것은 아니다. 무손실 여부와 키프레임 간격은
별도 설정이며, 이번 변환에서는 랜덤 접근을 위한 간격 설정이 빠졌다.

### 2.2 랜덤 프레임 접근 시 반복 디코딩

공개 openpi의 LeRobot 기반 fine-tuning과 마찬가지로 현재 경로도
`LeRobotDataset → transform → PyTorch DataLoader → JAX`를 사용한다.
별도의 RLT 모델 연산이 이 로딩 비용을 만든 것은 아니다.

현재 PyAV 경로는 요청 프레임 이전의 키프레임으로 이동한 뒤, 요청 시점까지
중간 프레임을 디코딩하고 RGB로 변환한다. 매 샘플마다 카메라별 파일을 열고
이 작업을 반복한다. 파일이 OS 캐시에 있어도 디코딩 결과가 캐시되는 것은 아니다.

원본 해상도도 1280×720 및 1024×768로 크며, 모델용 224×224 resize/padding은
프레임을 읽은 뒤 적용된다. 따라서 모델 입력 해상도가 작아도 중간 프레임의
디코딩 비용은 그대로 발생한다.

동일 랜덤 샘플 12개의 프로파일:

| 항목 | 측정 결과 |
| --- | ---: |
| 총 샘플 로딩 시간 | 43.08초 |
| 비디오 조회·디코딩 경로 | 42.51초, **98.7%** |
| 리사이즈 | 0.28초 |
| action window 조회 | 0.19초 |
| 필요한 카메라 이미지 | 24장 |
| 실제 디코딩·RGB 변환한 이미지 | **3,112장** |

LeRobot 디렉터리·메타데이터 형식을 따른다고 해서 내부 MP4가 자동으로 랜덤
접근에 적합해지는 것은 아니다. 설치된 LeRobot 인코더의 기본값은 `g=2`지만,
직접 인코딩한 원본에는 적용되지 않았다.

### 2.3 서버·디코더 환경과 측정 방식

다음은 함께 확인한 환경 조건이며, 직접 원인과 구분한다.

- **스토리지:** 로컬 NVMe의 ext4다. 관찰 구간에서 워커의 실제 디스크 읽기가
  0인 캐시 상태에서도 병목이 재현됐다. 디스크 속도가 이번 현상의 주원인이라는
  근거는 없었다.
- **CPU:** 128개 논리 CPU가 보이지만 관찰한 cgroup의 `cpu.max`는
  `1400000 100000`, 즉 14코어 상당이었다. 워커 두 개의 사용량은 합쳐 약
  3.4코어였다. 워커 수를 정할 때 고려할 한도지만 이번 병목의 주원인은 아니었다.
- **TorchCodec:** 패키지는 설치돼 있지만 FFmpeg의 `libavutil.so.56`–`.59`를
  찾지 못해 실행되지 않았다. 현재 config는 `video_backend="pyav"`를 명시한다.
  이번 수정은 디코더 설치를 변경하지 않고 동일 PyAV에서 검증했다.
- **기존 프로파일:** `training_profile.json`의 step `seconds`에는 데이터 대기가
  포함되지 않았다. 첫 step에는 컴파일도 포함되므로, 연산·컴파일·데이터 대기를
  구분해서 해석해야 한다.

다른 config도 같은 로더와 긴 GOP 영상을 사용하면 같은 문제가 생길 수 있다.
다만 다른 데이터셋까지 동일 원인이라고 검증한 것은 아니다.

## 3. 적용한 해결책

[변환 스크립트](../scripts/reencode_lerobot_videos.py)가 설치된 LeRobot의
`encode_video_frames()`를 호출하도록 했다. 코덱 관련 인자는 덮어쓰지 않는다.
이 프로젝트가 고정한 LeRobot revision은
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`다. 해당 환경의 기본값은 다음과 같다.

| 설정 | 기존 변환 | 새 학습용 변환 |
| --- | --- | --- |
| 인코더 | `libx264rgb` | `libsvtav1` |
| 저장 픽셀 포맷 | `gbrp` | `yuv420p` |
| 키프레임 간격 | 최대 250프레임 | 최대 **2프레임** |
| 품질 설정 | 무손실 `crf=0` | 손실 압축 `crf=30` |
| 원본 해상도 | 유지 | 유지 |

변환 범위와 검증:

- 124개 에피소드·248개 영상, 카메라 프레임 총 44,138장을 변환했다.
- 모든 출력 영상을 끝까지 디코딩해 프레임 수·시간축·해상도·최대 GOP를 검증했다.
- state/action 및 나머지 Parquet는 복사하고 SHA-256 일치를 검증했다.
- 기존 state/action 정규화 통계를 계속 사용한다. 이미지 통계는 새 영상에서
  기존 변환기와 같은 공간 샘플링 간격 8로 다시 계산했다.
- RGB 픽셀은 표준 손실 압축으로 달라진다. 무손실 원본은 별도 보존했다.
- 중간 결과에서 재개할 수 있고, 전체 검증 후에만 최종 데이터 경로를 공개한다.

| 용도 | 저장 경로 |
| --- | --- |
| 보존한 RGB 무손실 원본 | `datasets/3dtacdex_assembly/lerobot_all/` |
| 현재 학습용 데이터 | `datasets/3dtacdex_assembly/lerobot_default/` |
| 변환·검증 기록 | 새 데이터의 `meta/video_reencoding.json` |
| 이전 변환의 이력 | 새 데이터의 `meta/source_assembly_conversion.json` |
| 진단·성능 결과 | `experiments/pi05_tacdex/debug/default_encoding/` |

[학습 config](../src/openpi/training/config.py)의 `pi05_tacdex`와
`pi05_tacdex_lora`는 새 데이터 경로와 기본 워커 2개를 사용한다.
학습 에피소드 선택은 기존과 동일하게 0–49다.

[학습 스크립트](../scripts/train.py)에는 다음 측정값과 관련 로그를 추가했다.

- `initial_data_wait_seconds`: 워커 시작과 첫 배치 대기
- step별 `seconds`: 동기화된 학습 연산 시간, 첫 step은 컴파일 포함
- step별 `data_wait_seconds`: 해당 step 다음 배치를 가져오는 대기 시간 및
  host-to-device 전송 제출 시간. GPU 전송 완료만을 따로 측정한 값은 아니다.

체크포인트 저장 시간은 step의 `seconds`와 `data_wait_seconds`에서 제외된다.

## 4. 검증 결과와 해석 범위

배치 32·워커 2, 같은 seed와 같은 transform으로 비교했다.

| 지표 | 기존 | 변경 후 |
| --- | ---: | ---: |
| 동일 랜덤 샘플 12개 평균 로딩 | 3.590초 | **0.122초** |
| 해당 샘플들의 디코딩 이미지 수 | 3,112장 | **48장** |
| 첫 배치 대기 | 84.62초 | **3.15초** |
| 세 번째 배치 대기 | 92.28초 | **3.03초** |

샘플 평균 로딩은 약 **29.3배** 빨라졌다. 변경 후 연속 공급 실험의 초기 네
배치를 제외한 평균 배치 대기는 약 **1.84초**였다.

이어서 CPU에서 배치 소비 간격을 기존 GPU 기록의 1.712초로 모의 실행했을 때,
초기 네 배치를 제외한 평균 추가 데이터 대기는 약 **0.017초**였다. 실제 GPU
연산 대신 `sleep`으로 소비 간격을 만든 짧은 실험이므로, 이 수치를 GPU
utilization이나 장시간 학습 처리량의 실측치로 해석하면 안 된다.

추가 검증:

- 실제 JAX 로더의 CPU 배치: 카메라별 `(32, 224, 224, 3)`,
  state `(32, 32)`, actions `(32, 50, 32)`, 유한값 검증 통과
- 두 TacDex config의 데이터 경로·에피소드 선택·기본 워커 수 확인
- 변환 스크립트 재실행 시 완료된 데이터 인식 확인
- 변경 Python 파일 Ruff 검사·포맷 검사 및 `git diff --check` 통과

초기 진단 샘플 측정 일부는 변환 전 다른 CPU 진단과 겹쳤다. 위 수치는 병목
원인과 개선 규모를 확인한 결과이며, 엄격히 격리된 하드웨어 벤치마크는 아니다.
GPU 재학습, 학습 품질 및 로봇 성공률 평가는 이번 검증 범위에 포함하지 않았다.

## 5. 재현 및 학습 재시작

각 셸에서 먼저 기존 환경을 활성화한다. 환경을 다시 설치할 필요는 없다.

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
cd /workspace/users/sehwa/openpi-RLT
```

기본 인코딩으로 변환하거나 중단된 변환을 재개하는 명령:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python scripts/reencode_lerobot_videos.py \
  --jobs 8 --cpus-per-job 2
```

`--jobs`와 `--cpus-per-job`은 변환 작업의 CPU 병렬도다. 다른 서버에서는
CPU 제한과 공유 사용량에 맞춰 조정한다. 데이터 경로는 저장소 루트를 기준으로
해석된다. 새 데이터가 이미 완료된 현재 서버에서는 변환을 반복하지 않는다.

GPU 학습을 다시 실행할 때는 기존 명령의 `RUN`을 새로운 이름으로 지정하면 된다.
config가 새 데이터 경로를 사용하므로 추가 데이터 경로 인자는 필요 없다.
전체 학습 명령과 정규화·검사 방법은 [baseline 가이드](pi05_tacdex_baseline.md)를
따른다. 처음에는 컴파일 시간을 별도로 보고, 이후 `data_wait_seconds`와
학습 `seconds`를 비교한다.

## 6. 이후 데이터 변환 시 적용할 원칙

1. LeRobot 기본 인코딩 함수를 사용하고, 사용한 라이브러리 revision과 인자를 남긴다.
   직접 인코딩이 필요하면 GOP를 명시한다.
2. LeRobot 메타데이터 형식뿐 아니라 실제 출력 MP4의 키프레임 간격과 시간축을 검증한다.
3. GPU 학습 전에 실제 transform을 포함한 랜덤 배치 로딩 속도를 CPU에서 확인한다.
4. 워커·prefetch 확대는 디코딩 비용을 확인한 뒤 적용한다. 큐를 늘리는 것만으로
   지속적인 데이터 공급 속도가 개선되지는 않는다.

## 근거 파일

이 문서에 주요 수치를 포함했으므로 원시 실험 파일 없이도 문제와 조치를 파악할 수 있다.
아래 파일들은 현재 서버의 로컬 근거이며, `datasets/`와 `experiments/`는 Git에서
제외돼 있다. 프로젝트 외부나 다른 checkout에 자동으로 전달되지는 않는다.

| 근거 | 저장소 루트 기준 경로 |
| --- | --- |
| 초기 샘플 프로파일 | `experiments/pi05_tacdex/debug/data_loading_analysis/samples_profile.txt` |
| 초기 배치 대기 | `experiments/pi05_tacdex/debug/data_loading_analysis/result_workers2_partial.json` |
| 변환 전후 비교 | `experiments/pi05_tacdex/debug/default_encoding/comparison.json` |
| CPU 로더·모의 소비 실험 | `experiments/pi05_tacdex/debug/default_encoding/benchmark.json` |
| 실제 JAX 로더 검증 | `experiments/pi05_tacdex/debug/default_encoding/smoke.json` |
| 전체 영상 변환 기록 | `experiments/pi05_tacdex/debug/default_encoding/conversion.json` |
| 원본 변환 코드 스냅샷 | `experiments/trlt/3dtacdex_assembly/runs/assembly_8k_fullb64_wandb10_seed0_20261118_prepared/source_snapshot/src/openpi/trlt/tacdex_lerobot.py` |
