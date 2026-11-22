# LingBot-Touch-Tiny 통합 계획 — 2026-11-22

이 문서는 최초 다운로드·설계와 후속 선택의 기록이다. 후속 요청으로 양 브랜치 구현이
완료되었으며, 현재 준비 상태·실행 명령은 각 브랜치의 LingBot 사용 문서와
`experiments/trlt/lingbot_touch_validation/`의 실제 보고서를 참조한다.
장기 Stage A/B 및 최종 test 평가는 시작하지 않는다.
사용자 확정 사항: **frozen encoder + feature cache**, taxel mean pooling,
시간축 **고정 지수 감쇠 pooling**, fusion에 센서당 한 토큰.
추가 확정: T-Rex96points; PaXini tip/pulp를 포함한8pad 공동 encoding; PaXini의 XYZ 모두
`clip(raw/30,-1,1)` 그대로 사용(Z도 signed 유지).

## 확보한 소스와 실제 확인

- 작업 checkout: `TRLT`, `497e08b49f6ad9fe56d6f248f2f1ba4e4c410ca3`; 시작 시 clean.
- [공식 소스](https://github.com/robbyant/lingbot-touch):
  `third_party/lingbot-touch/`, commit `dec8025b212e7099b235cc2ef13a4921a7e59545`.
  nested clone은 Git ignore하며 upstream 라이선스/notice를 보존한다.
- [tiny bundle](https://huggingface.co/robbyant/lingbot-touch-tiny):
  `checkpoints/lingbot-touch-tiny/`, HF revision `f23969ccd4b6bba2163e69cc05a577455a075fe9`.
  config + 실제 encoder.pt(21,859,018 bytes)와 README/LICENSE 다운로드 완료.
  체크포인트 SHA256: `ddb8b435963348fff5b8a83bb48f995978dc8841d82f87be007c775bf5a88055`.
- 실제 config: depth12, hidden192, heads3, tubelet2. layer2/5/8/11의
  normalized outputs를 concatenate하므로 **최종 feature768D**. frozen parameters5,455,488.
- API: `signal[B,T,5,M,3]`, canonical `pos_xyz[5,M,3]` 또는 `[B,5,M,3]`,
  공유 spatial coverage `[5,M]`. `T`는2의 배수. 출력 `[B,(T/2)*valid_points,768]`.
  time/finger/XYZ 주소가 동반된다. `encode()`는 no_grad, encoder는 항상 eval/frozen이다.
- upstream의 Python3.12/Torch2.11/NumPy2.4 요구사항을 프로젝트에 설치하지 않았다.
  기존 `sehwa` + `uv run --no-sync` Python3.11.18/Torch2.7.1/NumPy1.26.4에서
  direct source import, strict weight load 및 아래 제한된 CPU 추론은 통과했다.
  이것이 upstream 전체 테스트/모든 실행환경의 호환성을 의미하지 않는다.
- [CPU 증거](../../experiments/trlt/lingbot_touch_planning/cpu_smoke.json):
  upstream 좌/우 Sharpa 샘플 각각 첫2프레임 `[1,2,5,200,3]` → `[1,1000,768]`
  → finger mean `[1,5,768]`; synthetic native240/T10/M32 sampler+encode;
  synthetic four-finger/T4/M15 coverage; 홀수T3 거부. 모두 PASS, CPU only.
  HF 파일 목록·크기·revision은 같은 폴더 `hf_metadata.json`, bundle SHA256은 cpu_smoke.json.
  실제 T-Rex·PaXini 입력의 의미/geometry 일치를 검증한 결과는 아니다.

## 현재 코드와 변경 범위

| 데이터 | 확인한 기존 경로 | 계획한 새 fusion 입력 |
|---|---|---|
| T-Rex | 현재 브랜치 `TactileCNN`, history10 × sensor10 =100 tactile tokens; self-only4 layers | sensor10 tokens, self-only4 layers 유지 |
| 3DTac-Dex Assembly | 로컬 `TRLT-3DTacDex` commit `9ddbd63a14ceeae95fc7585fdfdb1a79dde0ee25`; `TactileTaxelEncoder`: ordered45D→512→512; history3 × pad8=24 tokens | pad8 tokens, self-only2 layers |

3DTac 코드는 `git show`로 확인했으며 checkout 전환/merge는 하지 않았다.
구현 시 두 데이터 경로가 공존하도록 해당 브랜치의 필요한 공통 확장을 통합하거나
그 브랜치 기반 별도 checkout에서 작업한다. 무관한 변경을 일괄 merge하지 않는다.
현재 `self_only`는 V+tactile+state의 전체 self-attention이다. 모달리티를 차단하지 않는다.

## pooling과 캐시 계약

1. query 시점까지의 과거 window만 encoder에 넣는다. full episode를 한 번 encode한 뒤
   query별로 자르는 방식은 사용하지 않는다. upstream attention은 noncausal이다.
2. taxel별 출력의 원래 sensor/pad membership을 유지한다. 각 tubelet·sensor의 유효 taxel만
   mean하여 `features[B,K,S,768]`, 시간·valid/count를 캐시한다. 전체768D를 사용한다.
   주소에 pad ID가 없으므로 compact spatial order와 별도 pad-index map을 저장한다.
3. 시간 pooling은 `h_s = sum_k w_ks * features_ks`,
   `w_ks = valid_ks * exp((u_ks - t)/tau) / sum_j(valid_js * exp((u_js-t)/tau))`.
   `u`는 tubelet의 실제 관측 timestamp 평균, `t`는 query 시간이며 미래는 허용하지 않는다.
   valid가 전부 false면 token0 및 sensor-valid=false. 안정적인 masked softmax로 계산한다.
4. tau는 초 단위의 고정 config 값. **사용자 채택값**: T-Rex0.15초, Assembly0.20초
   (기존 nominal history span의 약 절반). test로 조정하지 않는다.
   tau는 채택한 고정값이며 검증된 최적값을 의미하지 않는다.
5. pooled768D → trainable Dense512; sensor ID와 sensor별 weighted relative-time embedding을
   더해 fusion으로 보낸다. 기존 dataset별 modality embedding 정책은 유지한다.
   cached tubelet/time mask와 decoder용 원래 history mask를 구분한다.
   기존 `length*S` flatten과 validity 결합을 별도 encoder 분기로 대체해야 한다.
6. encoder 내부에서 이미 모든 유효 taxel/time 문맥이 섞인다. 주소별 pooling은 정보의
   출처를 완전히 분리하지 않으며, 지수 감쇠도 내부 과거 영향까지 제한하지 않는다.
   [upstream](https://github.com/robbyant/lingbot-touch#usage-of-lingbot-touch-representations)은
   attention querying을 권장하지만 이번 주 실험은 사용자가 정한 mean+time-weighted 방식이다.
   attention readout은 별도 후속 ablation으로 남긴다.
7. 캐시 key는 **query/window 단위**다. source revision/hash, 원본 frame IDs/timestamps/masks,
   split/protocol, modality, geometry·sampling·axis/scale/padding spec, encoder config/weight hash,
   source commit, dtype와 pooling stage를 포함한다. trainable projection 출력은 캐시하지 않는다.
   tau는 tubelet cache identity에는 불필요하지만 Stage A/latent/checkpoint identity에는 포함한다.

## T-Rex adapter

- 기존 64×64 CNN 전처리 cache를 LingBot 입력으로 쓰지 않는다. native240×240 Y-plane
  luma codes에서 **좌우 각각5손가락**으로 구성하고, 공식
  `prepare_pretraining_reference_input(..., layout_id="visuotactile_deformation240_canonical_right_v1", num_points=96)`
  경로를 우선 검토한다. 지원 점 수는200/96/32이며 **사용자 선택은96**이다.
- scalar를 `[0,0,Z/255]`, modality=`visuotactile`로 변환한다. 중복 /255 금지.
  sampling 전/후 resize는 공식 mapping과 달라지므로 하지 않는다.
- **필수 검증**: T-Rex native 픽셀 방향/순서와 공식 canonical-right map의 대응,
  왼손의 필요한 reflection/permutation. shape240 일치만으로 layout ID를 선언하지 않는다.
  기존 hand-root pad-center FK cache는 taxel-level canonical geometry가 아니므로 대체재가 아니다.
- history10 → tubelet5, hand별 encoder output `[B,2400,768]` (M96),
  두 손을 결합한 tubelet-pooled cache `[B,5,10,768]`, 최종 `[B,10,512]`.
  비용이 크므로 encoder batch는 fusion batch와 분리하고 나중에 측정한다.
- Stage A tactile target은 기존64×64 history reconstruction을 유지한다.
  encoder의 native sampled 입력과 decoder target은 별도 전처리 경로로 관리한다.

## 3DTac-Dex / PaXini 조사와 adapter

- 사용자 제공 모델: tip=`PX6AX-GEN1-DP-S2716`, pulp=`PX6AX-GEN1-IP-S2516`.
  [공식 제품](https://mall.paxini.com/product/itpu/66f29633a39ea3c8f1b5425a) 및
  [dataset README](https://huggingface.co/datasets/kingchou007/3dtacdex/blob/main/README.md)와 일치한다.
  dataset의 source axis/sign은 보존되어 있고 Newton calibration은 보장되지 않는다.
- [원저자 좌표 코드](https://github.com/tianhaowuhz/3dtacdex/blob/e3177b30be2a60568d1426ce2c588e8c496652f4/diffusion_policy/real_world/fk/tactile.py)에
  `PAXINI_{TIP,PULP}_ORI_COORDS` 각15×3 및 좌표 변환이 있다.
  [constants.py](https://github.com/tianhaowuhz/3dtacdex/blob/e3177b30be2a60568d1426ce2c588e8c496652f4/diffusion_policy/real_world/fk/constants.py)는
  링크 좌표(미터)와 mount offsets를 제공한다. 순서/축 매핑의 출발점이며 LingBot calibration은 아니다.
- [force 전처리](https://github.com/tianhaowuhz/3dtacdex/blob/e3177b30be2a60568d1426ce2c588e8c496652f4/diffusion_policy/real_world/data_preprocess/tactile_processer.py)는
  sensor→link에서 X/Z 교환을 사용한다. 이 변환을 canonical LingBot 축 변환으로 그대로 간주하지 않는다.
  원 데이터 pad 순서와 좌표 순서를 대조하고 pad별 접촉 normal 방향을 검증한다.
- **최종 사용자 확정: tip/pulp를 포함한8개 pad 공동 encoding**. 실제 손가락4개에
  tip15+pulp15를 배치하여 M30으로 한 번 encode하고 pinky는 coverage=false로 둔다.
  모든8pad 사이에 encoder attention이 있으며, taxel mean은 pad별로 구분하여8개 token을 만든다.
  입력 순서와 별도 pad membership map을 고정하여 tip/pulp가 pooling에서 합쳐지지 않게 한다.
  M30은 API가 지원하나 pretrained magnetic 해상도 목록 밖이다.
- **사용자 확정 normalization**: encoder 입력과 기존 reconstruction target 모두
  XYZ에 `clip(raw/30,-1,1)`을 적용하며 Z도 signed로 보존한다. 추가 /255,
  abs(Z), (Z+1)/2 또는 Z의[0,1] clipping을 적용하지 않는다.
  공식 `prepare_sensor_input`은 Z를[0,1]로 clip하므로 이 helper를 사용하지 않는다.
  프로젝트 adapter가 float32 signal/coverage를 준비해 low-level `encoder.encode`를 호출한다.
  현재 low-level validation은 shape/device/finite를 검사하며 signed Z를 거부하거나 clip하지 않는다.
  이 경로는 코드 검토 결과이며 아직 해당 signed 입력의 실제 추론을 실행하지 않았다.
  공식 pretrained signal 계약(Z[0,1])과의 분포 차이는 실험 provenance에 명시한다.
- PaXini tokenizer는 signed3축을 보존하는 `magnetic` route를 구현 후보로 둔다.
  이 route 사용 자체가 실제 GEN1 센서의 물리적 sensing principle 확인을 뜻하지 않는다.
  scalar piezoresistive route는 XY를 버리므로 이번 signed XYZ 실험에는 사용하지 않는다.
  canonical taxel 좌표/배치 및 원본 축의 의미는 공개 좌표 코드와 train/validation으로
  계속 검증한다. 고정 normalization은 검증 결과를 핑계로 자동 변경하지 않는다.
- history3는 tubelet2와 맞지 않는다. **사용자 채택**: oldest valid frame을 앞에 한 번 복제하여4프레임
  (정상 window `[t-2,t-2,t-1,t]`), 원래3 history와 target/mask는 유지한다.
  인위적 frame은 관측 count/loss에 포함하지 않으며 실제 timestamp/복제 flag를 보존한다.
  후속 weighted pooling에서 tubelet별 실제 관측 기여도를 명시한다.
  시간 tokenization 변경과 prefix 복제 정책을 config/provenance에 고정한다.
- upstream coverage는 공간축 공유 mask뿐이며 batch/time별 missing mask를 attention에 주지 못한다.
  segment 시작/불연속/부분 pad 결측에는 causal valid suffix별 encode/bucketing 또는 명시적
  과거 관측 padding 정책이 필요하다. 단순0 padding을 valid 관측으로 encode하지 않는다.
  padding이 encoder context에 영향을 줄 수 있음을 기록하고, 관측이 없으면 출력 sensor를 무효화한다.
  새로운 frame이나 segment 경계를 넘어 과거를 가져오지 않는다.

## 구현 순서와 검증

1. source/weight pin 및 센서 adapter spec부터 작성한다. PaXini 좌표·순서·축·normalization과
   T-Rex native layout 검증은 train/validation만 사용한다. 미확인 항목은 명시적으로 거부한다.
2. `src/openpi/trlt/lingbot_touch.py`(예정): optional source import, frozen CPU/GPU 추출,
   query-local causal windows, modality adapter, taxel mean 및 tubelet cache manifest/reader.
   `scripts/cache_lingbot_touch_features.py`(예정): plan-only와 split별 명시적 추출 CLI.
3. `config.py`, `trlt.py`, data/loading/runner에 optional tactile encoder 분기와 시간 pooling,
   768→512 projection을 연결한다. 기존 기본값/parameter tree/config identity는 보존한다.
   decoder와 reconstruction loss는 기존 dataset 계약을 유지한다.
4. 별도 LingBot 설정 두 개: T-Rex self4 / Assembly self2. seed0를 초기 제안으로,
   Stage A 예산은 각각 기존15k/batch32,8k/batch64를 초기 제안으로 유지한다.
   정확한 microbatch와 extraction batch는 추후 bounded 검증으로 결정한다.
   Stage B는 기존256×2 MLP/AdamW1e-3/batch256/max100epochs/patience10.
5. CPU 검사: pooling 수식·실제 시간·불균등 taxel 수·all-invalid/zero-contact·padding·미래
   누출·finger/pad 순서·wrong geometry/hash 거부·동기/worker cache 동일성·old config 호환성.
   축/부호 변환과 실제 train sample을 육안/수치 검증한다. reduced synthetic A/B pipeline으로
   frozen weight 불변성과 projection gradient, resume/cache provenance를 확인한다.
6. 이후 별도 실행 결정 후 bounded GPU/extraction 검증 → train/validation tactile cache →
   Stage A best selection → frozen train/validation latent export → Stage B best selection →
   selections seal → test cache/export/evaluate. 기존 visual cache/split/normalization을 재사용한다.
   2026-11-22 후속 요청으로 구현과 학습 전 준비 및 GPU 검증이 허용되었다.
   본 benchmark 학습 큐/Stage A/B 장기 학습과 최종 test 평가는 시작하지 않는다.
7. primary: T-Rex episode-macro normalized action MSE; Assembly 원본 trajectory-macro MSE.
   token 수, pretrained/frozen 여부, pooling이 모두 바뀌므로 encoder architecture만의 순수 ablation은
   아니다. 필요하면 동일 sensor-token 수로 scratch baseline을 추가한다. 이미 확인한 test 결과를
   새 설정 선택에 사용하지 않는다. upstream pretraining과 평가 데이터의 중복 여부는 아직 미검증이다.

## 남은 결정 / 확인

확정: frozen/cache, taxel mean, 고정 exponential 시간 가중치, sensor당1token,
T-Rex self-only / Assembly self-only2layer.

추가 확정(최신 요청): T-Rex96points, PaXini8pad 공동 encoding, XYZ signed raw_clip30,
tau0.15/0.20초 및 Assembly3→4 causal prefix duplication. seed0,
T-Rex Stage A15k/batch32 및 Assembly8k/batch64, Stage B 기존 설정 유지.

사용자는 TRLT/T-Rex와 TRLT-3DTacDex/Assembly를 각각 multi-agent로 구현하고,
학습 전 준비를 완료하도록 요청했으며 GPU 검증을 허용했다. 별도 worktree로 두 브랜치를
보존하며 본 학습은 시작하지 않는다. 실행 상태와 증거는 후속 구현 문서를 참조한다.
센서 좌표·축 검증은 공개 원저자 코드를 이용하며, 확인되지 않은 calibration은
검증 완료 사실로 기록하지 않는다.
