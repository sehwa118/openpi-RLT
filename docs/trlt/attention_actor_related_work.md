# 어텐션 액터(Attention Actor) 선행 연구 정리 및 요약본

- **작성일 / 문헌 기준일:** 2026-11-21
- **대상 브랜치:** `attention-actor` (`openpi-RLT`)
- **관련 내부 문서:** [`attention_actor.md`](./attention_actor.md), [`attention_actor_results_2026-11-20.md`](./attention_actor_results_2026-11-20.md)

이 문서는 **Body-Part Attention Actor**(`PartAttentionActionProbe` 및 Stage 2 `TouchPartAttentionActionProbe`)와 직결되는 핵심 선행 논문 3편을 정리해, ① **발표·논문 작성에 바로 활용할 수 있는 선행 연구(Related Work) 통합 서술 파트**와 ② **논문별 상세 요약본**을 함께 제공합니다.

---

## 1. 선행 연구(Related Work) 파트 초안

고자유도 양손·다지 로봇 조작(예: `58D` T-Rex 양손 제어)을 위한 정책 아키텍처는 크게 **(1) 로봇 신체 구조를 반영한 어텐션 토폴로지 설계**, **(2) 로봇 제어 Transformer의 안정적 조건화(`AdaLN-Zero`)**, **(3) 양손·다지 환경의 시각–촉각–손가락 단위 토큰 사전학습**의 세 축에서 연구되어 왔다.

### (1) 신체 구조 기반 토큰화 및 마스크드 어텐션
초기 Transformer 기반 제어 정책은 로봇의 관절과 센서 상태를 평면적인 시퀀스나 단일 벡터로 처리해 신체 기구학(kinematic topology)의 귀납적 편향을 잃는 한계가 있었다. **Body Transformer (`BoT`)**(Sferrazza et al., CoRL 2024)는 로봇 신체를 센서·액추에이터 노드와 물리적 연결 그래프($M = I + A$)로 정의하고, 어텐션 마스크를 통해 각 부위가 인접한 이웃 부위와 우선적으로 상호작용하도록 제한했다. 동시에 모든 층을 강하게 마스킹(`BoT-Hard`)할 때 발생하는 원거리 부위 간 전역 협응 병목을 완화하기 위해, 마스크 층과 비마스크(전역) 층을 교차 배치하는 `BoT-Mix`를 제안했다. 우리 Stage B `PartAttentionActionProbe` 역시 `58D` 관절 상태를 12개 신체 부위 토큰(`TREX_PARTS`)으로 분해하고 `kinematic`/`hierarchical`/`full` 마스크를 체계적으로 비교함으로써 이 계보를 잇는다. 실제로 `2026-11-20` `seed 0` 실험에서도 `kinematic`은 `arms` 오차(`0.001481`)에서, `hierarchical`은 손내부 협응이 허용된 `hands` 오차(`0.000328`)에서 각각 최상위 성능을 보였고, 팔←손가락 상향 피드백과 양손 협응이 모두 열린 `full`이 종합 1위(`-46.56%` vs `mlp`, `-45.16%` vs 동등 용량 `resmlp`)를 기록하여 `BoT-Mix`류 하이브리드 마스크 설계의 직접적 근거를 제공했다.

### (2) 로봇 Transformer 정책의 `AdaLN-Zero` 조건화
고차원 로봇 조작에서 Transformer에 외부 관측·시간·목표 조건을 주입할 때, 표준 Cross-Attention 기반 블록은 학습 초기 그래디언트 불안정이나 붕괴에 취약한 경향이 있다. **DiT-Block Policy**(*The Ingredients for Robotic Diffusion Transformers*, Dasari et al., 2024)는 확산 Transformer(DiT)의 **0-초기화 적응형 레이어 정규화(`adaLN-Zero`)** 블록을 양손 ALOHA 장기 조작 태스크에 도입해 별도의 까다로운 튜닝 없이도 학습 안정성과 스케일링 성능을 크게 개선했다. 우리 Stage B 어텐션 액터는 이 `AdaLN-Zero` 메커니즘을 확산 모델이 아닌 **경량(`~0.76M`) 결정론적 `H=6` 액션 청킹 액터**에 적용해 고빈도 제어에 맞게 경량화했으며, 특히 조건 벡터 `cond = MLP([stop_gradient(z), visual_age_seconds])`에서 `proprioception`을 의도적으로 제외함으로써 전역 `AdaLN` 경로를 통한 부위 마스크 격리 누출까지 구조적으로 방지한다.

### (3) 양손·다지 시각–촉각–손가락 단위 토큰 사전학습
시각 정보만으로는 접촉 힘과 미끄러짐을 구분하기 어려운 양손 다지 조작에서는 시각·촉각·행동 표현을 정렬하는 사전학습이 중요하다. **VTAO-BiManip**(Sun et al., 2025)은 1인칭 RGB, 20포인트 촉각 장갑 데이터, 모션 캡처 기반 6-DoF 물체 자세·손 궤적을 결합한 **Masked Visual-Tactile-Action Autoencoder**를 제안하고, 양손 각 **손가락/관절 단위 액션 토큰**으로 분할해 미래 행동과 물체 속성을 마스킹 복원함으로써 양손 병뚜껑 열기(Isaac Gym 및 실기)에서 큰 폭의 성공률 향상을 입증했다. 우리 연구는 모션 캡처 기반의 별도 물체 6-DoF 자세 라벨 없이도 Stage A(`TRLTEncoder`)에서 `pi0.5` 시각 특징과 10개 손가락 `64×64` 촉각 변형 이력을 `512D` 잠재 표현 `z`로 동시 오토인코딩하고, Stage B의 부위 토큰화와 Stage 2 `TouchPartAttentionActionProbe`(`PerFingerTouchCrossAttention`)를 통해 손가락별 고해상도 촉각(`LingBot-Touch`)을 대응 손가락 토큰에 직접 라우팅한다는 점에서 차별화된다.

---

### 한눈에 보는 3편 vs 우리 연구 비교표

| 구분 | 1. Body Transformer (`BoT`) | 2. DiT-Block Policy | 3. VTAO-BiManip | **우리 연구 (`openpi-RLT`, `attention-actor`)** |
| --- | --- | --- | --- | --- |
| **핵심 문제** | 평면 정책의 신체 기구학 구조 손실 | 로봇 Transformer 조건화·학습 불안정 | 고자유도 양손·촉각–행동 정렬 및 탐색 난제 | 동결 시각-촉각 표현(`z`) 기반 고빈도 `58D`/`22D` 정밀 제어 |
| **토큰 단위** | 센서·액추에이터 그래프 노드 | 카메라별 ResNet 시각 토큰 + 상태·노이즈 액션 | 시각 + 촉각 + 양손 손가락/관절 액션 토큰 | `PartSpec` 신체 부위 토큰 (`58D→12`, `22D→5`) |
| **조건 주입 방식** | 노드 토큰 내 상태 결합 + 그래프 마스크 | **DiT `adaLN-Zero`** + FiLM(DistilBERT 언어) | Masked Autoencoder 교차 모달 Transformer | **`AdaLN-Zero` (`z, age`, proprio 제외)** + `[N, N]` 부위 마스크 |
| **촉각 활용** | 미포함 (고유수용감각 중심) | 미포함 (다중 카메라 시각 중심) | 촉각 장갑 + 모션캡처(손·물체 6-DoF) | Stage A(10지 `64×64` 복원) → Stage 2 **손가락별 `LingBot-Touch` Cross-Attn** |
| **학습 패러다임** | IL(MoCapAct) · RL(Isaac Gym PPO) | Diffusion Policy (양손 ALOHA BC) | Masked Pretraining + 2단계 Curriculum RL | Stage A 자기지도 오토인코더(동결) → Stage B 경량 청킹(`~0.76M`, BC/PPO 확장 가능) |

---

## 2. 핵심 선행 논문 3편 상세 요약본

### 논문 1. Body Transformer: Leveraging Robot Embodiment for Policy Learning (`BoT`)
- **서지 정보:** Carmelo Sferrazza, Dun-Ming Huang, Fangchen Liu, Jongmin Lee, Pieter Abbeel (UC Berkeley) · **CoRL 2024** · [arXiv:2408.06316](https://arxiv.org/abs/2408.06316)
- **핵심 아이디어:**
  1. **로봇 신체의 그래프 표현:** 로봇의 센서와 액추에이터를 그래프의 노드(node)로, 관절과 링크의 물리적 기구학 연결을 엣지(edge)로 모델링한다.
  2. **형태 유도 마스킹 어텐션(`BoT-Hard`):** 인접 행렬 $A$와 항등 행렬 $I$의 조합($M = I + A$)으로 어텐션 마스크를 구성해, 각 노드가 자기 자신과 물리적 1-hop 이웃 노드만 참조하도록 제한한다. 이는 국소 외란에 먼저 주변 관절이 반응한 뒤 깊은 층을 거쳐 전역으로 전파되는 생물학적 반사 신경과 유사한 귀납적 편향을 부여한다.
  3. **`BoT-Mix` (마스크 층 + 비마스크 층 교차):** 순수 국소 마스킹(`BoT-Hard`)만 사용하면 원거리 부위(예: 발끝 충격과 손끝 균형) 간 빠른 전역 협응이 병목에 걸릴 수 있음을 확인하고, 마스크 층과 비마스크(`full`) 전역 어텐션 층을 번갈아 배치한 `BoT-Mix`를 제시한다.
- **실험 환경 및 결과:**
  - 모방 학습(MoCapAct 모션 캡처 추적)과 강화학습(Isaac Gym **PPO**: `Humanoid-Mod`, `Humanoid-Board`, `Humanoid-Hill`, Unitree `A1-Walk`)에서 MLP와 기본(Vanilla) Transformer를 비교했다.
  - 특히 정밀 균형과 크레딧 할당(credit assignment)이 까다로운 고난도 탐색 태스크(`Humanoid-Board`, `Humanoid-Hill`)에서 Vanilla Transformer는 학습이 불안정했던 반면, `BoT`/`BoT-Mix`는 더 높은 샘플 효율과 최종 성능을 보였으며 구조적 마스크의 정규화 효과 덕분에 파라미터 확장 시 과적합에도 강했다.
- **우리 연구 시사점:**
  - `kinematic` 마스크가 `arms` 오차(`0.001481`) 1등을, `hierarchical` 마스크가 `hands` 오차(`0.000328`) 1등을, `full`이 종합 1등(`-46.56%`)을 기록한 어제 실험 현상을 가장 직접적으로 설명하는 문헌이다. **1층 국소 마스크(`kinematic` 또는 `hierarchical`) + 2층 `full` 교대 (`BoT-Mix` 변형)**가 자연스러운 후속 비교군으로 도출된다.

---

### 논문 2. The Ingredients for Robotic Diffusion Transformers (`DiT-Block Policy`)
- **서지 정보:** Sudeep Dasari, Oier Mees, Sebastian Zhao, Mohan Kumar Srirama, Sergey Levine (CMU & UC Berkeley) · [arXiv:2410.10088](https://arxiv.org/abs/2410.10088) · [프로젝트 페이지](https://dit-policy.github.io) · [GitHub](https://github.com/SudeepDasari/dit-policy)
- **핵심 아이디어:**
  1. **Cross-Attention의 불안정성과 `adaLN-Zero` 채택:** 로봇 확산 정책(Diffusion Policy)의 백본을 CNN U-Net에서 대용량 Transformer로 바꿀 때 조건 주입에 표준 Cross-Attention을 그대로 쓰면 학습이 붕괴되거나 세팅별 하이퍼파라미터에 극도로 민감해지는 문제를 짚는다. 이를 해결하기 위해 이미지 생성 DiT의 **`adaLN-Zero`** 블록(조건 벡터와 확산 스텝으로 LayerNorm의 scale/shift를 변조하고 출력 투영·게이트를 0 초기화)을 로봇 정책에 이식했다.
  2. **시각–언어 입력 토크나이저:** 순수 ViT 패치 입력으로 CNN을 완전히 대체하기보다 카메라 시점마다 분리된 CNN 인코더(ResNet-18/26 등)로 공간 사전(spatial prior)을 보존하고, 언어 지시는 사전학습된 DistilBERT 임베딩을 **FiLM**으로 시각 토큰에 결합한다.
- **실험 환경 및 결과:**
  - 접촉이 많고 호흡이 긴(`1,500+` 타임스텝) **양손 ALOHA(Bi-manual ALOHA)** 정밀 조작(다단계 조립, 스시 자르기, 펜 뚜껑 열기 등) 및 10시간 분량의 대규모 다중 모달·언어 데이터셋(*BiPlay*)에서 평가했다.
  - 기존 U-Net Diffusion Policy 및 단순 Cross-Attention Transformer를 크게 앞섰으며, 절제 실험(ablation)을 통해 **`adaLN-Zero` 교체가 학습 붕괴 제거와 대규모 다중 모달 스케일링에 핵심적인 요소**임을 확인했다.
- **우리 연구 시사점:**
  - 우리 `AdaLNTransformerBlock`([`src/openpi/models/trlt_actor.py`](../../src/openpi/models/trlt_actor.py#L132-L176))이 `adaln_modulation` 및 `final_adaln`을 0으로 초기화하고(`alpha = 0`) Step 1 항등 매핑에서 점진적으로 학습을 여는 설계 근거의 로봇 제어 직계 문헌이다. 우리는 이를 무거운 확산 디노이징 대신 **동결 Stage A 표현 `z` 위의 경량(`~0.76M`) 결정론적 청킹 액터**로 경량화해 30Hz 및 고빈도 폐루프 제어에 맞췄다.

---

### 논문 3. Masked Visual-Tactile-Action Pre-training with Object Understanding for Bimanual Dexterous Manipulation (`VTAO-BiManip`)
- **서지 정보:** Zhengnan Sun, Zhaotai Shi, Jiayin Chen, Qingtao Liu, Yu Cui, Qi Ye, Jiming Chen · [arXiv:2501.03606](https://arxiv.org/abs/2501.03606)
- **핵심 아이디어:**
  1. **다중 모달 데이터 수집(VTAO):** 1인칭 RGB 카메라(Microsoft HoloLens 2), 손 동작·접촉용 20포인트 촉각 장갑, 정밀 손 궤적 및 물체 6-DoF 자세를 기록하는 모션 캡처(ChingMu)를 결합해 인간 수준의 양손 조작 데이터를 구축했다.
  2. **Masked Visual-Tactile-Action Autoencoder:** Transformer 기반 오토인코더가 시각·촉각·연속 손 동작 궤적 토큰을 입력으로 받으며, 특히 양손 액션을 **손 단위(per-hand)가 아닌 개별 손가락/관절 단위(per-finger/per-joint) 액션 토큰**으로 세분화해 융합 성능을 높인다.
  3. **물체 이해(Object Understanding) + 커리큘럼 RL:** 입력 일부를 마스킹하고 누락된 시각/촉각 정보와 미래 액션뿐 아니라 물체 속성(6-DoF 자세·크기)을 함께 예측하도록 정규화한 뒤, 사전학습된 표현 위에서 **2단계 커리큘럼 강화학습(Curriculum RL)**으로 접근·파지·비틀기·당기기 하위 기술을 안정적으로 연결한다.
- **실험 환경 및 결과:**
  - 양손 정밀 협응·힘 제어·다단계 기술 연쇄가 동시에 요구되는 **양손 병뚜껑 열기(bimanual bottle-cap unscrewing)** 태스크를 Isaac Gym(양손 Leap Hand + Unitree Z1 팔) 및 실제 로봇 하드웨어에서 검증했다.
  - 기존 시각 전용 또는 기본 시각-촉각 사전학습 베이스라인 대비 **성공률 20% 이상 향상**과 양손 동기화·sim-to-real 전이 성능 향상을 보고했다.
- **우리 연구 시사점:**
  - Stage A(시각+촉각 오토인코더 사전학습)와 Stage B(양손 손가락 단위 토큰 정책)라는 전체 파이프라인 관점에서 가장 가까운 최신 연구다. 반면 우리는 모션 캡처 기반의 별도 물체 6-DoF 자세 라벨 없이 `pi0.5` + 10지 `64×64` 촉각 변형 이력만으로 `z`를 학습하고, Stage 2에서는 `LingBot-Touch` 기반 손가락별 Cross-Attention(`PerFingerTouchCrossAttention`)으로 센서 해상도·포터빌리티를 직접 확장할 수 있다.

---

## 3. 선행 연구 대비 우리 연구 포지셔닝 & 후속 실험 연결

1. **확실히 입증된 점 (`2026-11-20` `seed 0`):**
   - `DiT-Block Policy` 스타일 `AdaLN-Zero` + `Body Transformer` 스타일 부위 토큰화 조합만으로, **파라미터 수가 동일한(`~0.76M`) 평면 `ResMLP` 대비 `-34.64% ~ -45.16%`** (`mlp` 대비 `-36.32% ~ -46.56%`, `first_action_mse -61.2%`) 성능 향상을 확인했다.
2. **문헌에 기반한 즉시 테스트 가능한 후속 아이디어:**
   - **`BoT-Mix` 구조 (`1층 hierarchical 또는 kinematic + 2층 full`):** `hierarchical`이 보여준 손가락 강점(`hands: 0.000328`)과 `full`이 보여준 양손·팔 강점(`arms: 0.001502`)을 단일 스윕으로 통합 검증할 수 있다.
   - **Per-Head Masking (`[4, 12, 12]`):** 4개 어텐션 헤드에 각각 기구학 / 손내부 파지 / 양손 대칭 / 전역(`full`) 마스크를 배정하면, 추가 파라미터 `0`개로 마스크 간 트레이드오프를 해소할 수 있다.

---

## 4. 참고 문헌 링크 (검증 기준일: 2026-11-21)

1. Sferrazza et al., *"Body Transformer: Leveraging Robot Embodiment for Policy Learning"*, CoRL 2024 — https://arxiv.org/abs/2408.06316
2. Dasari et al., *"The Ingredients for Robotic Diffusion Transformers"*, 2024 — https://arxiv.org/abs/2410.10088 · https://dit-policy.github.io · https://github.com/SudeepDasari/dit-policy
3. Sun et al., *"Masked Visual-Tactile-Action Pre-training with Object Understanding for Bimanual Dexterous Manipulation"*, 2025 — https://arxiv.org/abs/2501.03606
