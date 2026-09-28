# TPSC: Tiny-aware Prototype Semantic Calibration 설계

## 1. 목적과 연구 질문

TPSC는 COXNet의 CLFM을 완전히 제거한 same-stage FPN에서, RGB와 Thermal의
target-relevant semantics를 compact prototype으로 요약하고 그 관계로 **RGB 채널만**
보정하는 pre-AAM calibration 모듈이다.

```text
same-stage RGB/Thermal P3,P4
  -> detail-preserving cross-scale descriptors
  -> shared-slot modality prototypes
  -> prototype-set relation mixing
  -> bounded channel modulation on RGB P3,P4 only
  -> original AAM/HOFM(calibrated RGB, unchanged Thermal)
```

검증할 연구 질문은 다음과 같다.

> 밀집된 full-resolution cross-modal interaction이나 위치별 hard candidate matching 대신,
> 객체 중심의 작은 semantic prototype 집합에서 모달·scale 관계를 학습한 뒤 RGB의
> 채널 표현만 조절하면 배경 간섭을 줄이면서 RGBT tiny-person detection을 개선하는가?

TPSC는 ProtoHGF-Net의 prototype-level interaction과 dense-feature modulation 원리를
참고하지만, 해당 구현을 복사하거나 전체 구조를 이식하지 않는다. COXNet의 AAM과
역할이 겹치는 modality fusion, 양쪽 모달 보정, teacher-mask distillation은 첫 버전에서
제외한다.

## 2. 범위와 불변 조건

### 2.1 유지하는 COXNet 구성

- RGB와 Thermal backbone은 기존 dual ResNet-50을 유지한다.
- RGB/Thermal FPN 모두 `start_level=1`로 P3/P4/P5/P6를 출력한다.
- 기존 AAM/HOFM, DSR, GFL head, QLSAssigner와 test-time NMS를 변경하지 않는다.
- Thermal feature tensor와 좌표는 TPSC에서 변경하지 않는다.
- TPSC는 강화된 RGB와 원본 Thermal을 기존 AAM/HOFM에 전달한다.
- 첫 통제 실험은 원 COXNet recipe와 맞추기 위해 `wf_loss=True`,
  `wf_loss_mode='kl_v2'`, `wf_loss_weight=0.1`을 유지한다.

### 2.2 제거하는 CLFM 구성

- RGB/Thermal cross-stage pairing
- DeConv resolution matching
- Haar DWT/IDWT와 LL/high-frequency fusion
- CLFM parameter 복사와 legacy initialization 경로

### 2.3 첫 구현에서 제외하는 요소

- spatial threshold, local-maximum NMS, top-k/top-100 candidate selection
- RGB-to-Thermal 또는 Thermal-to-RGB local spatial search
- instance-level hard/mutual prototype matching
- FFT/frequency band decomposition
- EDL, utility router, counterfactual detector branch
- modality-specific frozen teachers와 teacher-mask distillation
- RGB feature warp, deformable alignment, offset prediction
- prototype에서 dense spatial feature를 직접 재구성하는 경로

공간 정렬은 기존 AAM의 책임으로 남긴다. TPSC prototype은 동일 객체의 좌표 대응이
아니라 장면 내 target-relevant semantic basis로 해석한다.

## 3. 적용 level과 통제 모델

Canonical treatment는 P3/P4 `apply_levels=(0, 1)`에 적용한다.

- P3는 tiny 객체의 local detail을 보존하고 강화한다.
- P4는 tiny2/tiny3와 small 객체를 위한 semantic context를 제공한다.
- P5/P6은 TPSC를 거치지 않고 기존 AAM/HOFM으로 바로 전달한다.

세 개의 seed-0 모델을 동일한 학습 recipe로 비교한다.

1. `same_stage_control`: CLFM과 TPSC가 없는 same-stage AAM/HOFM
2. `TPSC_core`: level별 shared-slot RGB–Thermal prototype conditioning과 RGB modulation
3. `TPSC_relation`: prototype relation mixer까지 포함한 canonical model

`TPSC_core`는 같은 slot의 RGB/Thermal prototype만 결합한다. `TPSC_relation`은 여기에
모달·scale 전체 관계를 추가한다. 이 분리가 없으면 prototype conditioning과 relation
중 무엇이 기여했는지 판단할 수 없다.

## 4. Detail-preserving cross-scale descriptor

원 RGB/Thermal feature는 AAM 입력용으로 그대로 보존한다. Prototype extraction에만
별도 descriptor를 사용한다.

P3 descriptor는 원 P3, local high-pass detail, P4 semantic context를 결합한다.

```text
detail_m3   = F_m3 - ValidAvgPool3(F_m3)
semantic_m3 = BilinearUpsample(Conv1x1(F_m4), size=F_m3.shape)
D_m3 = Conv1x1([F_m3, detail_m3, semantic_m3])
D_m4 = Conv1x1(F_m4)
```

`m in {RGB, Thermal}`이다. `ValidAvgPool3`은 padding을 제외한 normalized average를
사용하며 padding 위치의 descriptor는 이후 attention에서 mask한다.

P4는 prototype descriptor에만 사용한다. P4 feature를 P3 spatial feature에 직접
더하거나 P3를 warp하지 않는다. 따라서 CLFM이 제공하던 상위 semantic context는
복구하되 P3 위치·경계 정보는 원 feature에 남는다.

## 5. Shared-slot prototype extraction

각 level과 modality에서 기본 `K=8`개의 prototype을 추출한다. Slot query는 RGB와
Thermal 사이에서 공유하고, input projection은 modality별로 둔다.

```text
E_m_l(x) = LN(Conv1x1_m_l(D_m_l(x)))
score_m_l(k,x) = dot(q_l(k), E_m_l(x)) / sqrt(d)
A_m_l(k,x) = masked_softmax_x(score_m_l(k,x))
P_m_l(k) = sum_x A_m_l(k,x) * E_m_l(x)
```

- `q_l(k)`는 같은 level의 RGB/Thermal에서 공유한다.
- `masked_softmax`는 padding과 invalid 위치를 정확히 제외한다.
- Prototype은 특정 GT instance와 일대일 대응하지 않는다.
- 영상의 객체 수가 K보다 많아도 각 slot attention이 여러 target 위치에 질량을 둘 수
  있으므로 fixed candidate quota가 되지 않는다.
- RGB attention에 Thermal 좌표 target을 복사하지 않는다.

Modality별 projection은 radiometric 특성 차이를 수용하고, shared slot query는 hard
matching 없이 slot 의미의 일관성을 학습하도록 유도한다.

## 6. Thermal target coverage supervision

Thermal GT box를 각 feature level로 투영해 normalized Gaussian foreground distribution
`Y_l`을 만든다. 각 객체 Gaussian은 projected box 크기에 맞는 최소 radius 1을 사용하며,
여러 객체는 pixel maximum으로 합친다. Padding과 ignore 영역은 제외한다.

Thermal slot attention의 평균 coverage를 다음과 같이 정의한다.

```text
C_T_l(x) = mean_k A_T_l(k,x)
C_T_l <- C_T_l / sum_x C_T_l(x)
Y_l   <- Y_l   / sum_x Y_l(x)
L_coverage = symmetric_KL(Y_l, C_T_l)
```

GT가 없는 image-level 항은 생략한다. 이 loss는 prototype이 background만 요약하는 것을
방지할 뿐, slot을 개별 객체나 RGB 좌표와 매칭하지 않는다.

Slot collapse는 attention overlap으로 제한한다.

```text
L_diversity = mean_{i != j} cosine(A_T_l(i), A_T_l(j))
```

초기 weight는 `coverage_loss_weight=0.05`, `diversity_loss_weight=0.01`로 한다.
첫 구현에는 prototype alignment/contrastive loss를 추가하지 않는다. Complementary
modal information까지 같게 만드는 shortcut을 피하기 위해서다.

## 7. Prototype-set relation mixer

Core model은 각 level의 shared slot끼리만 RGB conditioning을 만든다.

```text
Q_R_l(k) = P_R_l(k)
         + MLP_l([P_R_l(k), P_T_l(k), P_T_l(k) - P_R_l(k)])
```

이는 slot index가 특정 instance를 뜻한다는 가정이 아니라, 공유 query로 추출된 같은
semantic basis끼리 정보를 교환하는 연산이다. `TPSC_core`는 이 `Q_R_l`을 바로
modulation head에 사용한다.

Relation model의 각 image prototype node 집합은 다음과 같다.

```text
X = [P_R3, P_T3, P_R4, P_T4]    # shape: B x (4K) x d
```

Fixed top-k graph나 non-differentiable edge selection을 사용하지 않는다. 대신 작은 node
집합에 pre-norm multi-head self-attention 한 block을 사용한다.

```text
X1 = X + MHSA(LN(X))
X2 = X1 + MLP(LN(X1))
```

기본값은 `d=64`, `heads=4`, `depth=1`, dropout 0이다. Prototype 수가 32개뿐이므로
full-resolution dense fusion과 달리 계산량과 background coupling이 제한적이다.

Mixer 출력 `X2`를 modality/level별로 다시 분리한 뒤, core와 같은 conditioning을
적용한다.

```text
Q_R_l(k) = P_R_l_x2(k)
         + MLP_l([P_R_l_x2(k), P_T_l_x2(k),
                  P_T_l_x2(k) - P_R_l_x2(k)])
```

이 `Q_R_l`만 modulation head에 사용한다. Thermal node는 reference와 message source
역할만 하며 Thermal spatial feature에는 쓰지 않는다.

`TPSC_core`에서는 mixer를 identity로 바꿔 동일한 extraction, shared-slot conditioning,
modulation 경로를 유지한다.

## 8. RGB-only bounded channel modulation

Conditioned RGB prototype을 flatten한 뒤 level별 channel scale을 예측한다.

```text
z_R_l = Flatten(Q_R_l)
gamma_l = Linear(z_R_l)          # B x C
scale_l = 0.1 * tanh(gamma_l)
F_R_l_cal = F_R_l * (1 + scale_l[:, :, None, None])
```

- Shift/bias 항은 첫 구현에서 사용하지 않는다. 공간적으로 균일한 bias가 background까지
  새 object evidence를 만드는 것을 피한다.
- 별도 support, confidence, reliability 또는 RGB objectness gate를 곱하지 않는다.
- `0.1`은 고정 residual bound다. 학습 가능한 scalar가 0으로 닫히는 경로를 만들지 않는다.
- `Linear`는 작은 비영 normal weight(`std=1e-2`)와 zero bias로 초기화한다.
- Prototype으로 RGB spatial feature를 복원하지 않으므로 원 RGB의 위치 정보가 유지된다.
- P3/P4 Thermal feature 값은 bitwise identity로 AAM에 전달한다. Detection loss의
  prototype branch gradient는 Thermal descriptor와 Thermal backbone까지 허용한다.

최종 경로는 다음과 같다.

```text
HOFM/AAM(F_R3_cal, F_T3)
HOFM/AAM(F_R4_cal, F_T4)
HOFM/AAM(F_R5,     F_T5)
HOFM/AAM(F_R6,     F_T6)
```

## 9. Loss와 진단 지표

전체 학습 loss는 다음뿐이다.

```text
L = L_detector
  + 0.05 * L_tpsc_coverage
  + 0.01 * L_tpsc_diversity
  + L_wf
```

다음 값은 detached monitor로만 기록한다.

- level별 Thermal foreground coverage와 target mass
- RGB/Thermal slot attention entropy
- level별 prototype pairwise cosine과 effective-rank proxy
- mixer 전후 RGB-Thermal prototype cosine
- cross-modal attention mass와 cross-scale attention mass
- predicted channel scale absolute mean/max
- `||F_R_cal - F_R|| / ||F_R||` modulation ratio
- prototype extractor, mixer와 modulation head의 gradient norm

Monitor는 loss에 합산하지 않는다. 특히 cosine 증가만으로 correspondence 정확도나
domain calibration 성공을 주장하지 않는다.

## 10. 인과 개입과 실패 판정

완료 checkpoint에 다음 inference intervention을 제공한다.

1. `disable_modulation`: `F_R_cal=F_R`로 평가
2. `shuffle_thermal_prototypes`: batch 안에서 Thermal prototype만 다른 image와 교환
3. `disable_relation`: relation mixer를 identity로 평가

다음 조건이면 prototype 경로가 유효하게 사용되지 않은 것으로 판정한다.

- modulation ratio가 학습 내내 수치 정밀도 수준에 머무름
- modulation head 또는 prototype extractor gradient norm이 지속적으로 0
- `disable_modulation`과 정상 평가 결과가 사실상 동일
- Thermal prototype shuffle에도 결과가 사실상 동일

성능이 비슷해도 위 조건에 해당하면 TPSC의 효과라고 해석하지 않는다.

## 11. 코드 경계

- 새 모듈: `mmdet/models/utils/tpsc.py`
- 새 integration flag: `use_tpsc`, `tpsc_cfg`
- integration: `mmdet/models/utils/fusion_strategy.py`
- detector plumbing: `mmdet/models/detectors/fusionnet_xo.py`
- configs:
  - `configs/coxnet/tpsc/same_stage_control.py`
  - `configs/coxnet/tpsc/TPSC_core.py`
  - `configs/coxnet/tpsc/TPSC_relation.py`
- tests: `tests/test_models/test_utils/test_tpsc.py`
- documentation: README의 method/config/training/result 표

기존 TRPC/OEPC/TOPC/PRLDFC 구현과 완료 실험 결과는 재현성을 위해 변경하거나 삭제하지
않는다. 활성 replacement 간 mutual-exclusion 검사에 TPSC를 추가한다.

공식 ProtoHGF 저장소는 AGPL-3.0이고 이 저장소는 MIT이므로, 외부 소스 파일을 복사하지
않고 논문에 공개된 개념을 바탕으로 독립 구현한다.

## 12. 테스트와 수용 기준

### 12.1 단위 테스트

- RGB/Thermal input과 calibrated output shape/dtype/device 보존
- AAM으로 전달되는 Thermal tensor 값이 입력과 bitwise identity
- Detection loss가 RGB/Thermal prototype extractor와 backbone 양쪽에 도달
- padding 위치가 prototype attention mass를 받지 않음
- all-padding/empty-GT/tiny map에서도 NaN/Inf 없이 동작
- attention은 slot별 유효 공간에서 합 1
- coverage/diversity loss가 empty-GT에서 finite zero
- relation on/off가 동일 interface를 유지
- 초기 modulation은 비영이면서 10% bound를 넘지 않음
- detection-style scalar loss에서 extractor/mixer/modulation projection에 비영 gradient
- intervention mode가 요청한 경로만 변경
- replacement mutual exclusion과 legacy regression

### 12.2 통합 검증

- config parse와 model construction
- CPU focused unit suite
- GPU 1 single-batch forward/backward smoke
- train/eval padding-mask 전달 회귀 테스트
- 첫 train iteration에서 모든 TPSC loss/monitor가 finite

### 12.3 실험 판정

먼저 seed 0에서 control, core, relation을 비교한다. Original COXNet과의 직접 비교는
동일 validation annotation과 best-epoch selection으로 수행한다.

- Primary: `bbox_mAP_50`
- Secondary: `bbox_mAP_75`, tiny, tiny1/2/3, small
- Mechanism: modulation/gradient/intervention diagnostics

Canonical `TPSC_relation`은 사용자가 요청한 대로 seed 0, 1, 2를 순차 실행하며 세 seed
mean/std와 각 seed 값을 모두 기록한다. `same_stage_control`과 `TPSC_core`는 원인 분리를
위한 seed-0 대조 실험으로 먼저 실행한다. 단, seed 0에서 modulation/gradient가 죽었거나
NaN 등 구현 문제가 확인되면 잘못된 동일 코드를 seed 1, 2로 반복하지 않고 원인을
수정·재검증한 뒤 세 seed를 처음부터 다시 시작한다.

수치 향상은 구현으로 보장하지 않는다. 목표는 prototype 방향을 실제로 활성화한 공정한
통제 실험으로 검증하고, 실패하더라도 extraction, relation, modulation 중 어느 단계가
원인인지 분리할 수 있게 하는 것이다.

## 13. Git과 학습 운영

1. 별도 implementation branch에서 test-first로 구현한다.
2. Focused tests와 GPU smoke가 통과한 commit만 `origin/main`에 push한다.
3. 데이터, checkpoint, work directory와 log는 Git에 포함하지 않는다.
4. GPU 1에서는 이미 실행 중인 작업과 lock을 먼저 확인한다.
5. Seed별 work directory와 console log를 분리한다.
6. 각 process exit code가 0일 때만 다음 seed를 시작한다.
7. Seed-0 control/core/relation 결과와 mechanism diagnostics를 확인한 뒤 canonical
   `TPSC_relation`의 seed 1, 2를 순차 실행한다.
8. 실행 명령만으로 학습 시작이나 완료를 주장하지 않고 PID/tmux, GPU process, log와
   checkpoint를 함께 확인한다.
