# TOPC: Thermal-Anchored Object Prototype Calibration 설계

## 역할

TOPC는 fusion이나 alignment 모듈이 아니다. 기존 AAM이 spatial alignment를 수행하기
전에 Thermal에서 발견된 객체를 semantic anchor로 사용해 대응 RGB local feature만
보정하는 pre-alignment semantic calibration 모듈이다.

```text
Thermal P3 -> object candidate -> object prototype
           -> local RGB semantic correspondence
           -> prototype discrepancy residual on RGB
           -> original AAM(calibrated RGB P3, original Thermal P3)
```

기존 CLFM의 dense frequency prior 전달을 object-wise semantic correction으로 대체한다.
RGB/Thermal FPN은 모두 `start_level=1`이고 TOPC는 같은 stride의 P3만 사용한다.
P4, cross-stage pairing, DeConv, DWT/IDWT는 사용하지 않는다.

## 제거 범위

- P4/cross-scale context
- EDL과 uncertainty
- learned utility/router/gate
- foreground/context/contrastive/cycle/diversity/alignment loss
- FiLM과 global K prototype
- RGB candidate head

전체 loss는 다음뿐이다.

```text
L = L_det + 0.1 * L_objectness
```

## 1. Thermal objectness

Thermal P3 `F_T`에서 작은 convolution head로 logits와 probability를 만든다.

```text
Z_T = ObjectnessHead(F_T)
H_T = sigmoid(Z_T)
```

GT box의 float P3 중심에 normalized Gaussian target을 그린다. projected box 크기에서
`radius=clamp(ceil(0.5*sqrt(w_f*h_f)),1,2)`, `sigma=(2r+1)/6`을 사용한다.
각 객체 Gaussian은 local maximum이 1이 되게 정규화하고 겹침은 pixel maximum으로
합친다. padding은 target/loss/candidate에서 제외한다.

후보는 image별 `H_T >= 0.05` cell 중 score top-100이다. local-peak NMS는 사용하지
않아 인접 tiny 객체의 cell을 억제하지 않는다. top-K index는 discrete지만 선택된
heatmap probability는 prototype weight에 그대로 사용되어 detection gradient도 받는다.
objectness prior는 `0.01`이다.

## 2. Candidate-specific Thermal prototype

candidate `c_i`의 3x3 Thermal patch를 추출한다. valid mask와 heatmap probability를
곱해 정규화한 뒤 raw Thermal P3 feature를 weighted pooling한다.

```text
w_i(x) = H_T(x) * valid(x), x in N_3x3(c_i)
P_T_i = sum_x w_i(x) F_T(x) / (sum_x w_i(x) + eps)
```

유효 weight mass가 0인 후보는 inactive로 처리한다. image-global prototype slot은 없다.
각 candidate가 정확히 하나의 object prototype을 가진다.

## 3. Local RGB semantic correspondence

RGB P3는 learnable `1x1` projection `phi`로 Thermal semantic space에 보낸다. candidate
center 반경 2의 5x5 RGB patch만 검색한다.

```text
K_R(x) = phi(F_R(x))
s_i(x) = cosine(P_T_i, K_R(x))
A_i(x) = softmax(s_i(x) / 0.2), x in valid local window
P_R_i = sum_x A_i(x) K_R(x)
confidence_i = clamp((max_x s_i(x) + 1) / 2, 0, 1)
```

feature tensor를 이동하거나 warp하지 않는다. attention은 Thermal prototype과 의미적으로
대응하는 RGB local support와 representation을 찾을 뿐이며 spatial alignment는 AAM에 남긴다.

## 4. Prototype discrepancy residual

```text
D_i = P_T_i - P_R_i
R_i = W(D_i)
```

`W`는 channel dimension을 유지하는 작은 linear projection이며 마지막 weight는
`Normal(0,1e-2)`, bias는 0으로 초기화한다. learned gate는 없다. 유일한 신뢰도 계수는
normalized cosine confidence다.

candidate residual은 association attention으로 RGB 5x5 위치에 쓴다. 여러 candidate가
겹치면 다음처럼 합산 폭증을 막는다.

```text
mass(x) = sum_i A_i(x)
average(x) = sum_i A_i(x) * confidence_i * R_i / (mass(x) + eps)
support(x) = clamp(mass(x), 0, 1)
delta(x) = support(x) * average(x)
F_R_cal = F_R + delta
```

후보 support 밖 RGB는 exact identity다. Thermal P3는 변경하지 않는다. 별도 residual
gate나 norm cap은 두지 않아 최종 information flow를 위 식 하나로 유지한다.

## 진단 지표

- objectness probability mean/max와 threshold 통과 수
- candidate count와 Gaussian center recall
- prototype valid ratio와 prototype RMS
- matching similarity/confidence/entropy
- RGB support ratio와 overlap ratio
- raw/final residual ratio

지표는 진단용이며 correspondence 정확도나 AP 향상을 스스로 입증하지 않는다.

## 코드 경계

- 새 모듈: `mmdet/models/utils/topc.py`, class `ThermalAnchoredObjectPrototypeCalibration`
- 새 integration flag: `use_topc`, `topc_cfg`
- canonical config: `configs/coxnet/topc/TOPC.py`
- 기존 `oepc.py`, TRPC config와 완료 실험은 변경하지 않는다.
- active TOPC graph는 P3 RGB만 보정하고 원 Thermal P3와 기존 HOFM/AAM을 호출한다.

## 수용 기준

- active graph에 CLFM/DWT/IDWT/DeConv/P4/EDL/utility/gate/FiLM이 없다.
- Thermal objectness logits는 RGB 입력과 무관하다.
- 각 GT Gaussian은 maximum 1이고 padding 밖은 0이다.
- 인접 above-threshold cell이 local NMS 없이 후보로 남는다.
- prototype은 candidate별 heatmap-weighted 3x3 Thermal pooling이다.
- association은 Thermal prototype query와 local RGB soft correspondence다.
- learned confidence gate 없이 normalized cosine confidence만 사용한다.
- overlap aggregation은 단일 candidate attention을 보존하고 중복 residual을 제한한다.
- no candidate/empty GT/padding/boundary 조건에서 finite identity 동작을 한다.
- 보조 loss는 objectness 하나뿐이다.
- legacy OEPC/TRPC regression tests가 통과한다.

## 배포와 학습

검증된 branch를 `origin/main`에 fast-forward push한 뒤 GPU 1에서 seed 0, 1, 2를
각각 별도 work directory로 순차 실행한다. 이전 seed exit code가 0일 때만 다음 seed를
시작한다. 데이터와 checkpoint/log는 Git에 포함하지 않는다.
