# TF-OPC: Thermal-First Object-Prototype Calibration 설계 (Superseded)

> 이 설계는 더 단순한 `2026-09-23-topc-design.md`로 대체되었다.

## 목적

TF-OPC는 COXNet의 CLFM을 완전히 대체하는 AAM 이전 보정 모듈이다. Thermal에서
작은 객체를 먼저 찾고, 객체마다 local prototype을 만든 뒤, 주변 RGB 대응 위치에만
객체 단서를 전달한다. 공간 feature 정렬은 수행하지 않으며 기존 AAM/HOFM/DSR/GFL을
그대로 유지한다.

```text
Thermal target discovery
  -> candidate-specific object prototype
  -> local RGB association
  -> selective RGB residual
  -> original AAM/HOFM/DSR/GFL
```

기존 balanced OEPC와 앞선 utility 기반 OEPC-v2 설계는 재현 이력으로 보존한다.
TF-OPC는 새 `oepc_v2.py` 구현과 `use_oepc_v2` 경로를 사용한다.

## 제거하는 요소

- EDL evidence/uncertainty head와 EDL loss
- RGB candidate head
- foreground auxiliary head/loss
- keep/trial detector utility 및 utility loss
- FiLM scale/shift
- 여러 신뢰도 값을 연속으로 곱하는 gate
- global K prototype과 hard identity matching
- DWT, IDWT, DeConv, cross-stage FPN pairing

첫 실험의 전체 loss는 다음뿐이다.

```text
L = L_det + 0.10 L_heatmap + 0.05 L_contrast + 0.02 L_cycle
```

## FPN과 적용 범위

- RGB/Thermal FPN 모두 `start_level=1`: P3/P4/P5/P6 = stride 8/16/32/64.
- TF-OPC 보정 대상은 P3 한 level뿐이다.
- 동일 모달 P4는 descriptor semantic context로만 사용한다.
- RGB P3 원 feature만 residual로 바꾼다.
- Thermal P3는 tensor 값과 좌표를 변경하지 않고 AAM에 전달한다.

## 1. Detail-preserving descriptor

모달리티 `M`에 대해:

```text
B_M = LN(Conv1x1(F_M^3))
L_M = B_M - ValidAvgPool3(B_M)
S_M = Up(LN(Conv1x1(F_M^4)))
Z_M = LN(Conv1x1([B_M, L_M, S_M]))
```

평균은 valid mask로 정규화해 padding을 제외한다. P4 resize는 descriptor 내부의
`bilinear, align_corners=False` 한 번만 허용한다. 같은 level RGB/Thermal shape이
다르거나 P4가 P3보다 작지 않으면 오류를 낸다.

## 2. Thermal Gaussian heatmap

후보 heatmap은 `Z_T`에만 의존한다.

```text
logits_T = HeatmapHead(Z_T)
probability_T = sigmoid(logits_T)
```

RGB 입력을 바꿔도 Thermal heatmap은 변하지 않아야 한다. 초기 prior는 `0.01`,
threshold는 `0.10`이다.

GT box의 float P3 center와 projected 크기로 Gaussian을 만든다.

```text
radius = clamp(ceil(0.5 * sqrt(w_f * h_f)), 1, 2)
sigma = (2 * radius + 1) / 6
G_i = exp(-distance_squared / (2 sigma_squared))
G_i = G_i / max(G_i)
target = max_i(G_i)
```

각 객체 target의 최고값은 sub-pixel center에서도 1이며, 겹치는 객체는 합산하지 않고
pixel maximum으로 병합한다. valid mask 밖 target은 0이다.

## 3. Straight-through sparse candidate selection

학습에서는 3x3 local-peak NMS를 사용하지 않는다. probability의 image별 top-K
index를 고른 뒤 threshold를 적용해 hard mask를 만든다. soft mask는 logit 기준이다.

```text
soft = sigmoid((logits_T - logit(threshold)) / support_temperature)
hard = thresholded_topk(probability_T, K)
selection = soft + stopgrad(hard - soft)
```

forward 값은 hard sparse mask이고 backward는 soft mask gradient를 사용한다. top-K
index 선택 자체는 discrete이며 선택된 candidate score 경로에 straight-through
gradient가 흐른다. 기본 `K=100`, `support_temperature=0.5`다.

추론에서는 threshold + top-K 전에 configurable local peak를 적용한다. 첫 config는
`inference_peak_kernel=3`이다. 학습 경로에는 local peak를 적용하지 않는다.

## 4. Candidate-specific Thermal prototype

선택된 Thermal candidate `i`마다 P3의 3x3 object window와 7x7 context ring을
추출한다. object weight는 고정 Gaussian center prior와 valid mask를 정규화한다.
context는 object 3x3을 제외한 ring과 valid mask를 정규화한다.

```text
P_obj_i = WeightedPool(Z_T, object_window_i)
P_ctx_i = WeightedPool(Z_T, context_ring_i)
P_detail_i = WeightedPool(L_T, object_window_i)
P_T_i = LN(MLP([P_obj_i, P_obj_i - P_ctx_i, P_detail_i]))
```

context mass가 0이면 해당 candidate의 contrast loss를 제외하고 `P_ctx_i=0`으로 둔다.
prototype은 image-global K slot이 아니라 candidate마다 하나씩 생성된다.

Local contrast loss는 다른 사람을 negative로 사용하지 않는다.

```text
L_contrast = mean_valid(relu(cos(proj(P_obj), proj(P_ctx)) - margin))
```

기본 margin은 `0.2`다.

## 5. Thermal-to-RGB top-M association

Thermal prototype `P_T_i`가 query이며 candidate center 반경 `r=2`의 RGB descriptor
`Z_R`만 검색한다.

```text
similarity_i,x = cosine(q(P_T_i), k(Z_R(x))) - distance_prior_i,x
topM = TopM(similarity_i, M)
attention_i = softmax(similarity_i[topM] / temperature)
P_R_i = sum_x attention_i,x * Z_R(x)
```

기본 `M=4`, temperature `0.2`, distance prior weight `0.1`이다. RGB feature map을
warp하지 않는다. attention은 Thermal object evidence를 어느 RGB 위치에 전달할지만
정한다.

## 6. Cycle consistency

forward association의 attention-weighted RGB coordinate를 반올림·detach해 reverse
search center로 사용한다. `P_R_i`를 query로 같은 반경의 Thermal descriptor를
검색하고 top-M soft attention으로 복귀 좌표 `y'_i`를 만든다.

```text
L_cycle = mean_active(smooth_l1((y'_i - y_i) / max(r, 1), 0))
```

선택된 active candidate와 유효 search window에만 적용한다. 이는 feature warp나
동일 인물 label이 아니라 local association의 왕복 일관성 감독이다.

## 7. Single gate와 RGB residual

후보별 direct residual과 gate를 만든다.

```text
R_i = MLP([P_T_i, P_R_i, P_T_i - P_R_i,
           match_similarity_i, heatmap_score_i])
g_i = sigmoid(MLP([heatmap_score_i, match_similarity_i, P_T_i, P_R_i]))
```

gate는 후보당 하나뿐이며 candidate score나 matching confidence를 correction에 다시
별도 곱하지 않는다. residual projection은 zero-init하지 않고 `Normal(0, 1e-2)`로
초기화하며 gate 마지막 bias는 0으로 초기화한다.

각 candidate residual은 forward top-M RGB 위치에 attention으로 broadcast한다.
겹치는 candidate는 합산하지 않고 normalized weighted average로 결합한다.

```text
weight_i,x = selection_i * attention_i,x
support_x = clamp(sum_i weight_i,x, 0, 1)
residual_x = sum_i weight_i,x * g_i * R_i / (sum_i weight_i,x + eps)
delta0_x = support_x * residual_x
```

최종 안전장치는 위치별 channel RMS cap이다.

```text
gamma_x = min(1, 0.2 * rms(F_R(x)) / (rms(delta0_x) + eps))
delta_x = gamma_x * delta0_x
F_R_cal = F_R + delta
```

support가 0인 위치는 exact identity다. Thermal P3는 수정하지 않는다.

## 8. 진단 지표

- heatmap probability mean/max, threshold pass count
- selected candidate count와 GT-center candidate recall
- object/context valid ratio
- top-M match similarity, confidence, entropy
- cycle displacement와 cycle loss
- gate mean/on-support
- support ratio, overlap count, raw/final delta ratio, cap ratio

이 지표는 학습 상태 진단용이며 correspondence 정확도나 성능 향상의 증명으로 해석하지
않는다.

## 9. 코드 경계

- 새 구현: `mmdet/models/utils/oepc_v2.py`
- 기존 재현 구현: `mmdet/models/utils/oepc.py` 그대로 유지
- 통합: `FusionLayer(use_oepc_v2=True, oepc_v2_cfg=...)`
- detector는 v2 auxiliary loss를 기존 detection loss에 더하되 utility counterfactual은
  실행하지 않는다.
- canonical config: `configs/coxnet/oepc/TF_OPC.py`
- 문서: `docs/TF_OPC.md`, `README.md`

## 10. 수용 기준

- active TF-OPC 경로에 CLFM/DWT/IDWT/DeConv/EDL/utility/FiLM이 없다.
- Thermal heatmap은 RGB 입력과 무관하다.
- Gaussian target은 각 객체마다 최고값 1을 가진다.
- 학습 candidate forward는 sparse hard top-K이고 heatmap logit gradient는 non-zero다.
- 학습에는 local peak NMS가 없고 추론에만 configurable peak가 있다.
- prototype이 candidate마다 object/context/detail로 생성된다.
- association은 Thermal query, local RGB top-M이며 feature warp가 없다.
- cycle, heatmap, local contrast 세 auxiliary loss만 존재한다.
- single gate와 residual branch가 detection loss gradient를 받는다.
- overlap aggregation과 RMS cap이 residual 폭증을 막는다.
- P4는 descriptor에서만 쓰고 원 Thermal P3가 AAM에 전달된다.
- padding, empty GT, no candidate, missing context, dense adjacent candidate 테스트가 통과한다.
- legacy OEPC/TRPC 테스트가 회귀하지 않는다.

## 11. 학습과 배포

코드/CPU/GPU smoke 검증 뒤 원격 `main`을 fast-forward push한다. 이후 GPU 1에서
동일 config로 seed 0, 1, 2를 한 process씩 순차 실행한다. 각 seed는 별도 work directory와
log를 사용하며, 앞 seed가 정상 종료한 뒤 다음 seed를 시작한다. 데이터 폴더와 checkpoint는
Git에 포함하지 않는다.
