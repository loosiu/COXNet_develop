# OEPC-v2: Thermal-first Tiny-object RGB Calibration 설계

## 1. 목적과 현재 상태

이 문서는 COXNet의 CLFM을 완전히 제거한 same-stage RGB/Thermal FPN 구조에서,
P3 RGB feature만 선택적으로 강화하는 OEPC-v2를 정의한다. 최종 목적은
RGB에서 약하지만 Thermal에서 식별 가능한 작은 사람의 단서를 RGB P3에 전달하되,
RGB의 위치별 세부 구조와 AAM의 정렬 역할을 보존하는 것이다.

구현 기준점은 `69d9037`의 balanced OEPC이며, 원격 `main`의 기존 이력을 포함하는
별도 worktree/branch에서 개발한다. 기존 balanced 설정과 결과 파일은 재현을 위해
삭제하지 않는다.

코드 재검토 결과, 처음 제안에 있던 다음 두 진단은 현재 기준점에는 적용되지 않는다.

1. 현재 후보 생성은 이미 `Thermal candidate head -> Thermal peak -> Thermal-to-RGB
   local attention` 순서다. RGB query가 Thermal 후보를 먼저 찾는 구조가 아니다.
2. 현재 detector utility는 별도 분류기의 대리 loss가 아니라 실제 GFL head의
   고정 assignment keep/trial detection loss를 사용한다.

따라서 OEPC-v2는 이 두 기능을 새로 가장하지 않고 유지·정리하며, 실제 병목인
dual-modality 후보 경쟁, hard sparse support, tiny-detail 평균화, P4 의미 문맥 소실,
선택성 진단 부족을 해결한다.

## 2. 목표

- Thermal에서 먼저 객체 후보를 발견해 weak-RGB 객체도 보정 기회를 얻는다.
- 가까운 tiny 객체가 P3의 3x3 peak suppression 때문에 하나로 합쳐지지 않게 한다.
- 학습 중 candidate 위치 선택에 detection gradient가 도달하게 한다.
- P3의 원 위치 feature, local detail, 객체-배경 대비를 descriptor에 보존한다.
- P4는 descriptor의 의미 문맥에만 사용하고 AAM 입력 P3에는 직접 섞지 않는다.
- Thermal P3는 수정하지 않고 원본 그대로 AAM/HOFM에 전달한다.
- 실제 GFL detection loss로 transfer router의 국소 utility를 감독한다.
- 보정 영역 밖 RGB는 exact identity이고, 보정량은 명시적으로 제한한다.
- 이전 실험과 직접 비교할 수 있도록 데이터, detector, AAM, 학습 schedule은 유지한다.

## 3. 비목표

- OEPC에서 RGB/Thermal feature map 전체를 warp하거나 정렬하지 않는다.
- AAM을 제거하거나 대체하지 않는다.
- RGB 후보 head로 Thermal에서 보이지 않는 객체를 별도로 제안하지 않는다.
- P4 feature를 P3 RGB/AAM 입력에 직접 더하거나 concatenate하지 않는다.
- DWT, IDWT, CLFM DeConv, cross-stage FPN pairing을 되살리지 않는다.
- 첫 실험에서 P3 이외의 P4-P6까지 OEPC를 확장하지 않는다. 보정 대상은 P3뿐이다.
- local attention을 동일 인물의 정답 correspondence 확률로 해석하지 않는다.
- 구현 완료만으로 AP 향상을 주장하지 않는다. 성능 주장은 통제 실험 뒤에만 한다.

## 4. 전체 데이터 흐름

RGB와 Thermal FPN은 모두 `start_level=1`을 사용해 stride 8, 16, 32, 64의
P3-P6를 생성한다. OEPC-v2는 P3를 보정 대상으로 사용하고 P4는 descriptor용
semantic context로만 읽는다.

```text
Thermal P3 + Thermal P4
          |
          v
detail-preserving Thermal descriptor
          |
          v
Thermal candidate heatmap + differentiable support
          |
          v
Thermal query -> local RGB descriptor association
          |
          v
bounded spatial RGB modulation + detector-utility router
          |
          v
RGB P3 calibrated ------------------+
                                     +--> existing AAM/HOFM --> GFL
Thermal P3 unchanged ----------------+
```

P4는 `FusionLayer.forward()`가 P3 OEPC에 `rgb_semantic=rgb_feats[1]`,
`thermal_semantic=thermal_feats[1]`로 전달한다. 같은 level의 RGB/Thermal shape이
다르거나 P4가 P3보다 작지 않으면 입력 검증에서 오류를 낸다. P4 projection 뒤 P3
크기로 올리는 것은 설계된 유일한 resize이며 `bilinear, align_corners=False`를 쓴다.

## 5. Detail-preserving descriptor

모달리티 `M in {R, T}`에 대해 P3와 P4에서 다음 표현을 만든다.

```text
B_M = LN(Conv1x1(F_M^3))
S_M = Upsample(LN(Conv1x1(F_M^4)), size=P3)
L_M = B_M - ValidAvgPool3(B_M)
```

`B_M`은 위치별 중심 표현이고, `L_M`은 3x3 평균으로 제거되기 쉬운 local detail이다.
기존처럼 object prototype 자체를 3x3 평균으로 만들지 않는다.

배경 context는 7x7 ring에서 계산한다. 먼저 `B_M`으로 foreground probability
`P_fg,M`을 예측하고, 각 중심 위치에서 ring weight를 다음과 같이 정의한다.

```text
W_M = ring7 * valid * (1 - P_fg,M)
context_mass = sum(W_M)
context_coverage = context_mass / sum(ring7 * valid)
C_M = sum(W_M * unfolded(B_M)) / max(context_mass, eps)
A_M = 1[context_coverage >= min_context_coverage]
Q_M = (B_M - C_M) * A_M
```

기본 `min_context_coverage`는 `0.25`다. 유효 ring 자체가 없는 경우 coverage는
0으로 두고 context-dependent auxiliary loss를 제외한다. 기존의
`context_mass > 1e-6` 조건처럼 거의 모든 위치를 유효하다고 처리하지 않는다.

최종 descriptor는 다음과 같다.

```text
D_M = LN(Conv1x1([B_M, L_M, Q_M, S_M]))
```

P3 원 feature `F_R^3`는 descriptor에만 읽히며, calibration 전 주 경로에서는
pooling하거나 재배치하지 않는다.

## 6. Thermal-only candidate와 target

후보 logit은 Thermal descriptor에서만 만든다.

```text
Z_T = CandidateHead(D_T)
P_T = sigmoid(Z_T)
```

`rgb_candidate_head`, RGB candidate loss, RGB-origin indicator는 제거한다.
RGB foreground/evidence head는 context 구성과 router 입력을 위해 유지할 수 있지만,
후보의 존재 여부를 결정하지 않는다.

### 6.1 Gaussian center target

기존 single-cell target 대신 Thermal GT 중심 주위에 작은 Gaussian을 그린다.
GT box를 P3 좌표로 변환한 폭과 높이를 `w_f`, `h_f`라 할 때 다음을 사용한다.

```text
radius = clamp(ceil(0.5 * sqrt(w_f * h_f)), min=1, max=2)
sigma = (2 * radius + 1) / 6
Y(x, y) = exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
```

여러 GT의 Gaussian이 겹치면 pixel별 maximum을 취한다. 중심은 float 좌표로
유지해 양자화 오차를 줄인다. padded image 크기와 실제 P3 shape에서 scale을
계산하며 valid mask 밖 target은 0이다. 기존 balanced focal objective는 soft target을
받도록 테스트로 고정한다.

### 6.2 초기 prior

- `candidate_prior=0.01`
- `candidate_threshold=0.10`
- `support_logit_temperature=0.5`

따라서 학습 시작 시 prior가 threshold보다 작다. 초기부터 top-k가 포화되는 현재
조건을 제거한다.

## 7. Training support와 inference support

학습과 추론의 선택 규칙을 분리한다.

### 7.1 Training

학습에서는 local maximum, top-k, detached peak를 사용하지 않는다.

```text
tau_logit = logit(candidate_threshold)
S_T_soft = sigmoid((Z_T - tau_logit) / support_logit_temperature) * valid
```

이 support는 local association, router, RGB residual까지 연속적으로 연결되므로
detector loss가 candidate logit에 gradient를 전달할 수 있다. candidate focal loss도
동시에 유지해 support가 detection loss shortcut만 따르지 않게 한다.

### 7.2 Inference

추론에서는 `P_T >= candidate_threshold`인 유효 cell 중 image별 top-100을 고른다.
3x3 local-maximum NMS는 사용하지 않는다(`peak_kernel=1`과 동등). 두 인접 cell이
각기 높은 확률을 가지면 둘 다 남을 수 있다. 선택된 점의 score로 hard support를
구성한다.

동일 RGB 위치에 여러 후보의 support가 도착해도 correction을 합산하지 않는다.
support는 `[0, 1]`로 clamp하고, Thermal condition은 도착 weight의 정규화된 평균으로
만든다. 이 규칙으로 겹친 후보가 residual 크기를 무제한 키우지 못하게 한다.

## 8. Thermal-to-RGB local association

각 Thermal P3 위치 `y`의 `D_T(y)`를 query로, 같은 좌표 주변 반경 2의
RGB descriptor `D_R(x)`를 key로 사용한다.

```text
a(y, x) = softmax_x((q(D_T(y)) dot k(D_R(x))) / temperature
                    - distance_prior(y, x))
```

Thermal condition/value는 `D_T(y)`와 `Q_T(y)`에서 만든다. attention을 이용해
Thermal support와 condition을 RGB 좌표에 scatter한다. 이 과정은 보정 위치를
선택하기 위한 local association이며 RGB 또는 Thermal feature map을 warp하지 않는다.
실제 공간 정렬과 최종 융합은 기존 AAM/HOFM이 맡는다.

attention에는 동일 객체 correspondence label을 부여하지 않는다. contrastive
auxiliary loss를 유지할 경우에는 Thermal GT foreground와 신뢰 가능한 background를
구분하는 projection 공간에만 적용하고, 다른 사람을 자동 negative로 사용하지 않는다.

## 9. RGB calibration과 안전 조건

RGB calibration은 기존 spatial FiLM residual의 역할을 유지한다.

```text
[scale, shift] = Film([D_R, H_T])
raw_delta = tanh(scale) * LN(F_R^3) + tanh(shift)
G = transfer_probability(router_inputs)
delta = Cap(S_R * G * raw_delta, residual_scale, F_R^3)
F_R_cal = F_R^3 + delta
```

여기서 `S_R`과 `H_T`는 Thermal support/condition을 local association으로 RGB에
옮긴 결과다. router 입력에는 RGB/Thermal foreground probability, EDL uncertainty,
attention confidence/entropy, context availability, local modality disagreement를 쓴다.
RGB foreground가 낮다는 이유만으로 correction을 0으로 만드는 곱셈 gate는 두지 않는다.

안전 조건은 다음과 같다.

- `S_R == 0`인 위치는 bitwise 가능한 범위에서 exact identity다.
- residual norm은 입력 feature norm과 `residual_scale`로 제한한다.
- support와 route는 `[0, 1]` 범위다.
- overlapping candidate의 condition은 평균하고 support는 최대 1로 제한한다.
- `thermal_out is thermal_in`을 보장해 Thermal P3를 수정하지 않는다.

## 10. Detection-aware utility

별도의 utility classifier를 만들지 않는다. 현재 구현의 실제 GFL 기반
counterfactual supervision을 Thermal-only 후보에 맞게 재사용한다.

학습 batch에서 유효 Thermal 후보 하나를 score 비례로 sample한다. 그 후보에 대해:

- `keep`: 나머지 후보 correction만 적용하고 sampled candidate correction은 제거한다.
- `trial`: 나머지 correction에 sampled candidate의 ungated bounded trial residual을 더한다.
- 두 feature를 동일한 기존 HOFM과 GFL head에 통과시킨다.
- normal branch가 계산한 assignment cache를 재사용해 target 변화가 utility 비교에
  섞이지 않게 한다.

```text
u_target = sigmoid((L_det(keep) - L_det(trial)
                    - lambda_penalty * residual_penalty) / T_utility)
L_utility = BCE(route_prediction, stopgrad(u_target))
L_trial = lambda_trial * L_det(trial)
```

이 비교는 전역 scalar route가 아니라 sampled candidate의 국소 route를 감독한다.
keep branch는 target 생성과 gradient에서 분리하고, trial branch는 correction 경로로
gradient를 전달한다. utility sample을 만들지 못한 image/batch에서는 두 loss를 0으로
두되 로그에 sampling failure를 남긴다.

## 11. Loss 구성

첫 OEPC-v2 실험은 다음 loss만 사용한다.

1. 기존 GFL detection loss.
2. Thermal Gaussian candidate focal loss.
3. RGB/Thermal foreground supervision. 중심 target과 foreground/bag target을 분리한다.
4. 신뢰 가능한 context가 있을 때만 적용하는 object-background contrastive loss.
5. 낮은 가중치의 EDL loss와 EDL uncertainty router input.
6. 실제 GFL keep/trial detection-utility loss.

RGB/Thermal 전체 feature distribution alignment loss나 동일 객체 hard matching loss는
추가하지 않는다. auxiliary loss weight는 별도 config에 명시하고 기존 balanced config를
수정하지 않는다.

## 12. 진단 지표

학습 로그에는 다음을 level별로 기록한다.

- `candidate_probability_mean/max`
- `candidate_threshold_pass_count`
- `soft_support_mass`와 `soft_support_ratio`
- inference 규칙을 모사한 `hard_candidate_count`
- GT Gaussian 중심에서의 candidate recall
- `context_coverage_mean`, `context_available_ratio`
- local attention entropy와 maximum confidence
- mean transfer probability와 transfer-on-support
- `delta_ratio`, residual cap 적용 비율
- utility sample 비율, keep/trial detection loss, utility target/prediction

`candidate_count == max_candidates`, context availability 1.0, attention entropy 1.0,
route가 초기값에 머무는 현상을 조기에 확인할 수 있어야 한다. scalar metric은
보정의 의미적 정확도를 입증하는 지표로 과대 해석하지 않는다.

## 13. 코드 변경 범위

- `mmdet/models/utils/oepc.py`
  - Gaussian target, detail/context/semantic descriptor, Thermal-only candidate,
    train/inference support 분리, bounded association/calibration 구현.
- `mmdet/models/utils/fusion_strategy.py`
  - P3 OEPC에 동일 모달 P4 descriptor context 전달.
  - AAM/HOFM의 실제 입력은 calibrated RGB P3와 원 Thermal P3로 유지.
- `mmdet/models/detectors/fusionnet_xo.py`
  - 기존 fixed-assignment detector utility 경로를 새 payload와 연결.
- `mmdet/models/dense_heads/gflq_head.py`
  - 가능한 한 기존 target cache와 sampled loss를 재사용하며 필요한 최소 변경만 한다.
- `configs/coxnet/oepc/OEPC_v2.py`
  - same-stage FPN, P3-only, detector utility 활성화, 새 hyperparameter 정의.
- `README.md`와 OEPC 문서
  - 실제 구조, 실행 명령, baseline/ablation 구분을 갱신.
- `tests/`
  - 아래 단위·통합 회귀 테스트 추가.

## 14. 테스트 전략

### 14.1 단위 테스트

- 인접한 두 tiny GT의 Gaussian 중심이 모두 보존된다.
- float 중심 이동에 따라 target이 비정상적으로 한 cell에서 사라지지 않는다.
- training soft support를 통한 loss가 candidate head에 non-zero finite gradient를 준다.
- inference top-k가 인접한 두 high-score cell을 3x3 NMS로 제거하지 않는다.
- RGB 입력만 바꿔도 Thermal candidate logits 자체는 변하지 않는다.
- P4를 바꾸면 descriptor/보정은 변하지만 Thermal 출력 tensor는 변하지 않는다.
- support 밖 RGB 출력은 입력과 동일하다.
- residual cap을 넘는 입력에서도 출력 변화가 설정 상한을 넘지 않는다.
- background context가 부족한 boundary/crowded synthetic case에서 availability가 false가 된다.
- overlapping candidate가 residual을 중복 합산하지 않는다.
- padding mask가 candidate, context, attention, support 전체에 적용된다.

### 14.2 통합 테스트

- CPU 작은 tensor로 `FusionLayer` P3/P4 forward/backward가 통과한다.
- detector utility payload가 실제 GFL sampled loss까지 연결되고 route/trial branch에
  non-zero finite gradient가 생긴다.
- training과 `simple_test` 모두 `img_metas` valid mask를 전달한다.
- config build, one-iteration train smoke test, padded inference smoke test가 통과한다.
- 기존 balanced config가 여전히 build되어 과거 실험 재현 경로가 깨지지 않는다.

## 15. 비교 실험과 판정

구현 검증 뒤 GPU 1에서 seed 0을 먼저 실행한다. seed 0이 정상 종료되고 로그/평가가
유효한 것을 확인하기 전에는 seed 1, 2를 자동 시작하지 않는다. 비교 조건은 다음과 같다.

1. 통제된 COXNet baseline.
2. 현재 balanced OEPC (`69d9037` 설정).
3. OEPC-v2 full.
4. 필요 시 `OEPC-v2 without detector utility` 한 가지 ablation.

전체 AP50뿐 아니라 Tiny1/2/3, Small, 후보 수, GT candidate recall, support ratio,
context availability, route, delta를 함께 본다. Tiny1만 상승하고 Tiny2/3·Small이 다시
하락하면 일반적인 대체 성공으로 판단하지 않는다.

## 16. 수용 기준

코드 완료 기준은 다음과 같다.

- CLFM/DWT/IDWT/DeConv/cross-stage pairing이 active config와 실행 경로에 없다.
- RGB/Thermal FPN은 같은 P3-P6 stage를 제공한다.
- 후보 logit은 Thermal descriptor에만 의존한다.
- training support는 differentiable이고 candidate head에 detection gradient가 도달한다.
- inference는 인접 peak를 3x3 NMS로 억제하지 않는다.
- P4는 descriptor에만 들어가고 AAM의 Thermal 입력은 원 Thermal P3다.
- context availability가 합성 invalid case에서 실제로 false가 된다.
- actual GFL keep/trial utility가 활성화되고 gradient/finite test를 통과한다.
- 단위 테스트, config build, forward/backward, padded inference smoke test가 모두 통과한다.
- README가 구현과 일치한다.
- 검증 결과와 남은 한계를 커밋 메시지/보고에 명확히 기록한다.

성능 성공 기준은 코드 수용 기준과 분리한다. 최소한 seed 0에서 balanced OEPC 대비
전체 AP50/Tiny2/Tiny3/Small의 회복 여부와 Tiny1 이득 유지 여부를 확인한다. seed 0의
방향이 유효할 때만 동일 조건 seed 1, 2를 순차 실행한다.

## 17. 배포와 Git 정책

구현은 `/data/siwoo/COXNet-OEPC-v2`의 `oepc-v2-main` branch에서 진행한다.
기존 `/data/siwoo/COXNet-OEPC-balanced-core` 학습 worktree는 수정하지 않는다.

최종 검증 뒤 원격을 다시 fetch하고 다음 조건을 확인한다.

1. 원격 `main`이 이 branch의 기반 이력에 포함된다.
2. 예상하지 못한 원격 변경이 없다.
3. worktree가 clean이고 모든 요구 테스트가 통과한다.

조건을 만족하면 `HEAD:main`으로 fast-forward push한다. 강제 push는 사용하지 않는다.
원격 `main`이 앞서 있으면 push를 중단하고 변경을 검토·통합한 뒤 다시 검증한다.

## 18. 검토한 대안

### A. 기존 dual RGB/Thermal candidate 유지

Thermal에서 놓친 객체를 RGB가 보완할 수 있지만, 이번 모듈의 핵심 가설과 candidate
origin이 다시 섞이고 실제 로그에서 후보 cap이 두 모달 각각 포화됐다. OEPC-v2 첫
검증에서는 Thermal-first causal path를 분명히 하기 위해 제외한다.

### B. straight-through hard support

forward train/inference 차이는 줄지만, 초기 candidate 오차가 hard topology를 고정하고
dense tiny 객체를 다시 억제할 수 있다. 첫 구현은 pure soft training support를 사용하고,
train/inference gap이 실제 문제로 확인될 때만 후속 ablation으로 둔다.

### C. P4를 P3 RGB에 직접 fusion

의미 문맥은 늘지만 RGB P3의 공간 표현을 바꾸고 CLFM과 AAM의 역할이 다시 섞인다.
P4는 descriptor에서 candidate/association 판단에만 사용한다.

### D. full-image utility scalar

구현은 단순하지만 여러 후보 중 어느 correction이 유익했는지 분리하지 못한다.
현재 코드가 이미 제공하는 sampled-candidate actual GFL counterfactual이 국소 router
감독에 더 적합하므로 이를 유지한다.
