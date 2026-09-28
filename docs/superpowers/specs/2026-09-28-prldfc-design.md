# PRLDFC: Prototype-Routed Local Dynamic Frequency Calibration 설계

## 1. 목적

PRLDFC는 COXNet의 CLFM을 완전히 제거한 same-stage FPN에서, Thermal 객체 근거가
어떤 주파수 대역에서 RGB 표현을 보강해야 하는지를 객체 조건으로 결정하는
pre-AAM calibration 모듈이다.

```text
RGB/Thermal same-stage features
  -> Thermal one-to-one dense seed field
  -> detail-preserving local prototype field
  -> learnable low/mid/high frequency feature maps
  -> band-wise Thermal-to-RGB local association
  -> prototype-routed band/reliability weights
  -> bounded RGB residual
  -> original AAM/HOFM(calibrated RGB, original Thermal)
```

PRLDFC는 다음 연구 질문을 검증한다.

> Tiny 객체마다 유용한 주파수 대역과 신뢰할 모달이 다르므로, 모든 위치에 동일한
> frequency fusion을 적용하는 것보다 Thermal 객체 조건으로 local cross-modal
> frequency transfer를 조절하는 편이 밀집·tiny 객체의 표현을 더 잘 보강하는가?

성공은 전체 AP 증가만으로 정의하지 않는다. matched same-stage control보다 성능이
재현 가능하게 높고, 객체별 band routing과 RGB residual이 실제로 사용되며, 밀집
객체의 중심·폭 오류가 감소해야 한다.

## 2. 범위와 불변 조건

### 2.1 유지하는 COXNet 구성

- RGB와 Thermal backbone은 기존 dual ResNet-50을 유지한다.
- RGB/Thermal FPN 모두 `start_level=1`로 P3/P4/P5/P6를 출력한다.
- 기존 AAM/HOFM, DSR, GFL head, QLSAssigner, test-time NMS를 변경하지 않는다.
- Thermal feature tensor와 좌표는 PRLDFC에서 수정하지 않는다.
- PRLDFC는 강화된 RGB와 원본 Thermal을 기존 AAM/HOFM에 전달한다.
- baseline과 loss recipe를 맞추기 위해 첫 통제 실험은 `wf_loss=True`를 유지한다.

### 2.2 제거하는 CLFM 구성

- RGB/Thermal cross-stage pairing
- DeConv resolution matching
- Haar DWT/IDWT 재구성
- CLFM LL fusion과 high-frequency gate
- CLFM의 모든 parameter 복사·초기화 경로

### 2.3 첫 구현에서 제외하는 요소

- hard top-k candidate quota와 local-peak NMS
- EDL, contrastive, cycle, prototype alignment/diversity loss
- detector counterfactual utility router
- RGB objectness를 correction의 필수 gate로 사용하는 구조
- Mamba/SSM과 별도 Transformer decoder
- spatial feature warp와 별도 deformable alignment
- patch-wise FFT와 phase alignment

마지막 두 항목은 기존 AAM의 alignment 역할을 침범하지 않기 위한 제한이다.

## 3. 적용 level과 실험 단계

구현은 임의의 same-stage level을 지원한다. canonical treatment는 P3/P4
`apply_levels=(0, 1)`이고, 인과 분리를 위해 P3-only `apply_levels=(0,)` config도
제공한다. P5/P6은 첫 실험에서 PRLDFC를 거치지 않고 기존 AAM/HOFM으로 바로 간다.

- P3: 매우 작은 객체와 국소 경계 보강
- P4: tiny2/tiny3 및 small 객체의 semantic context 보완
- P5/P6: 구현 복잡도와 고주파 noise를 제한하기 위해 control과 동일하게 유지

P3-only가 P3/P4보다 낫다면 canonical 범위를 P3로 축소한다. level 확장은 seed 0
ablation으로 결정하며 가정으로 고정하지 않는다.

level별 seed supervision은 resize 이후 입력 좌표의 `sqrt(w*h)`로 구분한다. 초기
scale range는 P3 `[0, 32)` px, P4 `[16, 64)` px이며 overlap 구간의 GT는 두 level을
모두 감독한다. `N_gt_l`은 해당 level의 eligible GT 수다. 이 범위는 detector assignment를
바꾸지 않으며 PRLDFC seed auxiliary target에만 적용된다.

## 4. One-to-one dense Thermal seed field

각 적용 level의 Thermal feature에서 seed existence, sub-cell center offset, scale을
예측한다.

```text
Z_l(x), O_l(x), S_l(x) = SeedHead_l(F_T^l)(x)
P_l(x) = sigmoid(Z_l(x))
```

- `Z_l`: object-seed existence logit
- `O_l in [-0.5, 0.5]^2`: feature-cell 내부 중심 offset
- `S_l`: positive box width/height의 log-scale

### 4.1 One-to-one assignment

학습 시 각 GT 중심 주변 반경 2 cell을 seed 후보로 만든다. GT와 후보 cell 사이에
다음 cost를 사용해 one-to-one matching한다.

```text
cost(g, x) = lambda_center * L1(c_g, x + O_l(x))
           + lambda_score  * focal_cost(Z_l(x))
           + lambda_scale  * L1(log size_g, S_l(x))
```

GT와 전체 후보 cell의 bipartite cost matrix에 deterministic Hungarian assignment를
적용한다. 동률은 `(GT index, flattened cell index)` 순으로 안정적으로 해소한다.

- 한 GT는 최대 한 seed cell에 할당된다.
- 한 seed cell은 최대 한 GT에 할당된다.
- 두 GT의 정수 중심이 같아도 주변 cell과 offset을 이용해 서로 다른 seed를 가질 수 있다.
- valid padding 밖 cell은 후보와 negative에서 제외한다.
- ignore box와 겹치는 cell은 negative에서 제외한다.
- 후보 반경 안에 유일한 cell이 부족한 GT는 `unassigned_gt`로 기록하며 억지로 중복
  할당하지 않는다.

모든 unmatched valid cell은 negative다. 단, 서로 다른 GT 후보 집합의 교집합에서
matching cost 차이가 작은 ambiguous cell은 classification loss에서 ignore한다.

### 4.2 Top-k 없는 soft gate

학습과 기본 추론 경로는 고정 개수 candidate를 gather하지 않는다.

```text
G_l(x) = sigmoid((Z_l(x) - logit(tau_seed)) / T_seed)
```

`G_l`은 dense soft gate이며 모든 연산이 미분 가능하다. 추론의 효율 측정을 위한
선택적 sparse mode에서만 `sigmoid(Z_l) >= threshold` cell을 처리한다. 이때도 후보
수는 영상 내용에 따라 0개 이상으로 변하며 top-k quota를 사용하지 않는다.

메모리 안전장치는 score top-k가 아니라 candidate chunking으로 구현한다. 디버그용
`emergency_max_candidates`는 기본 비활성화하고, 발동하면 evaluation 결과와 로그에
명시해 정상 결과로 취급하지 않는다.

### 4.3 Cardinality calibration

existence focal loss와 함께 probability mass가 객체 수와 과도하게 어긋나지 않도록
약한 cardinality loss를 사용한다.

```text
L_count = abs(sum_x P_l(x) - N_gt_l) / max(N_gt_l, 1)
```

이는 객체 수를 고정하지 않는다. 영상별 GT 수를 target으로 하므로 zero-object와
극밀집 장면을 모두 허용한다.

## 5. Detail-preserving prototype field

prototype은 Thermal 정보를 RGB residual로 직접 복원하지 않는다. 객체·주변 대비와
RGB/Thermal reliability를 요약해 frequency router의 조건으로만 사용한다.

각 위치의 shared embedding을 만든다.

```text
E_M^l = LN(Conv1x1(F_M^l)),  M in {R, T}
D_M^l = E_M^l - ValidAvgPool3(E_M^l)
C_M^l = ValidAvgPool3(E_M^l) - ValidAvgPool7(E_M^l)
```

padding을 제외한 normalized pooling을 사용한다. Thermal prototype field는 predicted
offset 위치를 bilinear sample해 구성한다.

```text
P_T^l(x) = MLP_T([sample(E_T^l, x + O_l(x)),
                  sample(D_T^l, x + O_l(x)),
                  sample(C_T^l, x + O_l(x)),
                  S_l(x)])
```

이 구조는 P3의 단순 3x3 average가 입력 기준 약 24x24 영역의 배경과 이웃 객체를
함께 평균하던 문제를 피한다. `E`는 중심 의미, `D`는 local detail, `C`는
object-context 차이를 나타낸다.

RGB condition은 nominal Thermal 중심 주변의 broadband local attention으로 읽는다.

```text
A_0^l(x, y) = softmax_y(cos(q(P_T^l(x)), k(E_R^l(y))) / temp_0)
P_R^l(x) = sum_{y in N_r(x)} A_0^l(x, y) * v(E_R^l(y))
```

이 attention은 동일 객체 correspondence label로 해석하지 않는다. router가 주변 RGB
상태를 읽는 제한된 condition이고 feature를 warp하지 않는다.

## 6. Differentiable frequency-band decomposition

첫 구현은 계산량과 재현성을 위해 feature-map FFT를 한 번 수행한 뒤 spatial map으로
복원한다. `local`은 FFT patching이 아니라 객체 support 안의 association과 correction을
의미한다. Patch-wise FFT는 별도 ablation 전까지 구현하지 않는다.

적용 level마다 RGB/Thermal을 `frequency_dim`으로 투영하고 orthonormal FFT를 수행한다.

```text
X_M^l = Conv1x1(F_M^l)
Xhat_M^l = rfft2(X_M^l, norm="ortho")
```

2D `rfft2`의 모서리까지 포함해 normalized radial frequency가 정확히
`rho in [0, 0.5]`가 되도록 다음처럼 정의하고, 세 개의 soft mask를 사용한다.

```text
rho(f_x, f_y) = 0.5 / sqrt(2)
                * sqrt((f_x / 0.5)^2 + (f_y / 0.5)^2)
```

```text
M_b(rho) = sigmoid((rho - k_b) / T_f)
         - sigmoid((rho - k_{b+1}) / T_f)
```

경계는 `k_0=0`, `k_3=0.5`로 고정한다. `B=3`, 최소 폭 `w_min`일 때 학습 가능한
logit `a_b`에서 band width와 경계를 다음처럼 만들어 단조 증가와 최소 폭을 동시에
보장한다.

```text
width_b = w_min + (0.5 - B * w_min) * softmax(a)_b
k_0 = 0
k_{b+1} = k_b + width_b
```

초기 경계는 `{0, 1/8, 1/4, 1/2}`가 되도록 `a_b`를 초기화한다. 이 parameterization은
`sum_b width_b=0.5`와 `width_b>=w_min`을 만족해 band collapse를 막는다.

```text
B_M^{l,b} = irfft2(M_b * Xhat_M^l, size=(H_l, W_l), norm="ortho")
```

세 mask의 합은 주파수 전 범위에서 1에 가깝도록 정규화한다. reconstruction test는
`sum_b B_M^{l,b}`가 projection feature `X_M^l`과 tolerance 안에서 같은지 확인한다.

```text
M_b <- M_b / (sum_j M_j + eps)
```

band 경계는 level별로 공유되는 전역 학습 파라미터다. PRLDFC에서 `dynamic`은 seed마다
달라지는 band routing weight와 reliability를 뜻하며, 객체마다 FFT 경계를 새로
예측한다는 뜻은 아니다.

DyFCLT와 달리 Q/K/V 각각에 별도 frequency mask를 만들지 않는다. 동일 band map에서
spatial Q/K/V projection을 만들어 Q-only frequency leakage와 불필요한 FFT 반복을
피한다.

## 7. Band-wise Thermal-to-RGB local association

각 Thermal seed 위치와 band마다 radius `r_l`의 RGB neighborhood에서 local attention을
계산한다.

```text
A_b^l(x, y) = softmax_y(
    dot(q_b(P_T^l(x), B_T^{l,b}(x)), k_b(B_R^{l,b}(y))) / sqrt(d_b)
)

H_R^{l,b}(x) = sum_y A_b^l(x, y) * v_b(B_R^{l,b}(y))
T_obj^{l,b}(x) = v_b(B_T^{l,b}(x + O_l(x)))
```

기본 search radius는 P3/P4 모두 2 cell이다. attention은 RGB 위치를 고르는 데만
사용하며 `max(attention)`을 별도 confidence로 다시 곱하지 않는다. attention entropy는
router 입력과 진단값으로만 사용한다.

band residual source는 dense frequency feature에서 만든다.

```text
R_i^{l,b} = ResidualMLP_b([T_obj^{l,b}, H_R^{l,b},
                           T_obj^{l,b} - H_R^{l,b}])
```

prototype은 위 residual vector를 직접 생성하지 않는다.

## 8. Prototype router

router는 객체별 band weight와 Thermal transfer reliability를 예측한다.

```text
U_l(x) = [P_T^l(x), P_R^l(x), P_T^l(x) - P_R^l(x),
          objectness=P_l(x), scale=S_l(x),
          entropy(A_0), contrast_T, contrast_R]

W_l(x) = softmax(BandRouter_l(U_l(x)))       # three bands
Q_l(x) = sigmoid(ReliabilityRouter_l(U_l(x))) # one reliability
```

- `W_l`은 low/mid/high 합이 1인 객체별 band routing weight다.
- `Q_l`은 Thermal evidence를 RGB에 전달할지 판단하는 단일 reliability다.
- RGB objectness를 필수 multiplier로 사용하지 않는다.
- candidate probability와 attention maximum을 추가 multiplier로 반복 적용하지 않는다.
- router는 detection loss로 학습하며 pseudo utility label이나 별도 router loss를 두지 않는다.

마지막 router bias는 `Q=0.5`가 되도록 0으로 초기화한다. band router는 균등한
`1/3`에서 시작한다. 특정 band를 사전 우선하지 않는다.

## 9. Bounded RGB residual

각 seed의 band residual을 prototype routing으로 결합한다.

```text
R_l(x) = Q_l(x) * sum_b W_l^b(x) * R_i^{l,b}(x)
```

RGB 위치로의 broadcast에는 band attention 평균을 사용한다.

```text
A_l(x, y) = sum_b W_l^b(x) * A_b^l(x, y)
mass_l(y) = sum_x G_l(x) * A_l(x, y)

avg_l(y) = sum_x G_l(x) * A_l(x, y) * R_l(x)
           / (mass_l(y) + eps)

support_l(y) = 1 - exp(-mass_l(y))
delta_l(y) = epsilon_l * support_l(y) * tanh(avg_l(y))
F_R_cal^l = F_R^l + delta_l
```

이 aggregation은 다음을 보장한다.

- soft seed gate는 한 번만 residual mass에 적용된다.
- local attention은 위치 선택에만 쓰이며 confidence로 중복 감쇠되지 않는다.
- 중복 seed는 residual 크기를 선형 증폭하지 않고 normalized average된다.
- dense soft gate 때문에 학습 중 background support는 수치적으로 0에 가까울 수 있지만
  정확히 0일 필요는 없다. thresholded sparse inference에서 active seed가 없는 위치는
  exact identity다.
- `epsilon_l`이 correction의 channel-wise 절댓값 상한을 제공한다.

기본 `epsilon_l=0.1`은 고정하고 residual projection은 `Normal(0, 1e-3)`의 작은 비영
초기화를 사용한다. 초기 identity를 강제하는 zero-init은 사용하지 않는다.

최종 출력은 다음과 같다.

```text
(F_R_cal^l, F_T^l) -> original AAM/HOFM
```

## 10. 학습 objective와 `wf_loss`

첫 treatment의 전체 loss는 다음과 같다.

```text
L = L_detection
  + lambda_seed   * L_seed_focal
  + lambda_offset * L_center_offset
  + lambda_scale  * L_seed_scale
  + lambda_count  * L_cardinality
  + L_wf
```

초기 weight는 다음으로 시작한다.

- `lambda_seed=0.10`
- `lambda_offset=0.05`
- `lambda_scale=0.02`
- `lambda_count=0.01`
- 기존 `wf_loss_mode='kl_v2'`, `wf_loss_weight=0.1`

`wf_loss`는 PRLDFC의 novelty나 필수 구성으로 주장하지 않는다. COXNet training recipe를
맞추기 위한 controlled constant다. PRLDFC가 matched control을 이긴 뒤에만
`wf_loss on/off`를 별도 ablation한다.

기존 `wf_loss`는 객체/level 수로 정규화되지 않고 Thermal을 KL target으로 사용하므로,
그 수식 개선은 PRLDFC 첫 구현과 묶지 않는다.

## 11. 진단 지표

### 11.1 Seed quality

- image별 GT 수, probability mass, threshold candidate 수
- matched/unassigned GT 비율
- GT별 candidate recall과 candidate/GT ratio
- 한 GT 주변의 duplicate seed 수
- predicted center offset 및 scale error
- P3/P4별 cardinality error

### 11.2 Frequency behavior

- level별 learned boundaries `k_1`, `k_2`
- RGB/Thermal low/mid/high energy ratio
- object size·이웃 거리·모달 가시성별 band weight
- band usage entropy와 collapsed-band 비율
- static/learned/prototype-routed band 결과

### 11.3 Transfer behavior

- broadband 및 band별 attention entropy
- Thermal reliability `Q`의 평균·분포
- RGB/Thermal local contrast에 따른 `Q`
- support ratio와 overlap mass
- raw residual, projected residual, final `delta_ratio`
- residual cap 도달 비율

이 값들은 모듈 활성 상태를 보여줄 뿐 성능 향상이나 correspondence 정확도를 스스로
입증하지 않는다.

## 12. 대조 실험과 go/no-go 순서

모든 모델은 같은 backbone/head/assignment/augmentation/epoch/seed와
`wf_loss=True`를 사용한다.

### A. Same-stage no-calibration control

```text
RGB/T P3-P6 -> original AAM/HOFM
```

CLFM 제거와 RGB FPN `start_level=1` 변경의 비용을 측정한다.

### B. Local dynamic frequency calibration without prototype routing

- one-to-one dense seed와 local band association은 유지한다.
- band weight는 모든 객체에서 `1/3`으로 고정한다.
- reliability는 1로 고정한다.

frequency transfer 자체의 순수 효과를 측정한다.

### C. Full PRLDFC

- prototype router가 object-specific band weight와 reliability를 예측한다.

prototype conditioning의 추가 이득을 측정한다.

### D. Level ablation

- P3 only
- P3 + P4

### E. Frequency ablation

- one full band
- static three bands `{0, 1/8, 1/4, 1/2}`
- globally learnable three bands
- globally learnable bands + object-conditioned routing

### 진행 규칙

1. seed 0에서 A/B/C를 먼저 비교한다.
2. B가 A보다 낮으면 prototype을 추가로 튜닝하지 않고 frequency path를 진단한다.
3. C가 B보다 높지 않으면 prototype router를 최종 방법에서 제거한다.
4. C가 A/B보다 목표 지표에서 개선될 때만 seeds 1/2를 실행한다.
5. full PRLDFC checkpoint에서 residual-off와 band-shuffle inference intervention을 한다.

## 13. 성공 및 반증 기준

PRLDFC를 유효하다고 판단하려면 다음 조건이 함께 필요하다.

- full PRLDFC가 matched same-stage control보다 seed에 걸쳐 재현 가능하게 높다.
- prototype-routed C가 fixed-router B보다 높다.
- candidate count가 maximum capacity에 포화되지 않고 GT 수와 함께 변한다.
- dense crowd subset에서 GT candidate recall이 유지되며 duplicate/GT가 감소한다.
- `delta_ratio`가 비영이고 residual-off에서 성능 또는 목표 오류가 악화된다.
- band shuffle 또는 Thermal feature shuffle에서 성능이 악화된다.
- 같은 크기에서도 밀집 객체의 중심·폭 오류가 control보다 감소한다.

다음 결과는 핵심 가설을 약화한다.

- C와 B가 같음: prototype routing의 추가 가치 없음
- B와 A가 같음: local frequency transfer의 추가 가치 없음
- residual-off 변화 없음: calibration path가 사용되지 않음
- band weight가 항상 균등하거나 한 band로 붕괴: object-conditioned routing 실패
- reliability 개선에도 background FP 증가: Thermal transfer 선택 실패
- AP는 오르지만 밀집 위치 오류가 그대로임: 최초 failure hypothesis와 불일치

## 14. 코드 경계

기존 TOPC 결과와 재현 경로는 보존한다.

- 새 모듈: `mmdet/models/utils/prldfc.py`
- 새 integration flag: `use_prldfc`, `prldfc_cfg`
- 새 config:
  - `configs/coxnet/prldfc/PRLDFC.py`
  - `configs/coxnet/prldfc/PRLDFC_p3.py`
  - `configs/coxnet/prldfc/same_stage_no_calibration.py`
- integration:
  - `mmdet/models/utils/fusion_strategy.py`
  - `mmdet/models/detectors/fusionnet_xo.py`
- unit tests: `tests/test_models/test_utils/test_prldfc.py`
- method doc: `docs/PRLDFC.md`
- README에는 canonical config와 결과 상태만 반영한다.

`use_clfm`, `use_trpc`, `use_oepc`, `use_topc`, `use_prldfc`는 상호 배타적이다.
기존 실험 config/checkpoint/log를 덮어쓰지 않는다.

## 15. 테스트 요구사항

### Seed와 geometry

- one-to-one assignment가 GT/seed 중복 할당을 만들지 않음
- 같은 정수 중심의 두 GT가 가능한 경우 다른 seed+offset을 받음
- zero GT, ignore GT, padding, boundary box에서 finite 동작
- cardinality target과 valid mask가 resize/padding 좌표와 일치

### Frequency decomposition

- band boundary가 단조 증가하고 최소 폭을 만족
- 세 mask 합과 band reconstruction이 tolerance를 만족
- RGB/Thermal shape·dtype·device 보존
- odd/even feature size에서 `rfft2/irfft2` shape 보존
- band parameter와 projection에 비영 gradient가 흐름

### Router와 residual

- 초기 band weight는 `1/3`, reliability는 `0.5`
- RGB objectness가 없어도 Thermal seed가 correction을 허용
- attention maximum을 residual amplitude에 중복 곱하지 않음
- overlapping seed에서 residual이 선형 폭증하지 않음
- thresholded sparse inference에서 no-support 위치는 exact identity
- dense training의 background residual은 seed prior와 `epsilon_l` bound 안에 있음
- Thermal output은 bitwise unchanged
- residual의 channel-wise 절댓값은 `epsilon_l` bound를 넘지 않음
- detection loss에서 seed, band, router, residual projection에 비영 gradient가 흐름

### Integration

- active PRLDFC graph에 CLFM/DeConv/DWT/IDWT가 없음
- RGB/Thermal FPN shape는 P3-P6에서 같음
- calibrated RGB와 원 Thermal이 기존 AAM/HOFM으로 전달됨
- train/inference 모두 padding mask를 전달함
- legacy TOPC/TRPC/OEPC focused tests가 통과함
- GPU batch forward/backward/optimizer smoke가 통과함

## 16. 초기 config

첫 구현의 시작값이며 성능 주장값이 아니다.

```python
model = dict(
    neck=dict(start_level=1),
    neck_t=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
    use_oepc=False,
    use_topc=False,
    use_prldfc=True,
    wf_loss=True,
    wf_loss_mode='kl_v2',
    wf_loss_weight=0.1,
    prldfc_cfg=dict(
        apply_levels=(0, 1),
        frequency_dim=64,
        prototype_dim=64,
        num_bands=3,
        search_radius=(2, 2),
        seed_prior=0.01,
        seed_threshold=0.10,
        seed_temperature=0.25,
        frequency_temperature=0.02,
        min_band_width=0.05,
        level_scale_ranges=((0, 32), (16, 64)),
        residual_epsilon=(0.1, 0.1),
        seed_loss_weight=0.10,
        offset_loss_weight=0.05,
        scale_loss_weight=0.02,
        cardinality_loss_weight=0.01,
    ),
)
```

## 17. 선행 연구와 차별점

### DyFCLT (CVPR 2026)

DyFCLT의 learnable low/mid/high frequency interaction과 background suppression 결과를
출발점으로 삼는다. PRLDFC는 layer-global symmetric fusion 대신 object-conditioned
asymmetric RGB calibration을 수행하고 기존 AAM을 유지한다.

### LFBNet (CVPR 2026)

LFBNet의 local frequency와 misalignment 문제의식을 참고하지만 phase alignment와
deformable fusion은 도입하지 않는다. PRLDFC의 local association은 feature를 warp하지
않고 correction support만 결정한다.

### 기존 TOPC

TOPC는 prototype discrepancy를 직접 RGB residual로 만들었다. PRLDFC는 prototype을
band/reliability router로 제한하고 dense band feature가 실제 정보를 전달한다. 고정
top-100, duplicate cell prototype, max-attention confidence, 3x3 average prototype을
사용하지 않는다.

## 참고 문헌

- Chaolang Li et al., "DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning
  Transformer for Multimodal Tiny Object Detection," CVPR 2026.
  <https://openaccess.thecvf.com/content/CVPR2026/html/Li_DyFCLT_Dynamic_Frequency-Decoupled_Cross-Modal_Learning_Transformer_for_Multimodal_Tiny_Object_CVPR_2026_paper.html>
- Shenghui Huang et al., "UAV-CB: A Complex-Background RGB-T Dataset and Local
  Frequency Bridge Network for UAV Detection," CVPR 2026.
  <https://openaccess.thecvf.com/content/CVPR2026/html/Huang_UAV-CB_A_Complex-Background_RGB-T_Dataset_and_Local_Frequency_Bridge_Network_CVPR_2026_paper.html>
- Haodong Zhu et al., "WaveMamba: Wavelet-Driven Mamba Fusion for RGB-Infrared
  Object Detection," ICCV 2025.
  <https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_WaveMamba_Wavelet-Driven_Mamba_Fusion_for_RGB-Infrared_Object_Detection_ICCV_2025_paper.html>
