# Same-stage TRPC 설계 및 문제 분석

## 결론

CLFM을 완전히 제거한 same-stage TRPC 경로를 구현했다. RGB와 Thermal은
모두 stride 8/16/32/64 feature를 사용하며, 각 동일 stride 쌍에 TRPC를
적용한 뒤 기존 AAM/HOFM으로 전달한다. 이 경로에는 DWT, IDWT, 주파수
융합, CLFM DeConv, shape 보정용 interpolation이 없다.

다만 기존 cross-stage TRPC 체크포인트를 분석한 결과, **검출 성능은
유지됐지만 학습된 prototype residual은 사실상 사용되지 않았다.** 따라서
기존 Table 1 결과를 prototype calibration의 효과로 해석할 수 없다.
same-stage 재설계는 이 문제를 자동으로 해결하는 방법이 아니라, CLFM과
DeConv의 영향을 제거하고 TRPC 자체를 검증하기 위한 더 깨끗한 실험
조건이다.

## 구조 변경

| 항목 | 기존 COXNet | 첫 TRPC 실험 | Same-stage TRPC |
|---|---:|---:|---:|
| RGB stride | 16/32/64/128 | 16/32/64/128 | **8/16/32/64** |
| Thermal stride | 8/16/32/64 | 8/16/32/64 | 8/16/32/64 |
| RGB-Thermal pairing | cross-level | cross-level | **same-level** |
| CLFM 주파수 연산 | 사용 | 제거 | 제거 |
| CLFM DeConv | 사용 | 유지 | **제거** |
| Prototype calibration | 없음 | 사용 | 사용 |
| 이후 AAM/HOFM | 유지 | 유지 | 유지 |

Same-stage 경로는 다음과 같다.

```text
RGB P3/P4/P5/P6 ─┐
                  ├─ same-stride TRPC ─ AAM/HOFM ─ detector head
T   P3/P4/P5/P6 ─┘
```

`TRPC_same_stage.py`는 RGB FPN의 `start_level`을 2에서 1로 바꾼다.
TRPC 내부의 `visible_upsample`은 `Identity`이며, 양 모달 feature shape가
다르면 interpolation하지 않고 즉시 오류를 낸다. 따라서 잘못된 level
pairing을 조용히 허용하지 않는다.

## 파라미터 확인

동일한 로컬 코드에서 측정한 trainable/non-trainable 전체 파라미터 합이다.

| 구성 | 전체 모델 | 교체 대상 모듈 | 그중 DeConv |
|---|---:|---:|---:|
| COXNet CLFM | 71,335,628 | 10,493,952 | 1,050,624 |
| Cross-stage TRPC | 62,198,552 | 1,356,876 | 1,050,624 |
| **Same-stage TRPC** | **61,279,256** | **306,252** | **0** |
| Same-stage no-TRPC control | 60,973,004 | 0 | 0 |

Same-stage TRPC core는 기존 CLFM보다 약 97.1% 작고, 전체 모델은 약
14.1% 작다. 이는 파라미터 감소이며 FPS 또는 FLOPs 개선을 직접 의미하지
않는다.

## 기존 TRPC에서 실제로 확인된 문제

### 1. Calibration residual이 사실상 비활성 상태다

Cross-stage TRPC seed 0의 best checkpoint(`epoch_10.pth`)에서 validation
앞 128장, 4개 level의 512개 관측을 고정밀도로 다시 측정했다.

| 진단값 | 측정값 |
|---|---:|
| `match_rate` | 0.272705 |
| `match_confidence` | 0.065261 |
| `gate_mean` | 0.032573 |
| **실제 feature `delta_ratio` 평균** | **7.30e-10** |
| `delta_ratio` 중앙값 | 7.03e-11 |
| `delta_ratio` 최댓값 | 5.41e-09 |
| prototype cosine 변화(after-before) | -0.000329 |

로그의 `delta_ratio=0.0000`은 단순 표시 자릿수 문제가 아니었다. 실제 AAM
입력에 더해진 보정은 원 feature norm 대비 약 10억 분의 1 수준이었다.

같은 checkpoint에서 `delta_to_rgb.weight`만 0으로 만들어 전체 validation
1,225장을 재평가했다. AAM으로 들어가는 원래 Thermal feature는 유지했다.

두 평가는 같은 현재 코드와 GPU 1에서 실행했다.

| 지표 | 원 checkpoint | TRPC residual off | 변화(%p) |
|---|---:|---:|---:|
| AP25 | 59.87 | 59.87 | 0.00 |
| AP50 | 46.25 | 46.25 | 0.00 |
| AP75 | 5.45 | 5.45 | 0.00 |
| tiny | 47.96 | 47.96 | 0.00 |
| tiny1 | 9.08 | 9.08 | 0.00 |
| tiny2 | 36.82 | 36.96 | +0.14 |
| tiny3 | 53.20 | 53.19 | -0.01 |
| small | 29.49 | 29.49 | 0.00 |

AP50은 보고 자릿수에서 완전히 같고 세부 구간의 차이도 매우 작다. 이
결과는 기존 TRPC의 성능이 prototype 보정보다 DeConv와 기존 Thermal
AAM/HOFM 경로에 의해 유지됐다는 해석을 강하게 지지한다. 단, 이는 seed 0
best checkpoint에 대한 추론 개입이며 별도 학습 대조군을 대체하지 않는다.

### 2. 보정 신호가 여러 곱셈 항에서 동시에 약해진다

실제 residual에는 RGB objectness, mutual-match confidence, channel gate,
output projection, residual scale이 연속으로 적용된다. 기존 checkpoint에서
평균 confidence는 0.065, confidence가 반영된 gate 평균은 0.033 수준이었다.
RGB objectness까지 곱해지므로 zero-initialized projection이 학습 중 거의
열리지 않으면 모든 상위 prototype 경로의 detector gradient도 함께
약해진다.

특히 RGB가 약하고 Thermal만 선명한 위치에서는 RGB objectness도 낮을 수
있다. 현재 재투영은 바로 그 위치의 Thermal 보정을 RGB objectness로 다시
억제할 수 있다.

### 3. Mutual matching은 의미적으로 맞는 객체 대응을 보장하지 않는다

RGB와 Thermal에는 서로 독립적인 embedding projection이 있고, 양 embedding을
동일한 의미 좌표계로 만드는 직접 목적함수는 없다. `targetness`는 Thermal
객체 영역, `diversity`는 prototype 간 비유사성, detector loss는 최종 검출을
감독하지만 어느 것도 “RGB prototype i와 Thermal prototype j가 같은
객체/의미”임을 직접 감독하지 않는다.

`match_rate`도 mutual top-1 교집합이 존재하는 비율일 뿐 대응 정확도가
아니다. 128장 probe에서 선택된 cosine 평균은 0.1855였지만, GT 객체 단위의
올바른 대응인지는 이 값으로 판단할 수 없다.

### 4. Prototype이 분화됐다기보다 넓게 퍼져 있다

같은 probe에서 정규화 attention entropy는 RGB 0.9096, Thermal 0.8940이고,
유효 prototype 사용량은 8개 중 7.73개였다. 따라서 한두 prototype으로
붕괴한 상태는 아니지만, 각 prototype이 구분되는 국소 객체를 담당한다고
볼 증거도 없다. 전역적으로 비슷하게 넓은 영역을 보는 8개 prototype이면
밀집 객체 분리와 경계 전달에 도움이 제한적일 수 있다.

### 5. 전역 rank-K 보정은 밀집 객체의 국소 정보를 제한할 수 있다

공간과 채널을 행렬로 펼치면 재투영 residual의 rank는 최대 prototype 수
`K=8`이다. RGB 원 feature는 보존되지만 Thermal에서 새로 전달하는 보정
정보는 최대 8개 벡터의 조합으로 제한된다. 가까운 여러 사람의 서로 다른
경계와 중심을 전달해야 하는 COXNet의 밀집 위치 오류에는 이 제약이 불리할
가능성이 있다. 현재 결과만으로 원인이라고 확정할 수는 없으며, 객체별
대응/박스 폭 분석으로 검증해야 한다.

### 6. Thermal stop-gradient는 correctness가 아니라 설계 선택이다

`P_T.detach()` 때문에 RGB calibration의 detector gradient는 Thermal
prototype extractor로 흐르지 않는다. Thermal extractor는 주로 targetness와
diversity loss로 학습되고, raw Thermal feature/backbone은 별도의 AAM/HOFM
경로로 detector gradient를 받는다. Reference 안정성과 의미 표현 학습 사이의
trade-off이며, stop-gradient 자체가 좋은 Thermal reference를 보장하지 않는다.

## Same-stage가 해결하는 것과 해결하지 않는 것

| 항목 | Same-stage 변경의 효과 |
|---|---|
| CLFM/DeConv와 TRPC 효과의 혼재 | 제거 |
| cross-level semantic 차이 | 제거 |
| level 해상도 보정/interpolation | 제거 |
| residual 비활성화 | **해결 보장 없음** |
| semantic matching 정확성 | **해결 안 됨** |
| RGB objectness에 의한 억제 | **해결 안 됨** |
| rank-8 국소 정보 제한 | **해결 안 됨** |

따라서 same-stage 구조의 목적은 성능 향상을 미리 주장하는 것이 아니라,
TRPC의 순수 기여를 검증 가능한 형태로 만드는 것이다.

## 반드시 필요한 대조 실험

두 설정은 RGB/Thermal stride, AAM/HOFM, detector head, assignment, loss 및
학습 recipe가 같고 TRPC 유무만 다르다.

```text
configs/coxnet/trpc/same_stage_no_trpc.py  # control
configs/coxnet/trpc/TRPC_same_stage.py     # treatment
```

판정 순서는 다음과 같다.

1. seed 0에서 두 설정을 각각 학습한다.
2. TRPC가 control보다 나을 때만 seeds 1/2를 진행한다.
3. 학습된 same-stage TRPC에서 residual-off 및 Thermal-prototype shuffle을
   수행한다.
4. 전체 AP와 함께 밀집 객체의 recall, 중심 오차, 폭 비율, 후처리 탈락률을
   비교한다.

TRPC가 유효하다고 판단하려면 최소한 다음 세 조건이 함께 필요하다.

- `TRPC_same_stage > same_stage_no_trpc`가 seed에 걸쳐 재현될 것.
- `delta_ratio`가 수치적으로 의미 있는 크기로 열리고 residual-off에서
  성능 또는 목표 오류 지표가 악화될 것.
- 실제 RGB-Thermal 객체 대응 정확도 개선과 밀집 위치 오류 감소가 같은
  사례에서 연결될 것.

반대로 control과 같고 residual-off에도 변화가 없다면, prototype 수나 loss를
추가하기 전에 현재 multiplicative gate와 재투영 설계를 수정하거나 TRPC를
폐기하는 편이 타당하다.

## 검증 상태

- TRPC 단위 테스트 10개 통과.
- 전체 detector에서 두 모달 feature shape가 모든 level에서 동일함을 확인:
  `(16,20)`, `(8,10)`, `(4,5)`, `(2,3)`.
- same-stage TRPC에 DeConv/IDWT 파라미터가 없음을 확인.
- zero initialization에서 TRPC 출력이 입력 RGB feature와 정확히 같음을 확인.
- 같은 seed에서 control과 treatment의 AAM/HOFM 초기값이 동일함을 확인.
- 전체 detector의 synthetic `forward_train`/backward를 통과하고 첫 step의
  `delta_to_rgb` gradient가 유한한 비영 값임을 확인.
- 아직 `TRPC_same_stage.py`와 control의 새 학습은 수행하지 않았다.

기존 cross-stage checkpoint는 RGB FPN level 구성과 TRPC 파라미터 구성이
달라 same-stage 모델에 재사용할 수 없다.
