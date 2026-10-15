# Dual-stream RT-DETR + 관계 기반 자기 밴드 조절 MoFE

상대 모달리티의 콘텐츠를 직접 합성하는 대신, 상대 밴드와의 관계로 **수신 모달리티 자신의 value를 조절**하는 네 expert를 사용한다. 기존 [CNN spatial MoFE](../dual_rtdetr_cnn_mofe_zero/README.md)는 보존한다. COXNet/MMDetection과 독립적으로 실행하는 실험이다.

## 학습 및 검증 실행

이 실험 폴더에서 실행한다. 검증 환경은 Python3.9, Torch1.13.1+cu116, torchvision0.14.1+cu116이다. Torch/torchvision은 CUDA 환경에 맞는 조합을 먼저 설치하고 `pip install -r requirements.txt`로 나머지 의존성을 준비한다.

```text
data/RGBTDronePerson/
  train_thermal.json
  val_thermal.json
  train/visible/  train/thermal/
  val/visible/    val/thermal/
weights/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth
```

데이터를 위 위치에 연결하거나 `configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml`의 `data.train/val.ann_file`, `image_root`를 수정한다. 공개 [PResNet50-vd 가중치](https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth)는 위 `weights/` 경로에 별도로 준비한다. 실제 검증 가중치 SHA256은 `641dc59835d5228e6013ac45acb633592ec825144fb3d78055bc07699f404353`이다.

```bash
cd experiments/dual_rtdetr_self_band_mofe
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v

# 해당 GPU가 비어 있는지 확인한 뒤 독립 20epoch 학습
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u tools/train.py \
  --config configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml \
  --device cuda:0

# AP50로 선택된 best checkpoint의 EMA를 검증
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/evaluate.py \
  --config configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml \
  --checkpoint outputs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0/best.pth \
  --weights ema --device cuda:0 \
  --output-dir outputs/self_band_mofe_best_validation
```

학습 결과는 `outputs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0/{best.pth,last.pth,train.jsonl,validation/metrics.json}`에 저장된다. 기존 nonempty 출력을 덮어쓰지 않는다. 데이터·가중치·checkpoint는 Git에 포함하지 않는다.

## 구조와 변경 범위

```text
RGB/T 독립 backbone → 각 HybridEncoder(AIFI/FPN/PAN) 완료
→ 양쪽 P3에 Self-band MoFE
→ P3/P4/P5별 기존 RGB/T concat+CSP fusion → 기존 RT-DETR decoder

한 방향의 expert bank:
receiver / donor → 기존 256→64 projection → signed SWT
Low = LL, High = Concat(LH,HL,HH)

receiver Low/High → 공유 V_L/V_H (bias 없는 1×1 projection)
receiver/donor Low/High → 밴드별 adapter → 위치별 채널 LayerNorm
→ [U*D, abs(U-D)] → 조합별 PW1→DW3×3→GELU→PW2→2sigmoid
→ gate × receiver value → 조합별 64→256 출력 projection
→ 기존 위치별 softmax 혼합 → receiver + gamma × 혼합 보완량
```

Thermal 강화의 Low–High는 **Thermal Low value를 RGB High와의 관계로 조절**한다. RGB High를 value로 직접 전달하는 경로가 아니다. L–L/L–H는 같은 V_L, H–L/H–H는 같은 V_H를 사용하며 gate와 출력 projection은 네 조합별로 독립적이다. 양방향 bank는 독립 파라미터이고, 둘 다 강화 전 원래 RGB/T pair를 읽는다.

관계 adapter의 출력은 64채널이다. N은 각 위치의 채널 LayerNorm(affine 없음, eps=1e-5)이다. Gate는 위치·채널별 0~2 범위를 가지며 마지막 PW2의 weight/bias를 0으로 초기화해 정확히 1로 시작한다. Value와 출력 projection은 일반 초기화하고, 독립적인 `gamma_rgb`, `gamma_thermal`만 0으로 시작한다. 따라서 초기 예측은 A와 동일하고 첫 backward에서는 gamma, 이후 expert/router와 관계 경로로 gradient가 전달된다.

SWT·네 조합·기존 spatial router·residual과 현재 삽입 위치를 유지한다. 이번 비교는 expert를 바꾸며 별도 위치별 beta head는 추가하지 않는다. 추가 FFT나 Q/K/V attention을 사용하지 않는다. 상대 모달리티는 gate와 router에 관여하고 최종 RGB/T fusion도 유지된다. 이 gate는 공간 정렬이나 교정된 불확실도, DFCA의 정확한 재현을 뜻하지 않는다.

## A와 동일하게 유지한 학습 조건

seed0, batch2, workers4, FP32, 20epoch/49,000updates, 공개 backbone pretrained부터 독립 학습한다. 학습한 A·이전 MoFE·2차 checkpoint로 초기화하지 않는다. AdamW LR2.5e-4, backbone LR배율0.1, weight decay1e-4, 2,000step warmup, gradient clip0.1, EMA0.9999/warmups2,000 및 기존 detection loss(VFL/L1/GIoU, auxiliary/denoising)를 유지한다.

Train resize는 448×576/512×640/576×704, eval은 512×640이다. Thermal-reference person/rider/crowd, train4,900/val1,225장을 사용한다. 별도 utility/evidence/teacher/distillation loss 또는 epoch 준비 단계를 추가하지 않는다.

## 검증과 실험실 순차 실행

원본 실험에서는 새 기능·큐 테스트12개와 관련 회귀 테스트41개가 통과했다. GPU0 실제 train pair로 6회 disposable detection 업데이트를 수행해 다음을 확인했다.

- A의 공유 초기 tensor와 CPU RNG 일치, 초기 prediction의 모듈 bypass와 정확한 일치.
- optimizer에 새 모듈 parameter 전체 포함.
- 모듈80개 parameter tensor 모두 유한하고 나중에 nonzero gradient 관측; 38개 tensor의 실제 값 변경.
- 수신 특징이0이면 상대 특징만으로 보완량을 만들지 않는 value 경로, L/H별 공유, 양방향 원본 입력, 유한하고 반복 가능한 EMA 평가.

검증 가중치는 저장하거나 본 학습에 재사용하지 않았다. **게시 시점 새 실험은 기존 2차 종료 뒤 GPU1에서 시작하도록 대기 중이며, 새 모델의 AP 결과는 아직 없다.** 구조·gradient 검증은 AP 향상 근거가 아니다.

- [원본 focused tests](results/original_focused_tests.txt), [원본 regression tests](results/original_regression_tests.txt), [GPU preflight](results/gpu_preflight.json), [stdout](results/gpu_preflight_stdout.txt), [읽기 전용 최종 review](results/final_review.json).
- [게시 폴더 전체 테스트](results/publication_tests.txt), [게시 검증](results/publication_verification.json), [정확한 소스 대응](results/source_manifest.json).
- `tools/self_band_queue.py`, `tools/self_band_preflight.py`, `tools/staged_queue.py`는 **기존 실험실의 절대 경로·PID·resource lock에 연결된 원본 실행 코드**다. 일반 checkout에서는 위 `train.py`/`evaluate.py` CLI를 사용한다. 실험실 queue는 실행 중인 2차를 끊지 않고, 실제 정상 종료·20회 검증·best/last metadata 확인 후 idle 기존 큐만 퇴역시켜 이전 3차를 대체한다.
- [training_snapshot](training_snapshot/)에는 원래 코드·설정·테스트151개와 검증 certificate, 원본 한국어 학습 가이드를 보존했다. 이 archive의 데이터 경로는 실험실 원본이다.

추후 `prepare_frequency()`의 방향별 cache에서 `relation_gates`, `candidates`, `w`와 실제 `gamma*w*E`를 함께 추출할 수 있다. Gate 차이나 expert 균등 사용만으로 성공을 판단하지 않고 누락·오탐·점수·박스 변화와 연결해 검증한다. 출처·라이선스는 [third_party](third_party/)를 참고한다.
