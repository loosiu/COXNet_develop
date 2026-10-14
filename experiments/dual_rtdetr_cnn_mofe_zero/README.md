# Dual-stream RT-DETR + CNN spatial MoFE, gamma 0 초기화

현재 주 실험은 SWT의 네 CNN expert와 위치별 router를 사용하고 양방향 residual로 원래 특징을 보완한다. MoFE에는 추가 Q/K/V, FFT correlation, cross-attention을 넣지 않는다. RT-DETR의 기존 attention 및 detection loss는 유지한다. 기존 [관계 기반 실험](../dual_rtdetr_relation_mofe/README.md) 및 [이전 spatial 실험](../dual_rtdetr_spatial_mofe/README.md)은 보존한다. COXNet/MMDetection과 독립적으로 실행한다.

```text
RGB/T 독립 backbone → 각 HybridEncoder의 공통 채널 projection
→ 원래 RGB/T projected P3
→ 각 방향의 stride-one Haar/SWT: LL, LH, HL, HH
→ Low=LL, High=Concat(LH,HL,HH)
→ 네 CNN expert: LL–LL / LL–High / High–LL / High–High
→ 위치별 네 expert softmax 혼합
→ R_out=R+gamma_R*Delta_R, T_out=T+gamma_T*Delta_T
→ 기존 AIFI(P5)·FPN·PAN → scale별 RGB/T concat+CSP fusion → 공유 decoder
```

두 방향의 bank는 독립적인 파라미터를 갖고, 강화 전 원래 RGB/T 쌍을 각각 읽는다. 각 bank의 256→64 projection 뒤에 SWT를 수행한다. Expert는 주파수 pair를 concat한 뒤 `1×1 Conv → GELU → 3×3 depthwise Conv → GELU → 1×1 Conv`로 256채널의 보완량을 만든다. 내부 차원은 64다.

Router는 `1×1 Conv → GELU → 1×1 Conv`이며 이미지별 pooling을 사용하지 않는다. 가중치는 `[B,2,4,H,W]`이며 expert 축에서 FP32 softmax를 적용한다. 모든 expert를 계산하는 dense mixture다.

`gamma_R`, `gamma_T`는 독립적인 학습 가능한 scalar이며 **초기값은 0**이다. 고정 multiplier는 1.0이다. 초기 특징과 예측은 baseline과 동일하다. 첫 backward에서는 두 gamma에 탐지 gradient가 전달되고 나머지 모듈 parameter의 탐지 gradient는 0이다. Gamma가 갱신된 다음 step부터 CNN expert와 router에도 탐지 gradient가 전달된다. 별도 epoch 준비 단계나 추가 loss를 사용하지 않는다. Gamma는 부호와 크기를 제한하지 않는다.

## 학습 설정

주 설정: `configs/rgbtdroneperson_cnn_mofe_zero_seed0.yaml`.

A와 seed 0, batch 2, workers 4, FP32, 20 epoch / 49,000 updates, 공개 PResNet50-vd pretrained, AdamW LR 2.5e-4 / backbone LR 배율 .1 / weight decay 1e-4, 2,000-step warmup, gradient clip .1, EMA .9999 / warmups 2,000을 동일하게 유지한다. Train resize는 448×576 / 512×640 / 576×704, eval은 512×640이다. Thermal-reference person/rider/crowd 3개 class, train 4,900 / val 1,225장을 사용한다.

Detection loss는 VFL 1 + L1 5 + GIoU 2 및 기존 auxiliary/denoising loss다. 추가 mask/utility/evidence/teacher/distillation loss는 없다. 학습한 A/B checkpoint를 초기값으로 로드하지 않는다.

## 재현과 실행

실제 검증 환경은 Python 3.9, Torch 1.13.1+cu116, torchvision 0.14.1+cu116이다. COXNet용 환경과 별도로 준비한다. 아래 `python`은 이 환경의 interpreter여야 한다. 실제 실험실 검증은 `/home/viplab/anaconda3/envs/coxmamba/bin/python`을 사용했다.

```bash
cd experiments/dual_rtdetr_cnn_mofe_zero
conda create -n rtdetr-cnn-mofe python=3.9 -y
conda activate rtdetr-cnn-mofe
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

데이터를 별도로 준비하고 YAML의 경로를 수정하거나 다음 위치에 연결한다.

```text
data/RGBTDronePerson/
  train_thermal.json
  val_thermal.json
  train/visible/  train/thermal/
  val/visible/    val/thermal/
```

공개 [PResNet50-vd 가중치](https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth)를 `weights/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth`로 저장한다. 실제 가중치 SHA256은 `641dc59835d5228e6013ac45acb633592ec825144fb3d78055bc07699f404353`이다. 데이터와 weight/checkpoint는 Git에 포함하지 않는다.

GPU1이 사용 가능한지 확인한 뒤, 이 실험 폴더에서 실행한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u tools/train.py \
  --config configs/rgbtdroneperson_cnn_mofe_zero_seed0.yaml --device cuda:0

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/evaluate.py \
  --checkpoint outputs/rgbtdroneperson_cnn_mofe_zero_seed0/best.pth \
  --config configs/rgbtdroneperson_cnn_mofe_zero_seed0.yaml --weights ema \
  --device cuda:0 --output-dir outputs/cnn_mofe_zero_best_validation
```

출력은 `outputs/rgbtdroneperson_cnn_mofe_zero_seed0`이며 기존 nonempty 출력을 덮어쓰지 않는다. 평가에는 AP50로 선택한 best checkpoint의 EMA를 사용한다.

## 검증 기록과 현재 상태

전체 파라미터는 85,286,203 / 모듈 431,242다. 전체 원본 CPU 테스트 235개 중 234개 통과·CUDA 전용 1개 제외. 실제 GPU1의 첫 step에서는 gamma 두 개에 nonzero 탐지 gradient가 전달되고 다음 step부터 모듈의 36개 tensor 모두에 전달됐다. 표준 warmup의 세 번 갱신 뒤 28개 tensor의 값 변경을 확인했다. 기존 관계 기반 및 spatial 소스 259개는 변경되지 않았다.

**게시 시점에는 20-epoch 본학습과 AP 검증을 수행하지 않았다.** 구조와 gradient 검증은 AP 향상 근거가 아니다.

- [GPU1 preflight](results/gpu_preflight.json), [stdout](results/gpu_preflight_stdout.txt), [물리 매핑](results/physical_gpu_mapping.json): batch 2·세 해상도로 세 번의 실제 detection loss 갱신. 초기 prediction의 baseline 일치와 gamma 이후 expert/router gradient 전달을 검증한다. 검증 가중치를 저장하거나 본학습에 재사용하지 않는다.
- [전체 원본 CPU 테스트](results/original_full_tests.txt), [certificate](results/original_full_test_certificate.json), [게시 폴더 검증](results/publication_verification.json), [게시 테스트](results/publication_tests.txt).
- [Source manifest](results/source_manifest.json): 모델·학습·평가 코드는 검증한 원본과 byte 단위로 동일하다. YAML은 데이터 경로만 상대 경로로 바꿨고 integration A 설정 참조는 포함된 fixture로 바꿨다.
- [training_snapshot](training_snapshot/)에는 실제 실험실의 GPU guard/preflight와 전체 원본 코드·설정·테스트 archive를 보존했다. 이 preflight는 실험실 절대 경로/hash 기록을 요구하므로 일반 checkout에서는 위 CLI를 사용한다.

기존 CNN module/model/학습/평가 코드를 재사용하고 주 설정이 `spatial_mofe`를 선택한다. 관계 기반 파일이 보존되어 있어도 현재 active MoFE에 관계 expert가 등록되거나 실행되는 것은 아니다. S³AM에서 영감을 받은 detection 적용이며 원본 S³AM 전체 구현과 동일하다고 주장하지 않는다. 라이선스와 출처는 [third_party](third_party/)에 있다.
