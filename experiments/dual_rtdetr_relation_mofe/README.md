# 관계 기반 양방향 MoFE: 학습 가능한 residual 보정

기존 [A/이전 B](../dual_rtdetr_mofe/README.md) 및 [spatial MoFE](../dual_rtdetr_spatial_mofe/README.md)는 보존한다. 이 실험은 COXNet/MMDetection과 독립적으로 실행한다. 이번 모듈은 projected P3에서 SWT의 네 주파수 조합을 유지하며, 각 expert를 Q/K 관계 기반 후보 생성으로 바꾼다. 현재 주 실행 설정은 원래 P3를 유지하고 관계 expert의 보완량을 학습 가능한 강도로 더한다. 기존 direct 및 고정 0.5 multiplier 대조 설정은 보존한다.

```text
RGB/T 독립 backbone → 공통 채널 projection → 원래 RGB/T P3
→ 각 방향의 SWT LL/LH/HL/HH
→ 네 관계 expert: LL–LL / LL–H / H–LL / H–H
→ 위치별 softmax로 후보 보완량을 혼합
→ 원래 RGB/T P3 + 방향별 학습 가능한 gamma × 보완량
→ 기존 각 HybridEncoder의 AIFI(P5)·FPN·PAN
→ 기존 scale별 RGB/T concat+CSP fusion → 공유 RT-DETR decoder
```

두 방향은 원래 pair를 읽고 파라미터를 공유하지 않는다. 방향별 기존 256→64 projection과 spatial router 초기값, A의 공통 초기 state와 CPU RNG를 보존한다. Expert 연산은 다음과 같다.

1. 수신 모달리티의 해당 band에서 Q, 상대 모달리티의 해당 band에서 K/V를 각각 `1×1 Conv → 3×3 DWConv`로 만든다. H는 기존처럼 `Concat(LH,HL,HH)`다.
2. `C=IFFT(FFT(Q)*conj(FFT(K)))`로 channel별 circular cross-correlation을 구한다. FFT는 FP32로 수행한다.
3. 이미지·channel별 공간 RMS로 C를 정규화하고 `A=sigmoid(Conv3×3(C_norm))`를 얻는다.
4. `U=A*V`와 수신 정보 Q를 concat하고 channel LayerNorm과 1×1 projection으로 256채널의 완성된 후보 `F_hat_i`를 만든다. 원본 256채널 특징을 그대로 더하는 identity skip은 없다.
5. 기존 spatial router가 네 주파수 pair에서 `[B,2,4,H,W]` 가중치를 만들고 expert 축으로만 FP32 softmax를 적용한다.

현재 주 실행 설정의 출력:

`Delta_R(p)=sum_i w_R_i(p)*F_hat_R_i(p)`

`R_out(p)=R(p)+gamma_R*Delta_R(p)`

`T_out(p)=T(p)+gamma_T*Delta_T(p)`

`gamma_R/T`는 서로 독립적인 scalar `nn.Parameter`이며 초기값은 각각 0.1이다. 위치별 주파수 선택은 router가 담당하고 전체 보정 강도는 gamma가 담당한다. Gamma에는 범위 제한을 두지 않으며 기존 detection loss로 expert/router와 함께 학습한다. 고정 multiplier는 1.0이므로 추가적인 0.5 감쇠는 없다. Gamma를 0으로 두면 원본 특징과 baseline 예측을 유지한다.

보존한 direct 대조 모드는 `R_new=sum_i w_R_i*F_hat_R_i`, `T_new=sum_i w_T_i*F_hat_T_i`다. Direct 모드에는 gamma와 고정 0.5 multiplier가 없으며, 원본에 보완량을 더하는 경로도 없다. 공통 feature projection/backbone/pyramid/decoder의 초기 파라미터는 A와 같지만 새 P3를 넣으므로 초기 예측이 A와 같지는 않다. Padding/invalid cell만 원래 특징을 반환하며, 모든 유효 위치는 새 후보의 혼합으로 교체한다.

DFCA의 Q/K 주파수 상관과 V modulation 원리에 영감을 받았다. RMS normalization, Q와 attended V의 joint projection, SWT 네 pair와 spatial MoE는 이 실험의 적용 선택이다. 원본 DFCA의 학습 가능한 radial FFT band partition이나 SSE는 추가하지 않는다. SWT 분해는 한 번이며 이후 FFT는 correlation 계산용이다. FFT 상관 결과는 이동량(displacement lag)에 대한 지도이며, 이를 학습 가능한 convolution으로 변환하여 V를 조절한다. 각 RGB 위치가 주변 thermal 위치를 직접 검색·집계하는 local token softmax attention이나 기하학적 alignment를 보장하는 모듈은 아니다. S³AM 또는 DFCA 저자 구현의 완전 재현을 주장하지 않는다.

## 설정과 비교

- 주 실행: `configs/rgbtdroneperson_relation_learned_residual_seed0.yaml`. 원본 + gamma × 보완량이며 multiplier=1.0이다.
- 보존한 직접 교체 대조: `configs/rgbtdroneperson_relation_direct_seed0.yaml`.
- 대조 설정: `configs/rgbtdroneperson_relation_residual_seed0.yaml`. 같은 관계 expert와 router를 사용하되 `R+0.5*gamma_R*mixed`, `T+0.5*gamma_T*mixed`로 연결한다. Gamma는 0.1에서 시작하며 residual 모드에서만 optimizer에 등록한다. 이 모드에서는 후보를 보완량으로 학습하므로 별도로 처음부터 학습해야 한다.
- 기존 A와 seed 0, batch 2, workers 4, FP32, 20 epoch, 공개 backbone pretrained, AdamW/EMA, resize, 검출·auxiliary·denoising loss를 동일하게 유지한다. 추가 mask/utility/distillation/teacher loss는 없다.
- 현재 주 실행 파라미터: 전체 85,789,499 / 모듈 934,538. Direct 대조는 전체 85,789,497 / 모듈 934,536으로 gamma 2개가 없다.
- 현재 convolution+residual 모델과는 후보 생성과 출력 연결 두 항목이 바뀐다. 관계 expert residual 대조를 사용하면 후보 생성 변경과 출력 교체를 단계적으로 비교할 수 있다. Residual보다 direct가 반드시 우수하다는 주장은 하지 않는다.

## 재현과 실행

실제 검증 환경은 Python 3.9, Torch 1.13.1+cu116, torchvision 0.14.1+cu116이다. COXNet용 환경과 별도로 준비한다. 아래 명령의 `python`은 이 환경의 interpreter여야 한다. 원래 실험실에서는 `/home/viplab/anaconda3/envs/coxmamba/bin/python`으로 검증했다.

```bash
cd experiments/dual_rtdetr_relation_mofe
conda create -n rtdetr-relation-mofe python=3.9 -y
conda activate rtdetr-relation-mofe
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

데이터는 별도로 준비하고 YAML의 경로를 수정하거나 다음 위치에 연결한다.

```text
data/RGBTDronePerson/
  train_thermal.json
  val_thermal.json
  train/visible/  train/thermal/
  val/visible/    val/thermal/
```

실제 paired 파일 이름 및 폴더 선택은 `dyfclt/data.py`를 따른다. 공개 [PResNet50-vd 가중치](https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth)를 `weights/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth`로 저장한다. 실제 사용한 weight의 SHA256은 `641dc59835d5228e6013ac45acb633592ec825144fb3d78055bc07699f404353`이다.

아래 명령은 이 실험 폴더에서 실행한다. GPU1이 사용 가능한지 확인하고 실행한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v

# 주 실험: 원본 + 학습 가능한 gamma × 보완량. 공개 pretrained에서 20 epoch 새로 학습.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u tools/train.py \
  --config configs/rgbtdroneperson_relation_learned_residual_seed0.yaml --device cuda:0

# 보존한 0.5 residual 대조. 자동으로 실행하지 않는다.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u tools/train.py \
  --config configs/rgbtdroneperson_relation_residual_seed0.yaml --device cuda:0

# AP50로 선택한 best checkpoint의 EMA 평가.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/evaluate.py \
  --checkpoint outputs/rgbtdroneperson_relation_learned_residual_seed0/best.pth \
  --config configs/rgbtdroneperson_relation_learned_residual_seed0.yaml --weights ema \
  --device cuda:0 --output-dir outputs/relation_learned_residual_best_validation
```

세 설정은 서로 다른 output directory를 사용하며 기존 nonempty 출력을 덮어쓰지 않는다. A와 기존 spatial 모델의 설정도 `configs/`에 포함한다. 주 실험과 대조 실험 모두 학습한 A/B checkpoint를 초기값으로 로드하지 않는다.

## 검증 기록과 현재 상태

현재 learned-residual 설정: 전체 CPU 테스트 230개 중 229개 통과·CUDA 전용 1개 제외, 이 게시 폴더의 테스트 36개 통과. 실제 GPU1 검증에서 gamma 두 개를 포함한 모듈의 파라미터 텐서 100개 모두 유효한 gradient와 갱신을 확인했다.

**게시 시점에는 20-epoch 본학습과 AP 검증을 수행하지 않았다.** 실제 GPU 갱신 확인은 구현 검증이며 성능 향상 근거가 아니다.

- 직접 교체형을 구현한 시점의 원본 전체 CPU 테스트: 226개 실행, 225개 통과 / CUDA 전용 1개 제외. [로그](results/original_full_tests.txt), [certificate](results/original_full_test_certificate.json).
- 직접 교체형을 구현한 시점의 [실제 GPU1 preflight](results/gpu_preflight.json): batch 2·세 해상도로 세 번의 검출 loss 갱신. 모듈의 파라미터 텐서 98개 모두 유한한 nonzero gradient와 값 변경 확인, EMA 평가 유한·반복 가능. [물리 GPU 매핑](results/physical_gpu_mapping.json), [stdout](results/gpu_preflight_stdout.txt).
- A의 공통 초기 state tensor 1,401개와 CPU RNG가 동일하며 기존 spatial 실험의 원본 소스 125개를 보존했다. Direct 모드의 초기 예측은 A와 다르다.
- 기존 게시 기록은 [direct 게시 검증](results/publication_verification.json), [direct 게시 테스트 로그](results/publication_tests.txt), [direct source manifest](results/source_manifest.json)에 보존한다. 해당 hash는 그 시점의 source archive에 대응한다.
- 현재 learned-residual 설정의 검증은 [GPU1 preflight](results/learned_residual_gpu_preflight.json), [전체 CPU 테스트](results/full_tests_learned_residual.txt), [게시 검증](results/publication_verification_learned_residual.json), [게시 테스트 로그](results/publication_tests_learned_residual.txt), [source manifest](results/source_manifest_learned_residual.json)에 기록한다.
- [training_snapshot](training_snapshot/)에는 실제 실험실의 GPU guard·preflight와 전체 원본 코드/설정/테스트 archive를 보존했다. 이 preflight는 절대 경로와 원본 hash 기록이 필요하므로 일반 checkout에서는 위의 CLI를 사용한다. 검증에서 갱신한 가중치는 저장하거나 본학습에 재사용하지 않는다.

Model·학습·평가 소스는 검증한 원본과 byte 단위로 동일하다. YAML은 데이터 경로만 상대 경로로 바꿨고, integration 테스트의 A 설정 참조는 포함된 fixture로 바꿨다. 데이터와 pretrained·학습 checkpoint는 Git에 포함하지 않는다. 라이선스와 출처는 [third_party](third_party/)에 있다.

Learned-residual 모드는 `residual_scale`, `residual_multiplier`, `effective_residual_scale`을 함께 기록한다.

진단에는 router 평균/공간 표준편차/entropy, 관계 attention의 평균/포화 비율, 후보·출력·원본 RMS, 원본 대비 실제 변화량이 포함된다. Direct 모드의 `relative_update`는 `(새 특징−원래 특징)`의 RMS 비율이며 residual 보완량 크기를 뜻하지 않는다. 초기 학습 loss나 attention 가중치만으로 AP 향상을 주장하지 않는다.
