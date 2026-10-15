# 관계 기반 자기 밴드 조절 MoFE 실험

실행 중인 2차 donor-value MoFE를 보존하고, 아직 시작하지 않은 3차 donor+β 대신 이 실험을 GPU1에서 독립적으로 학습한다.

## 학습 설정과 출력

- seed=0, batch=2, 20 epoch, 49,000 optimizer updates, FP32, EMA AP50로 best 선택.
- A와 동일한 공개 ResNet50-vd backbone 사전학습부터 시작. A/1차/2차 checkpoint는 읽어 초기화하지 않는다.
- RGBTDronePerson train=4,900 / val=1,225, 세 클래스 person/rider/crowd, 기존 augmentation·optimizer·scheduler·검출 loss 유지.
- config: `configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml`
- 출력: `outputs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0/best.pth`, `last.pth`, `validation/metrics.json`, `train.jsonl`.
- 실행 로그: `artifacts/training_launch_self_band_seed0_gpu1/train.log`, 정상 종료 시 `exit_code.txt=0`.
- 순차 실행 상태: `artifacts/queue_self_band/status.json`.

## 구조

```text
RGB/T backbone → 각 HybridEncoder(AIFI/FPN/PAN) → P3 자기 밴드 조절 MoFE
→ 원래 P3/P4/P5별 RGB/T concat+CSP fusion → 원래 RT-DETR decoder

각 방향: receiver / donor → 기존 256→64 projection → signed SWT
Low=LL, High=Concat(LH,HL,HH)
receiver L/H → 공유 V_L/V_H
receiver/donor L/H → 밴드별 adapter → 위치별 채널 LayerNorm
→ [곱, 절대 차이] → 조합별 2sigmoid gate → receiver value 조절
→ 조합별 64→256 projection → 기존 spatial softmax 혼합
→ 원래 receiver + γ × 보완량
```

양방향 bank는 독립 파라미터를 사용하고 모두 강화 전 원래 pair를 읽는다. `γ_R=γ_T=0`, 관계 gate=1, value/output은 일반 초기화한다. 추가 attention·FFT·위치별 β·utility/evidence loss는 없다. 상대 모달리티는 gate와 router에 관여하며 value는 receiver에서만 온다. 이 gate는 정렬이나 교정된 신뢰도를 의미하지 않는다.

## GPU1에서 순차 학습하는 방법

준비된 supervisor가 실행 중이면 중복 실행하지 않는다. 이 명령은 2차 종료 후에만 본 학습을 시작한다.

```bash
cd /data/siwoo/DualRTDETR-SelfBand-MoFE
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/viplab/anaconda3/envs/coxmamba/bin/python -u tools/self_band_queue.py
```

Supervisor는 기존 GPU1 resource lock이 해제될 때까지 기다린다. 2차의 정상 exit, 20회 전체 검증, best/last checkpoint·EMA metadata를 확인하고 기존 큐가 이전 3차를 시작하기 직전에 idle 상태인 경우에만 퇴역시킨다. 검증 certificate와 source/input SHA가 일치해야 다음 명령을 **새 프로세스**로 실행한다.

```bash
cd /data/siwoo/DualRTDETR-SelfBand-MoFE
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/viplab/anaconda3/envs/coxmamba/bin/python -u tools/train.py \
  --config configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml \
  --device cuda:0
```

아래 학습 명령은 supervisor가 물리 GPU1(UUID `GPU-c1faf7b9-2a34-010b-9739-4712a6f089a2`)의 점유·잠금을 확인한 뒤 실행하는 내부 명령이다. 기존 학습이 돌고 있을 때 별도로 실행하지 않는다.

## 검증 근거

- `artifacts/self_band_tests.log`: 새 expert·queue의 의미 검증.
- `artifacts/regression_tests.log`: 기존 spatial MoFE, detection recipe, queue completion 등의 회귀 검사.
- `artifacts/gpu_preflight.json`: GPU0 실제 train pair에서 detection loss로 6회 disposable 업데이트. A 공유 초기값/RNG, 초기 예측 동일성, optimizer 포함, 모든 module parameter의 유한/nonzero gradient, EMA 검증을 기록한다.
- `artifacts/training_certificate.json`: 본 학습 직전에 검증하는 source/input/config/검증 artifact 해시.

GPU0 smoke 가중치는 저장하지 않으며 본 학습에 사용하지 않는다. smoke 통과는 AP 향상의 증거가 아니다. γ=0 및 gate=1 초기화로 첫 backward에서는 γ만 열리고, 그 뒤 expert와 gate 경로가 순차적으로 gradient를 받는다.

추후 시각화는 `prepare_frequency()`의 방향별 cache에서 `candidates`, `relation_gates`, `w`를 읽고 실제 `γ*w*E` 및 검출 변화를 함께 비교한다. 활성화의 밝기나 expert 균등 사용만으로 성공을 판단하지 않는다.
