# 관계 기반 자기 밴드 조절 MoFE

사용자 제공 설계를 실행하는 독립 실험이다. `/data/siwoo/DualRTDETR-MoFE-Staged`에서 실행 중인 2차 및 소스는 수정하지 않는다.

- P3 위치는 post_encoder, 기존 RGB/T fusion 이전이다.
- SWT 앞 256→64 projection, signed SWT, High=Concat(LH,HL,HH), 네 조합과 기존 spatial softmax router를 보존한다.
- 각 방향은 독립 bank이며 강화 전 원본 pair를 동시에 읽는다.
- L/H별 receiver value는 bias 없는 1×1 Conv로 64채널에 투영하고 두 조합이 공유한다.
- receiver/donor 각각 L/H별 1×1 adapter 후 위치별 채널 LayerNorm(affine 없음, eps=1e-5)을 적용한다.
- 관계 descriptor는 concat(U*D, abs(U-D)). 조합별 gate는 PW1→DW3×3→GELU→PW2→2sigmoid, 마지막 PW2를 zero 초기화해 gate=1로 시작한다.
- 조합별 출력 projection은 bias 없는 64→256 1×1 Conv이며 일반 초기화한다. γ만 0으로 초기화한다. G−1을 사용하지 않는다.
- receiver=0이면 donor가 0이 아니어도 해당 방향의 모든 expert 출력은 정확히 0이다.
- donor는 gate와 router에 관여한다. donor 내용을 value로 직접 전달하지 않는다. 이 구조는 공간 정렬 및 DFCA 재현을 주장하지 않는다.
- β head를 추가하지 않는다. loss, seed0, batch2, FP32, EMA, 20epoch/49000updates, 공개 backbone 사전학습 초기화, 데이터/augmentation/optimizer/scheduler는 1·2차와 동일하다.
- 이전 3차가 아직 시작되지 않은 경우에만 대체한다. 2차 정상 종료·20회 검증·best/last metadata를 확인한 뒤 새 실험을 시작한다.

실행 선택: 직접 구현 및 검증. Git 저장소가 없는 복사본이므로 별도 실험 디렉터리로 격리하며 commit 대신 SHA256 source snapshot을 기록한다.
