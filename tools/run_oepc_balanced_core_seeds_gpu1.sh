#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/siwoo/COXNet-OEPC-balanced-core"
PYTHON_BIN="/home/viplab/anaconda3/envs/coxmamba/bin/python"
CONFIG_PATH="configs/coxnet/oepc/OEPC_balanced_core.py"
RUN_ROOT="work_dir/coxmamba/rgbtdroneperson/oepc_balanced_core"
GPU_PHYSICAL_ID=1
LOCK_PATH="/tmp/coxnet_oepc_gpu1.lock"

cd "${REPO_DIR}"
EXPECTED_COMMIT="$(git rev-parse HEAD)"
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-oepc-balanced-core"
export TORCH_HOME="/tmp/torch-oepc-training"

mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}" "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/queue.log") 2>&1

echo "queue_time=$(date --iso-8601=seconds)"
echo "waiting_for_physical_gpu=${GPU_PHYSICAL_ID}"
echo "lock=${LOCK_PATH}"
echo "expected_commit=${EXPECTED_COMMIT}"

exec 9>"${LOCK_PATH}"
flock 9

echo "lock_acquired_time=$(date --iso-8601=seconds)"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
    echo "Repository HEAD changed while this run was queued; refusing mixed-code training." >&2
    exit 1
fi

"${PYTHON_BIN}" -c \
    "import torch; assert torch.cuda.is_available(); print('cuda_device=' + torch.cuda.get_device_name(0))"

for SEED in 0 1 2; do
    WORK_DIR="${RUN_ROOT}/seed${SEED}"
    mkdir -p "${WORK_DIR}"
    {
        echo "start_time=$(date --iso-8601=seconds)"
        echo "seed=${SEED}"
        echo "physical_gpu=${GPU_PHYSICAL_ID}"
        echo "python=${PYTHON_BIN}"
        echo "config=${CONFIG_PATH}"
        echo "git_commit=${EXPECTED_COMMIT}"
    } | tee "${WORK_DIR}/run_manifest.txt"

    "${PYTHON_BIN}" tools/train.py "${CONFIG_PATH}" \
        --work-dir "${WORK_DIR}" \
        --gpu-id 0 \
        --seed "${SEED}" \
        --deterministic \
        2>&1 | tee "${WORK_DIR}/console.log"

    echo "end_time=$(date --iso-8601=seconds)" | \
        tee -a "${WORK_DIR}/run_manifest.txt"
done

echo "queue_complete_time=$(date --iso-8601=seconds)"
