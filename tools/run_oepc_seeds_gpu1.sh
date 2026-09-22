#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/siwoo/TRPC-Thermal-Referenced-Prototype-Calibration"
PYTHON_BIN="/home/viplab/anaconda3/envs/coxmamba/bin/python"
CONFIG_PATH="configs/coxnet/oepc/OEPC_same_stage.py"
RUN_ROOT="work_dir/coxmamba/rgbtdroneperson/oepc_detector_utility"
GPU_PHYSICAL_ID=1
LOCK_PATH="/tmp/coxnet_oepc_gpu1.lock"

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL_ID}"
export MPLCONFIGDIR="/tmp/matplotlib-oepc-training"
export TORCH_HOME="/tmp/torch-oepc-training"

mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}" "${RUN_ROOT}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "GPU ${GPU_PHYSICAL_ID} OEPC lock is already held: ${LOCK_PATH}" >&2
    exit 1
fi

for SEED in 0 1 2; do
    WORK_DIR="${RUN_ROOT}/seed${SEED}"
    mkdir -p "${WORK_DIR}"
    {
        echo "start_time=$(date --iso-8601=seconds)"
        echo "seed=${SEED}"
        echo "physical_gpu=${GPU_PHYSICAL_ID}"
        echo "python=${PYTHON_BIN}"
        echo "config=${CONFIG_PATH}"
        echo "git_commit=$(git rev-parse HEAD)"
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
