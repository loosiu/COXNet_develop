#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ICBFC_PYTHON:-/home/viplab/anaconda3/envs/coxmamba/bin/python}"
config_path="configs/coxnet/icbfc/ICBFC.py"
relative_work_root="work_dir/coxmamba/rgbtdroneperson/icbfc"
work_root="${repo_root}/${relative_work_root}"
data_root="${MMDET_DATASETS:-/data/siwoo/COXNet-release/data/RGBTDronePerson/}"
dry_run="${ICBFC_DRY_RUN:-0}"
allow_existing="${ICBFC_ALLOW_EXISTING:-0}"
seeds=(0 1 2)

export CUDA_VISIBLE_DEVICES="${ICBFC_CUDA_VISIBLE_DEVICES:-0}"
export MMDET_DATASETS="${data_root}"
export MPLCONFIGDIR="${ICBFC_MPLCONFIGDIR:-/tmp/matplotlib-icbfc-training}"
export TORCH_HOME="${ICBFC_TORCH_HOME:-/tmp/torch-icbfc-training}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"

if [[ "${dry_run}" != "1" ]]; then
    exec 9>/tmp/coxnet_icbfc_gpu0.lock
    if ! flock -n 9; then
        echo "GPU 0 ICBFC lock is already held" >&2
        exit 1
    fi
fi

cd "${repo_root}"
expected_commit="$(git rev-parse HEAD)"
for seed in "${seeds[@]}"; do
    relative_work_dir="${relative_work_root}/seed${seed}"
    work_dir="${work_root}/seed${seed}"
    command=(
        "${python_bin}" tools/train.py
        "${config_path}"
        --work-dir "${work_dir}"
        --gpu-id 0
        --seed "${seed}"
        --deterministic
    )

    printf 'RUN seed=%s work_dir=%s command=' "${seed}" "${relative_work_dir}"
    printf '%q ' "${command[@]}"
    printf '\n'

    if [[ "${dry_run}" == "1" ]]; then
        continue
    fi

    if [[ -d "${work_dir}" ]] && \
            [[ -n "$(find "${work_dir}" -mindepth 1 -print -quit)" ]] && \
            [[ "${allow_existing}" != "1" ]]; then
        echo "Refusing non-empty fresh-run directory: ${work_dir}" >&2
        echo "Archive it or set ICBFC_ALLOW_EXISTING=1 explicitly." >&2
        exit 1
    fi
    if [[ "$(git rev-parse HEAD)" != "${expected_commit}" ]]; then
        echo "Repository HEAD changed during the queue; refusing mixed code." >&2
        exit 1
    fi

    mkdir -p "${work_dir}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
    {
        echo "start_time=$(date --iso-8601=seconds)"
        echo "seed=${seed}"
        echo "physical_gpu=${CUDA_VISIBLE_DEVICES}"
        echo "python=${python_bin}"
        echo "config=${config_path}"
        echo "git_commit=${expected_commit}"
    } | tee "${work_dir}/run_manifest.txt"

    "${command[@]}" 2>&1 | tee "${work_dir}/console.log"
    echo "end_time=$(date --iso-8601=seconds)" | \
        tee -a "${work_dir}/run_manifest.txt"
done
