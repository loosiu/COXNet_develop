#!/usr/bin/env bash
# Source in the invoking shell so lock file descriptors survive its GPU task.
exec 9>/tmp/dyfclt_gpu1.lock
flock -n 9
exec 8>/tmp/coxnet_oepc_gpu1.lock
flock -n 8
exec 7>/tmp/coxnet_dwt_dfca_gpu1.lock
flock -n 7
exec 6>/tmp/trpc_gpu1.lock
flock -n 6
exec 5>/tmp/coxnet_icbfc_gpu1.lock
flock -n 5
exec 4>/tmp/yolox_auair_s_1280_gpu1.lock
flock -n 4
RESIDUAL_GPU_UUID="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)"
RESIDUAL_GPU_MEMORY="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)"
RESIDUAL_GPU_JOBS="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)"
if (( RESIDUAL_GPU_MEMORY > 1000 )) || [[ "$RESIDUAL_GPU_JOBS" == *"$RESIDUAL_GPU_UUID"* ]]; then
    echo 'Physical GPU 1 occupied; refusing overlap.' >&2
    exit 1
fi
export RESIDUAL_GPU_UUID CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/residual_mofe_matplotlib
"${PYTHON_BIN:-/home/viplab/anaconda3/envs/coxmamba/bin/python}" - <<'PY_CHECK'
import json,os,subprocess,torch
from pathlib import Path
assert torch.cuda.device_count()==1
probe=torch.zeros(1,device='cuda:0')
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
actual=[row.split(',')[1].strip() for row in rows if int(row.split(',')[0])==os.getpid()]
assert actual==[os.environ['RESIDUAL_GPU_UUID']],actual
Path('artifacts/physical_gpu_mapping.json').write_text(json.dumps(dict(physical_gpu=1,
    uuid=actual[0],process_device='cuda:0',mapping_verified=True),indent=2))
PY_CHECK
