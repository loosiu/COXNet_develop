#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
(( $# == 0 )) || exit 2
PYTHON_BIN=/home/viplab/anaconda3/envs/coxmamba/bin/python
SPATIAL_RUN_DIR=artifacts/training_launch_spatial_mofe_seed0_gpu1
mkdir -p "$SPATIAL_RUN_DIR"
[[ ! -e "$SPATIAL_RUN_DIR/controller_pid.txt" && ! -e "$SPATIAL_RUN_DIR/launch.json" ]] || exit 2
exec 3>/tmp/spatial_mofe_gpu1.lock
flock -n 3
source tools/gpu1_guard.sh
printf '%s\n' "$$" > "$SPATIAL_RUN_DIR/controller_pid.txt"
SPATIAL_TRAIN_PID=''
trap 'SPATIAL_STATUS=$?; trap - EXIT; printf "%s\n" "$SPATIAL_STATUS" > "$SPATIAL_RUN_DIR/exit_code.txt"' EXIT
trap 'if [[ -n "$SPATIAL_TRAIN_PID" ]]; then kill -TERM "$SPATIAL_TRAIN_PID" 2>/dev/null || true; wait "$SPATIAL_TRAIN_PID" || true; fi; exit 143' TERM INT
"$PYTHON_BIN" -u tools/spatial_mofe_preflight.py > artifacts/preflight_spatial_gpu1.log 2>&1
"$PYTHON_BIN" -u tools/prepare_spatial_mofe_launch.py
"$PYTHON_BIN" -u tools/train.py --config configs/rgbtdroneperson_spatial_mofe_seed0.yaml --device cuda:0 &
SPATIAL_TRAIN_PID=$!
printf '%s\n' "$SPATIAL_TRAIN_PID" > "$SPATIAL_RUN_DIR/pid.txt"
wait "$SPATIAL_TRAIN_PID"
