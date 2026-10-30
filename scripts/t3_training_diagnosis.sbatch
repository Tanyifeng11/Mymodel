#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == "worker_pair" ]]; then
  # 账户同时只允许运行两个GPU作业；一个作业内两张卡分别执行独立分片。
  IFS=',' read -r -a diag_devices <<< "$CUDA_VISIBLE_DEVICES"
  [[ ${#diag_devices[@]} -eq 2 ]]
  CUDA_VISIBLE_DEVICES="${diag_devices[0]}" python -m tools.t3_training_diagnosis worker --shard 0 --shards 3 &
  diag_pid0=$!
  CUDA_VISIBLE_DEVICES="${diag_devices[1]}" python -m tools.t3_training_diagnosis worker --shard 1 --shards 3 &
  diag_pid1=$!
  set +e
  wait "$diag_pid0"; diag_status0=$?
  wait "$diag_pid1"; diag_status1=$?
  [[ "$diag_status0" == 0 && "$diag_status1" == 0 ]]
else
  python -m tools.t3_training_diagnosis "$@"
fi
