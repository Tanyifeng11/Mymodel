#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export E2A_JOB_SOFT_SECONDS=0
ACTION=$1
shift
if [[ "$ACTION" == train ]]; then
  PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port="$PORT" --mixed_precision=fp16 \
    -m tools.t3_controlled_continue train
elif [[ "$ACTION" == worker_pair ]]; then
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  [[ ${#devices[@]} -eq 2 ]]
  CUDA_VISIBLE_DEVICES="${devices[0]}" python -m tools.t3_controlled_continue worker --shard 0 --shards 2 &
  pid0=$!
  CUDA_VISIBLE_DEVICES="${devices[1]}" python -m tools.t3_controlled_continue worker --shard 1 --shards 2 &
  pid1=$!
  set +e
  wait "$pid0"; status0=$?
  wait "$pid1"; status1=$?
  [[ "$status0" == 0 && "$status1" == 0 ]]
else
  python -m tools.t3_controlled_continue "$ACTION" "$@"
fi
