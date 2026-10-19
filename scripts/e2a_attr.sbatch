#!/bin/bash
set -euo pipefail
RUN_ROOT=/share/home/u2515283058/Mymodel/output_eval/e2a_attr_01_20261019
cd "${RUN_ROOT}/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
ACTION=$1
shift
if [[ "$ACTION" == prepare ]]; then
  python -m tools.e2a_attr_prepare all --workers "${SLURM_CPUS_PER_TASK:-8}"
elif [[ "$ACTION" == train || "$ACTION" == smoke ]]; then
  GPUS=${E2A_GPUS:-1}
  PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  EXTRA=()
  if [[ "$GPUS" -gt 1 ]]; then EXTRA+=(--multi_gpu); fi
  if [[ "$ACTION" == smoke ]]; then EXTRA_ARGS=(--smoke); else EXTRA_ARGS=(); fi
  accelerate launch "${EXTRA[@]}" --num_processes="$GPUS" --num_machines=1 \
    --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port="$PORT" --mixed_precision=fp16 \
    tools/e2a_attr_train.py "$@" "${EXTRA_ARGS[@]}"
elif [[ "$ACTION" == eval ]]; then
  python -m tools.e2a_attr_eval "$@"
elif [[ "$ACTION" == stats ]]; then
  python -m tools.e2a_attr_stats "$@"
else
  echo "未知动作: $ACTION" >&2
  exit 2
fi
