#!/bin/bash
set -euo pipefail
RUN_ROOT=/share/home/u2515283058/Mymodel/output_eval/e2a_3d2d_tex_01_20261026
cd "${RUN_ROOT}/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled OMP_NUM_THREADS=1
ACTION=$1
shift
export E2A_JOB_START=$(date +%s)
if [[ "$ACTION" == prepare ]]; then
  python -m tools.e2a_matched_common
elif [[ "$ACTION" == advance ]]; then
  python -m tools.e2a_matched_dispatch "$@"
elif [[ "$ACTION" == smoke ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 E2A_JOB_SOFT_SECONDS=0
  python -c 'import torch; print("CUDA",torch.cuda.is_available(),"VISIBLE",torch.cuda.device_count(),"EXPECTED",1,flush=True); assert torch.cuda.is_available() and torch.cuda.device_count()==1'
  python -m tools.e2a_matched_smoke
elif [[ "$ACTION" == train ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 E2A_JOB_SOFT_SECONDS=0
  python -c 'import torch; print("CUDA",torch.cuda.is_available(),"VISIBLE",torch.cuda.device_count(),"EXPECTED",2,flush=True); assert torch.cuda.is_available() and torch.cuda.device_count()==2'
  PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port="$PORT" --mixed_precision=fp16 tools/e2a_matched_train.py "$@"
elif [[ "$ACTION" == eval ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1'
  python -m tools.e2a_matched_eval "$@"
else
  echo "Unknown action: $ACTION" >&2
  exit 2
fi
