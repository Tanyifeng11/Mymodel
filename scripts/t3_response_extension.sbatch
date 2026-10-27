#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_reference_response_20261027/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export E2A_JOB_SOFT_SECONDS=0
ACTION=$1
if [[ "$ACTION" == train ]]; then
  KEY=$2
  python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==2'
  PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port="$PORT" --mixed_precision=fp16 \
    tools/t3_response_extension.py train --key "$KEY"
  python -m tools.t3_response_extension evaluate --key "$KEY"
else
  python -m tools.t3_response_extension "$ACTION"
fi
