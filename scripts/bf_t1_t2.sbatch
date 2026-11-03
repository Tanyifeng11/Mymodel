#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/bf_t1_t2_20261103
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == calibration ]]; then
  test "$(python -c 'import torch;print(torch.cuda.device_count())')" = 2
  for arm in BF-SD-T1 BF-SD-T2; do
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.bf_t1_t2 calibrate --arm "$arm"
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.bf_t1_t2 preflight --arm "$arm"
  done
elif [[ "$1" == train ]]; then
  test "$(python -c 'import torch;print(torch.cuda.device_count())')" = 2
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.bf_t1_t2 formal --arm "$2"
  python -m tools.bf_t1_t2 generate --arm "$2"
else
  python -m tools.bf_t1_t2 "$1"
  if [[ "$1" == process ]]; then python scripts/bf_t1_t2_submit.py evaluate; fi
  if [[ "$1" == evaluate ]]; then python scripts/bf_t1_t2_submit.py package; fi
fi
