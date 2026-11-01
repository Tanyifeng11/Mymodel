#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/bf_fusion_t3_three_arm_20261101
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == train ]]; then
  N=$(python -c 'import torch;print(torch.cuda.device_count())')
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N" -m tools.bf_t3_three_arm train --arm "$2"
  python -m tools.bf_t3_three_arm generate --arm "$2"
elif [[ "$1" == closeout ]]; then
  python -m tools.bf_t3_three_arm evaluate
  python -m tools.bf_t3_three_arm package
else
  python -m tools.bf_t3_three_arm "$@"
fi
