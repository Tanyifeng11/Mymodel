#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028/source
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == prepare || "$1" == generate ]]; then
  export PATH=/share/home/u2515283058/.conda/envs/sgdiff/bin:$PATH
else
  export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
fi
python -m tools.t3_sgdiff_external "$@"
