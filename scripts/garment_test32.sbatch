#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028/source
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
PYTHON=/share/home/u2515283058/.conda/envs/Mymodel/bin/python
if [[ "$1" == generate && "${2:-}" == BF-Fashion ]]; then
  PYTHON=/share/home/u2515283058/.conda/envs/ldm/bin/python
elif [[ "$1" == generate && "${2:-}" == SGDiff ]]; then
  PYTHON=/share/home/u2515283058/.conda/envs/sgdiff/bin/python
elif [[ "$1" == generate && "${2:-}" == LOTS ]]; then
  PYTHON=/share/home/u2515283058/LOTS/.venv/bin/python
  export HF_HOME=/share/home/u2515283058/LOTS/cache/huggingface
fi
exec "$PYTHON" -m tools.garment_test32 "$@"
