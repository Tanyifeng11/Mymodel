#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_local_texture_pilot_20261101/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == fit ]]; then
  python -m tools.t3_local_pilot train --arm "$2"
  python -m tools.t3_local_pilot generate --arm "$2"
elif [[ "$1" == preflight ]]; then
  python -m tools.t3_local_pilot audit
  python -m tools.t3_local_pilot generate --arm T3
else
  python -m tools.t3_local_pilot "$@"
fi
