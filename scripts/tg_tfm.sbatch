#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/tg_tfm_e2_20261105
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == train ]]; then
  python -m tools.tg_tfm_experiment train --arm "$2" --gpus "$3"
  python -m tools.tg_tfm_experiment generate --arm "$2"
elif [[ "$1" == evaluate ]]; then
  python -m tools.tg_tfm_experiment evaluate
  python -m tools.tg_tfm_experiment package
else
  python -m tools.tg_tfm_experiment "$@"
fi
