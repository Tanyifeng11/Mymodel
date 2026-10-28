#!/bin/bash
set -euo pipefail
PROJECT=/share/home/u2515283058/LOTS
cd "$PROJECT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1 HF_HOME="$PROJECT/cache/huggingface"
if [[ "$1" == "fetch" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  exec /share/home/u2515283058/.conda/envs/texfusion-dit/bin/python run_experiment.py fetch
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  exec "$PROJECT/.venv/bin/python" run_experiment.py "$@"
fi
