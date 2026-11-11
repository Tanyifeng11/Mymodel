#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_time_gradient_20261111
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == prepare ]]; then
  mkdir -p "$RUN/tests"
  python -m pytest tests/test_e2_time_gradient.py tests/test_e2_texture_diagnostics.py tests/test_e2_compat.py tests/test_e2_geometry_fft.py -q | tee "$RUN/tests/prepare.txt"
  python -m tools.e2_time_gradient prepare
elif [[ "$1" == pilot ]]; then
  python -m tools.e2_time_gradient pilot-trace
  python -m tools.e2_time_gradient pilot-gradient
elif [[ "$1" == finish ]]; then
  python -m tools.e2_time_gradient aggregate
  python -m tools.e2_time_gradient bundle
else
  python -m tools.e2_time_gradient "$@"
fi
