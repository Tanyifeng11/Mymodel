#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_a_minimal_20261109
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == prepare ]]; then
  python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_e2_followup.py tests/test_e2_local_texture.py tests/test_e2_structure.py tests/test_e2_a_metrics.py -q
  python -m tools.e2_a_minimal prepare
elif [[ "$1" == pilot ]]; then
  python -m tools.e2_a_minimal regression
  python -m tools.e2_a_minimal smoke
elif [[ "$1" == train ]]; then
  python -m tools.e2_a_minimal train
elif [[ "$1" == finish ]]; then
  python -m tools.e2_a_minimal generate
  python -m tools.e2_a_minimal evaluate
  python -m tools.e2_a_minimal bundle
else
  python -m tools.e2_a_minimal "$@"
fi
