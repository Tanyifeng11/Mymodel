#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_texture_bottleneck_20261111
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == prepare ]]; then
  mkdir -p "$RUN/tests"
  python -m pytest tests/test_e2_texture_diagnostics.py tests/test_e2_compat.py tests/test_e2_geometry_fft.py -q | tee "$RUN/tests/prepare.txt"
  python -m tools.e2_texture_bottleneck prepare
elif [[ "$1" == diagnose ]]; then
  python -m tools.e2_texture_bottleneck diagnose --shard "${SLURM_ARRAY_TASK_ID}" --shards 2
elif [[ "$1" == review ]]; then
  python -m tools.e2_texture_bottleneck evaluate
  python -m tools.e2_texture_bottleneck bundle
else
  python -m tools.e2_texture_bottleneck "$@"
fi
