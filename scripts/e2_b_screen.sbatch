#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == prepare ]]; then
  python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_e2_followup.py tests/test_e2_local_texture.py -q
  python -m tools.e2_b_screen prepare
  python -m tools.e2_b_screen audit-training
elif [[ "$1" == pilot ]]; then
  python -m tools.e2_b_screen freeze-pairs
  python -m tools.e2_fft_experiment regression --out "$RUN"
  python -m tools.e2_fft_experiment generate-regression --out "$RUN"
  python -m tools.e2_b_screen calibrate
  python -m tools.e2_b_screen smoke
elif [[ "$1" == train ]]; then
  python -m tools.e2_b_screen train --arm "$2"
elif [[ "$1" == train-array ]]; then
  ARMS=(B0 B1 B2 B4 B5)
  python -m tools.e2_b_screen train --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
elif [[ "$1" == generate ]]; then
  python -m tools.e2_b_screen generate --arm "$2"
elif [[ "$1" == generate-array ]]; then
  ARMS=(B0 B1 B2 B4 B5)
  python -m tools.e2_b_screen generate --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
elif [[ "$1" == audit-final ]]; then
  python -m tools.e2_b_screen audit-training-final
elif [[ "$1" == evaluate ]]; then
  python -m tools.e2_b_screen evaluate
  python -m tools.e2_b_screen bundle
else
  python -m tools.e2_b_screen "$@"
fi
