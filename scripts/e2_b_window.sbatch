#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_b_window_20261111
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run() { python -m tools.e2_b_window "$1" --out "$RUN"; }
case "$1" in
  prepare)
    mkdir -p "$RUN/tests"
    python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_e2_local_texture.py tests/test_e2_time_gradient.py tests/test_e2_b_window.py -q | tee "$RUN/tests/prepare.txt"
    run prepare
    ;;
  pilot)
    run calibrate
    run smoke
    ;;
  finish)
    run generate
    run evaluate
    run bundle
    ;;
  *) run "$1" ;;
esac
