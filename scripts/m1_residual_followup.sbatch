#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/m1_residual_followup_20261112
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run() { python -m tools.m1_residual_followup "$@"; }
case "$1" in
  prepare)
    mkdir -p "$RUN/tests"
    python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_m1_patch_retrieval.py -q | tee "$RUN/tests/prepare.txt"
    run prepare
    ;;
  pilot)
    run pilot
    run smoke
    ;;
  review)
    run audit-training
    run generate
    run evaluate
    ;;
  *) run "$@" ;;
esac
