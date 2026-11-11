#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/m1_s2t_patch_20261111
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run() { python -m tools.m1_experiment "$@"; }
case "$1" in
  prepare)
    mkdir -p "$RUN/tests"
    python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_m1_patch_retrieval.py -q | tee "$RUN/tests/prepare.txt"
    run prepare
    ;;
  pilot)
    run regression
    run generate-regression
    run smoke
    ;;
  train|generate)
    arms=(M0 M1)
    arm=${2:-${arms[${SLURM_ARRAY_TASK_ID:-0}]}}
    run "$1" --arm "$arm"
    ;;
  review)
    run evaluate
    run diagnose
    run bundle
    ;;
  *) run "$@" ;;
esac
