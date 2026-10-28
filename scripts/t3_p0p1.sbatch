#!/bin/bash
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.t3_p0p1 "$@"
