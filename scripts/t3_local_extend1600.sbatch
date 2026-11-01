#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_local_texture_pilot_20261101/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export T3_RUN_NAME=t3_local_texture_extend1600_20261101
export T3_PARENT_NAME=t3_local_texture_continue_20261101
export T3_INITIAL_UPDATE=800 T3_FINAL_UPDATE=1600 T3_RESUME_OPTIMIZER=1
export T3_JOB_PREFIX=t3local_s1600
python -m tools.t3_local_continue "$@"
