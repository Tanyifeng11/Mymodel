#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
set -euo pipefail
run="$1"
action="$2"
cd "$run/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
if [ "$action" = gradient ]; then
    python -m tools.e2_state_b_pair_finetune gradient --out "$run" --shard "${3:-${SLURM_ARRAY_TASK_ID:-0}}"
elif [ "$action" = evaluate ]; then
    python -m tools.e2_state_b_pair_evaluation --out "$run" --arm "$3" --split "$4"
elif [ "$action" = finish ]; then
    python -m tools.e2_state_b_pair_summary --out "$run"
else
    python -m tools.e2_state_b_pair_finetune "$action" --out "$run"
fi
