#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
set -euo pipefail
run="$1"
cd "$run/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
python -m tools.e2_state_b_last_step generate --out "$run" \
  --source-run /share/home/u2515283058/Mymodel/output_eval/e2_state_b_residue_search_20261115 \
  --baseline-run /share/home/u2515283058/Mymodel/output_eval/e2_state_b_pastel_followup_20261115/trust_retry \
  --checkpoint /share/home/u2515283058/Mymodel/output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --shards "${2:-1}"
