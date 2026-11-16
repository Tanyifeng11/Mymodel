#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
set -euo pipefail
prior="$1"
run="$prior/autonomous_repair_20261116/scaled_retry"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
# verify 核对新 G 的 SHA，未通过预设复现门槛会退出 2，阻断 propose。
python -m tools.e2_state_b_scaled_retry verify --prior "$prior" --out "$run"
python -m tools.e2_state_b_pair_finetune propose --out "$run"
python -m tools.e2_state_b_pair_evaluation --out "$run" --arm C0 --split replay --device cuda:0 > "$run/C0-replay.log" 2>&1 &
p0=$!
python -m tools.e2_state_b_pair_evaluation --out "$run" --arm C1 --split replay --device cuda:1 > "$run/C1-replay.log" 2>&1 &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
# 这里只评估原 replay 门槛；不运行 holdout、不采用权重、不继续训练。
python -m tools.e2_b_scaled_replay_review --out "$run" --prior "$prior"
