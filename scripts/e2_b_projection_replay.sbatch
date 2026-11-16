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
run="$prior/autonomous_repair_20261116/scaled_projection_retry"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
# CPU 重试已生成候选；这里仅校验和评测，不能再次生成参数更新。
python -m tools.e2_b_scaled_projection_retry verify --prior "$prior"
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
python -m tools.e2_b_scaled_replay_review --out "$run" --prior "$prior"
python -m tools.e2_b_scaled_projection_retry verify --prior "$prior"
