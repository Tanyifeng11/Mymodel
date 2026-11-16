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
run="$prior/autonomous_repair_20261116/scaled_direction_bridge"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
# 两张GPU各自顺序处理固定探针；prepare仅一次，不生成新G或优化器。
python -m tools.e2_state_b_scaled_direction_bridge prepare --prior "$prior"
(
    for index in 0 2 4; do
        python -m tools.e2_state_b_scaled_direction_bridge probe --prior "$prior" \
            --probe-index "$index" --device cuda:0 > "$run/probe_$index.log" 2>&1
    done
) &
p0=$!
(
    for index in 1 3; do
        python -m tools.e2_state_b_scaled_direction_bridge probe --prior "$prior" \
            --probe-index "$index" --device cuda:1 > "$run/probe_$index.log" 2>&1
    done
) &
p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
python -m tools.e2_state_b_scaled_direction_bridge summarize --prior "$prior"
python -m tools.e2_state_b_scaled_direction_bridge verify --prior "$prior"
