#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
set -euo pipefail
prior="$1"
run="$prior/autonomous_repair_20261116/reference_terminal_r16_diagnostic"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
# 仅在G1独立结果审阅完成后由主流程提交；run内部首/末各完整来源验证。
# 0新S0、0 optimizer、0候选；失败仍归档已完成证据并保持非零退出。
python -m tools.e2_reference_terminal_r16_diagnostic run --prior "$prior" --out "$run" --device cuda:0
