#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu04
#SBATCH --time=00:30:00
set -euo pipefail
prior="$1"
run="$prior/autonomous_repair_20261116/terminal_head_projection_retry"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
python -m tools.e2_terminal_head_projection_retry prepare --prior "$prior"
status=0
python -m tools.e2_terminal_head_projection_retry diagnose --prior "$prior" > "$run/diagnose.log" 2>&1 || status=$?
if [ "$status" -ne 0 ]; then
    if [ -f "$run/solver_diagnostic.json" ]; then
        python -m tools.e2_terminal_head_projection_retry review --prior "$prior" || echo '失败归档未完成，保留诊断退出码' >&2
    fi
    exit "$status"
fi
python -m tools.e2_terminal_head_projection_retry propose --prior "$prior" > "$run/propose.log" 2>&1
python -m tools.e2_terminal_head_projection_retry verify --prior "$prior"
# 这里只生成CPU候选；GPU replay由独立脚本及资源申请完成。
