#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
set -euo pipefail
prior="$1"
run="$prior/autonomous_repair_20261116/reference_terminal_one_step"
qualified="$prior/autonomous_repair_20261116/reference_terminal_preflight_v2"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
# 文件不存在、passed=false或来源校验失败，均在新实验开始前退出。
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$qualified/qualification.json"
python -m tools.e2_reference_terminal_preflight verify --prior "$prior" --out "$qualified"
python -m tools.e2_reference_terminal_one_step prepare --prior "$prior" --out "$run"
for index in 0 1 2 3; do
    python -m tools.e2_reference_terminal_one_step gradient --prior "$prior" --out "$run" --shard "$index" \
        --device cuda:0 > "$run/gradient$index.log" 2>&1
done
proposal_status=0
python -m tools.e2_reference_terminal_one_step propose --prior "$prior" --out "$run" > "$run/propose.log" 2>&1 || proposal_status=$?
if [ "$proposal_status" -ne 0 ]; then
    if [ -f "$run/training/proposals.json" ] && python -c \
        'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["stop_before_replay"] else 1)' \
        "$run/training/proposals.json"; then
        python -m tools.e2_reference_terminal_one_step review --prior "$prior" --out "$run" || echo '失败证据归档未完成，保留原退出码' >&2
    fi
    exit "$proposal_status"
fi
for arm in C0 C1; do
    python -m tools.e2_reference_terminal_one_step evaluate --prior "$prior" --out "$run" --arm "$arm" \
        --device cuda:0 > "$run/evaluate$arm.log" 2>&1
done
python -m tools.e2_reference_terminal_one_step review --prior "$prior" --out "$run"
python -m tools.e2_reference_terminal_one_step verify --prior "$prior" --out "$run"
