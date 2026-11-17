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
run="$prior/autonomous_repair_20261116/reference_terminal_global_amplitude"
source_run="$prior/autonomous_repair_20261116/reference_terminal_one_step"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
# 每个新阶段都会再次校验source完整96回放、v2资格及原文件SHA。
python -m tools.e2_reference_terminal_global_amplitude prepare --prior "$prior" --out "$run" --source-run "$source_run"
archive_failure() {
    python -m tools.e2_reference_terminal_global_amplitude review --prior "$prior" --out "$run" --source-run "$source_run" \
        || echo '失败证据归档未完成，保留原退出码' >&2
}
for index in 0 1 2 3; do
    status=0
    python -m tools.e2_reference_terminal_global_amplitude gradient --prior "$prior" --out "$run" --source-run "$source_run" \
        --shard "$index" --device cuda:0 > "$run/gradient$index.log" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then archive_failure; exit "$status"; fi
done
for action in propose evaluate; do
    status=0
    python -m tools.e2_reference_terminal_global_amplitude "$action" --prior "$prior" --out "$run" --source-run "$source_run" \
        --device cuda:0 > "$run/$action.log" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then archive_failure; exit "$status"; fi
done
python -m tools.e2_reference_terminal_global_amplitude review --prior "$prior" --out "$run" --source-run "$source_run"
python -m tools.e2_reference_terminal_global_amplitude verify --prior "$prior" --out "$run" --source-run "$source_run"
