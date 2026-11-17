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
run="$prior/autonomous_repair_20261116/terminal_head_one_step"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
python -m tools.e2_terminal_head_one_step prepare --prior "$prior"
gpus="$(python - <<'PY'
import torch
print(min(2, torch.cuda.device_count()))
PY
)"
gradient() {
    python -m tools.e2_terminal_head_one_step gradient --prior "$prior" --shard "$1" \
        --device "cuda:$2" > "$run/gradient$1.log" 2>&1
}
evaluate() {
    python -m tools.e2_terminal_head_one_step evaluate --prior "$prior" --arm "$1" \
        --device "cuda:$2" > "$run/evaluate$1.log" 2>&1
}
join_workers() {
    status=0
    wait "$1" || status=1
    wait "$2" || status=1
    if [ "$status" -ne 0 ]; then exit "$status"; fi
}
if [ "$gpus" -eq 1 ]; then
    for index in 0 1 2 3; do gradient "$index" 0; done
elif [ "$gpus" -eq 2 ]; then
    (for index in 0 2; do gradient "$index" 0; done) &
    p0=$!
    (for index in 1 3; do gradient "$index" 1; done) &
    p1=$!
    join_workers "$p0" "$p1"
else
    echo '需要至少一张GPU' >&2
    exit 2
fi
# 两臂均零更新只归档失败证据，保留propose的非零退出；其他错误直接停止。
proposal_status=0
python -m tools.e2_terminal_head_one_step propose --prior "$prior" > "$run/propose.log" 2>&1 || proposal_status=$?
if [ "$proposal_status" -ne 0 ]; then
    if [ -f "$run/training/proposals.json" ] && python -c \
        'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["stop_before_replay"] else 1)' \
        "$run/training/proposals.json"; then
        python -m tools.e2_terminal_head_one_step review --prior "$prior" || echo '失败证据归档未完成，保留原提案退出码' >&2
    fi
    exit "$proposal_status"
fi
if [ "$gpus" -eq 1 ]; then
    evaluate C0 0
    evaluate C1 0
else
    evaluate C0 0 &
    p0=$!
    evaluate C1 1 &
    p1=$!
    join_workers "$p0" "$p1"
fi
python -m tools.e2_terminal_head_one_step review --prior "$prior"
python -m tools.e2_terminal_head_one_step verify --prior "$prior"
