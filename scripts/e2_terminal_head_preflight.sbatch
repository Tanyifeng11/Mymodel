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
run="$prior/autonomous_repair_20261116/terminal_head_preflight"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
python -m tools.e2_terminal_head_preflight prepare --prior "$prior"
gpus="$(python - <<'PY'
import torch
print(min(2, torch.cuda.device_count()))
PY
)"
cache() {
    python -m tools.e2_terminal_head_preflight cache --prior "$prior" --shard "$1" \
        --device "cuda:$2" > "$run/shard$1.log" 2>&1
}
if [ "$gpus" -eq 1 ]; then
    for index in 0 1 2 3; do cache "$index" 0; done
elif [ "$gpus" -eq 2 ]; then
    (for index in 0 2; do cache "$index" 0; done) &
    p0=$!
    (for index in 1 3; do cache "$index" 1; done) &
    p1=$!
    status=0
    wait "$p0" || status=1
    wait "$p1" || status=1
    if [ "$status" -ne 0 ]; then exit "$status"; fi
else
    echo '需要至少一张GPU' >&2
    exit 2
fi
python -m tools.e2_terminal_head_preflight review --prior "$prior"
python -m tools.e2_terminal_head_preflight verify --prior "$prior"
