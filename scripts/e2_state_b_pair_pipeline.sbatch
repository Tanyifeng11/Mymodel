#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
set -euo pipefail
run="$1"
cd "$run/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
git rev-parse HEAD > "$run/execution_commit.txt"
gradient() {
    if [ -f "$run/gradients/shard$1.pt" ]; then return; fi
    SLURM_CPUS_PER_TASK=1 python -m tools.e2_state_b_pair_finetune gradient --out "$run" \
        --shard "$1" --device "cuda:$2" > "$run/gradient_$1.log" 2> "$run/gradient_$1.err"
}
evaluate() {
    if [ -f "$run/evaluation/$1/$2/result.json" ]; then return; fi
    SLURM_CPUS_PER_TASK=1 python -m tools.e2_state_b_pair_evaluation --out "$run" \
        --arm "$1" --split "$2" --device "cuda:$3" > "$run/eval_$1_$2.log" 2> "$run/eval_$1_$2.err"
}
# 同一作业内按实际GPU数分配独立工作；并行度不改变数据和更新预算。
gpus=${SLURM_GPUS_ON_NODE:-1}
pids=()
for ((gpu=0; gpu<gpus; gpu++)); do
    (for ((shard=gpu; shard<4; shard+=gpus)); do gradient "$shard" "$gpu"; done) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo PAIR_ALL_GRADIENTS_COMPLETE
if [ ! -f "$run/training/proposals.json" ]; then
    python -m tools.e2_state_b_pair_finetune propose --out "$run" > "$run/propose.log" 2> "$run/propose.err"
fi
evaluate f68 replay 0
models=(f68 C0 C1 C0 C1)
splits=(holdout replay replay holdout holdout)
pids=()
for ((gpu=0; gpu<gpus; gpu++)); do
    (for ((index=gpu; index<5; index+=gpus)); do
        evaluate "${models[$index]}" "${splits[$index]}" "$gpu"
    done) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python -m tools.e2_state_b_pair_summary --out "$run"
echo PAIR_PIPELINE_COMPLETE
