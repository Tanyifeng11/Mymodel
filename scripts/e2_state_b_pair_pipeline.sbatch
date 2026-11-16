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
    SLURM_CPUS_PER_TASK=1 python -m tools.e2_state_b_pair_finetune gradient --out "$run" \
        --shard "$1" --device "cuda:$2" > "$run/gradient_$1.log" 2> "$run/gradient_$1.err"
}
evaluate() {
    SLURM_CPUS_PER_TASK=1 python -m tools.e2_state_b_pair_evaluation --out "$run" \
        --arm "$1" --split "$2" --device "cuda:$3" > "$run/eval_$1_$2.log" 2> "$run/eval_$1_$2.err"
}
# 同一Slurm作业内部并行，遵守账号提交数量上限；任何失败停止后续阶段。
gradient 0 0 & p0=$!
gradient 1 1 & p1=$!
gradient 2 2 & p2=$!
wait "$p0"
gradient 3 0 & p3=$!
wait "$p1"
wait "$p2"
wait "$p3"
echo PAIR_ALL_GRADIENTS_COMPLETE
python -m tools.e2_state_b_pair_finetune propose --out "$run" > "$run/propose.log" 2> "$run/propose.err"
evaluate f68 replay 0
evaluate f68 holdout 0 & e0=$!
evaluate C0 replay 1 & e1=$!
evaluate C1 replay 2 & e2=$!
wait "$e1"
evaluate C0 holdout 1 & e3=$!
wait "$e2"
evaluate C1 holdout 2 & e4=$!
wait "$e0"
wait "$e3"
wait "$e4"
python -m tools.e2_state_b_pair_summary --out "$run"
echo PAIR_PIPELINE_COMPLETE
