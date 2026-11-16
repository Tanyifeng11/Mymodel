#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
set -euo pipefail
prior="$1"
run="$2"
mkdir -p "$run"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
git rev-parse HEAD > "$run/execution_commit.txt"
retry() { python -m tools.e2_state_b_pair_retry "$1" --prior "$prior" --out "$run"; }
if [ ! -f "$run/run_provenance.json" ]; then retry prepare; fi
retry verify
if [ ! -f "$run/training/proposals.json" ]; then
    SLURM_CPUS_PER_TASK=2 python -m tools.e2_state_b_pair_finetune propose --out "$run" \
        > "$run/propose.log" 2> "$run/propose.err"
fi
retry rounded
evaluate() {
    if [ -f "$run/evaluation/$1/$2/result.json" ]; then return; fi
    SLURM_CPUS_PER_TASK=1 python -m tools.e2_state_b_pair_evaluation --out "$run" \
        --arm "$1" --split "$2" --device "cuda:$3" > "$run/eval_$1_$2.log" 2> "$run/eval_$1_$2.err"
}
# 只评测新的四个候选任务；f68的92图已逐SHA复制，不重新反传。
models=(C0 C1 C0 C1)
splits=(replay replay holdout holdout)
gpus=${SLURM_GPUS_ON_NODE:-1}
pids=()
for ((gpu=0; gpu<gpus; gpu++)); do
    (for ((index=gpu; index<4; index+=gpus)); do
        evaluate "${models[$index]}" "${splits[$index]}" "$gpu"
    done) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
retry verify
python -m tools.e2_state_b_pair_summary --out "$run"
echo PAIR_FP64_RETRY_COMPLETE
