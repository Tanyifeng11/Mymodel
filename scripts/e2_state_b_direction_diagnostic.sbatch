#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
set -euo pipefail
prior="$1"
run="$2"
fixed="$prior/projection_fp64_retry"
mkdir -p "$run"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
git rev-parse HEAD > "$run/execution_commit.txt"
diagnose() { python -m tools.e2_state_b_direction_diagnostic "$1" --fixed "$fixed" --out "$run"; }
if [ ! -f "$run/diagnostic_protocol.json" ]; then diagnose prepare; fi
diagnose verify
gpus=${SLURM_GPUS_ON_NODE:-1}
pids=()
for ((gpu=0; gpu<gpus; gpu++)); do
    (for ((index=gpu; index<4; index+=gpus)); do
        if [ ! -f "$run/case_$index.json" ]; then
            SLURM_CPUS_PER_TASK=1 python -m tools.e2_state_b_direction_diagnostic case \
                --fixed "$fixed" --out "$run" --case-index "$index" --device "cuda:$gpu" \
                > "$run/case_$index.log" 2> "$run/case_$index.err"
        fi
    done) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
diagnose verify
python -m tools.e2_state_b_direction_summary --out "$run"
echo B_DIRECTION_DIAGNOSTIC_COMPLETE
