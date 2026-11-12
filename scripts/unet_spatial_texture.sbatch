#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/unet_spatial_texture_20261112
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$1" in
    train)
        python -m tools.unet_spatial_texture preflight
        torchrun --standalone --nproc_per_node="${2:-1}" -m tools.unet_spatial_texture train
        ;;
    generate)
        python -m tools.unet_spatial_texture generate --shard "${SLURM_ARRAY_TASK_ID:-0}"
        ;;
    *)
        python -m tools.unet_spatial_texture "$@"
        ;;
esac
