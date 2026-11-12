#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/reader_reference_loss_20261112
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$1" in
    calibrate)
        python -m tools.reader_reference_loss preflight
        python -m tools.reader_reference_loss calibrate
        ;;
    train)
        torchrun --standalone --nproc_per_node=2 -m tools.reader_reference_loss train --arm "$2"
        ;;
    *)
        python -m tools.reader_reference_loss "$@"
        ;;
esac
