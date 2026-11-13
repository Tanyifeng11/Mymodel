#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/sd_bfm_nogam_20261114
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="/share/home/u2515283058/Mymodel/output_eval/sd_bfm_fresh_20261113/dependencies:$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == train ]]; then
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.sd_bfm_nogam train
    python -m tools.sd_bfm_nogam generate
elif [[ "$1" == evaluate ]]; then
    python -m tools.sd_bfm_nogam evaluate
    python -m tools.sd_bfm_nogam package
else
    python -m tools.sd_bfm_nogam "$@"
fi
