#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/imaggarment_fabric_gam_20261106
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$RUN/cache/huggingface"
if [[ "$1" == prepare ]]; then
  python -m baselines.imaggarment_fabric.prepare
elif [[ "$1" == check ]]; then
  python -m baselines.imaggarment_fabric.prepare
  python -m unittest baselines.imaggarment_fabric.test_contract baselines.imaggarment_fabric.test_official_processors -v
else
  N="${SLURM_GPUS_ON_NODE:-1}"
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N" -m baselines.imaggarment_fabric.run "$@"
fi
