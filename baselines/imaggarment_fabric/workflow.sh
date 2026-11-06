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
N=$(python -c 'import torch; print(torch.cuda.device_count())')
python -m unittest baselines.imaggarment_fabric.test_contract -v
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run cache --short
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run train --stage color --short
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run train --stage gam --short
python -m baselines.imaggarment_fabric.run smoke
# set -e：任何接口/短训/保存加载检查失败，均不会启动全量训练。
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run cache
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run train --stage color
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run train --stage gam
python -m torch.distributed.run --standalone --nproc_per_node="$N" -m baselines.imaggarment_fabric.run generate
python -m baselines.imaggarment_fabric.run evaluate
python -m baselines.imaggarment_fabric.evaluate
