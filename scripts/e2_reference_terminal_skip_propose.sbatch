#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
set -euo pipefail
prior="$1"
cd "$prior/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools:$PWD/utils"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# 只复用同零点的新支路合格G；两臂各一个独立AdamW提议，无采样或连续训练。
python -m tools.e2_reference_terminal_skip_propose --prior "$prior"
