#!/bin/bash
#SBATCH -J E40_fixed_E5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m tools.e40_runtime --shard "${E40_SHARD:-0}" --shards 2
