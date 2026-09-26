#!/bin/bash
#SBATCH -J E20_utilization
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e20/%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e20/%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python -m tools.e20_utilization --root "${ROOT}" --data-root /share/home/u2515283058/datasets/BF --steps 600 --device cuda:0
