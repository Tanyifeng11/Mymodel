#!/bin/bash
#SBATCH -J E19_2_A2
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e19_2_a2/run_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e19_2_a2/run_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python -m tools.e19_2_a2 --root "${ROOT}" --device cuda:0
