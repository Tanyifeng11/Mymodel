#!/bin/bash
#SBATCH -J E31A
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e31_anchor_correspondence_20261001/run_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e31_anchor_correspondence_20261001/run_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.e31_a_anchor_mining "${E31_ACTION:-all}"
