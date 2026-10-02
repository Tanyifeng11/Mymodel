#!/bin/bash
#SBATCH -J E33Rprior
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33r_rotation_causality_20261002/prior_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33r_rotation_causality_20261002/prior_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m tools.e33r_train_prior
