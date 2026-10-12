#!/bin/bash
#SBATCH -J E44_DRRA
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e44_mimicbrush_drra_20261012/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e44_mimicbrush_drra_20261012/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m "$@"
