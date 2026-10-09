#!/bin/bash
#SBATCH -J E33GC_G0
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --time=12:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_gc_generation_causality_20261009/job_%A_%a.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_gc_generation_causality_20261009/job_%A_%a.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.e33gc_g0 run --shards 4
