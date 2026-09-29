#!/bin/bash
#SBATCH -J E27C
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e27_20260929/job_c_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e27_20260929/job_c_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.e27_experiment --root "$PWD" --out "$PWD/output_eval/e27_20260929" --stage C
