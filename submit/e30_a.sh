#!/bin/bash
#SBATCH -J E30A
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e30_apacc_20260930/job_a_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e30_apacc_20260930/job_a_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m tools.e30_a_feature_feasibility --root "$PWD" --out "$PWD/output_eval/e30_apacc_20260930"
