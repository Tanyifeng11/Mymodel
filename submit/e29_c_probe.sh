#!/bin/bash
#SBATCH -J E29C
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e29_20260930/job_c_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e29_20260930/job_c_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m tools.e29_repair --root "$PWD" --out "$PWD/output_eval/e29_20260930"
