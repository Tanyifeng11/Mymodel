#!/bin/bash
#SBATCH -J E26report
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e26_20260929/job_report_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e26_20260929/job_report_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -m tools.e26_report --root "$PWD" --out "$PWD/output_eval/e26_20260929"
