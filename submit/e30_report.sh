#!/bin/bash
#SBATCH -J E30R
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e30_apacc_20260930/job_report_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e30_apacc_20260930/job_report_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
python -m tools.e30_report --root "$PWD" --out "$PWD/output_eval/e30_apacc_20260930"
