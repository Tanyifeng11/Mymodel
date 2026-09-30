#!/bin/bash
#SBATCH -J E30A2Report
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e30_a2_affinity_20260930/report_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e30_a2_affinity_20260930/report_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
python -m tools.e30_a2_report --out "$PWD/output_eval/e30_a2_affinity_20260930"
