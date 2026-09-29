#!/bin/bash
#SBATCH -J E28BS
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e28_20260929/job_bs_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e28_20260929/job_bs_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -m tools.e28_experiment --stage revise --root "$PWD" --out "$PWD/output_eval/e28_20260929"
python -m tools.e28_experiment --stage scaffold --root "$PWD" --e27 "$PWD/output_eval/e27_20260929" --out "$PWD/output_eval/e28_20260929"
