#!/bin/bash
#SBATCH -J E33report
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_counterfactual_reference_20261002/report_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_counterfactual_reference_20261002/report_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
review_args=()
if [[ -n "${E33_REVIEW_JSON:-}" ]]; then review_args=(--review-json "$E33_REVIEW_JSON"); fi
python -m tools.e33_report "${review_args[@]}"
