#!/bin/bash
#SBATCH -J E32report
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e32_target_supervised_pattern_field_20261001/report_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e32_target_supervised_pattern_field_20261001/report_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
review_args=()
if [[ -n "${E32_REVIEW_JSON:-}" ]]; then review_args=(--review-json "$E32_REVIEW_JSON"); fi
python -m tools.e32_report "${review_args[@]}"
