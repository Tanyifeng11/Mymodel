#!/bin/bash
# sbatch submit/diagnose_e9_existing_images.sh
# 仅分析已有 300 张生成图；不加载 checkpoint，不需要 GPU。
#SBATCH -J E9_image_audit
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_image_audit_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_image_audit_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e9_b_schedule_recheck_validation_100/111336}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-${EVAL_ROOT}/image_audit_${SLURM_JOB_ID:-local}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
cmd=(python tools/diagnose_e9_existing_images.py --eval-root "${EVAL_ROOT}"
     --output-dir "${AUDIT_OUTPUT_DIR}" --expected-count 100)
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
