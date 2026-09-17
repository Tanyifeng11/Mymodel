#!/bin/bash
# 三张固定样本 × 四组；以 112600 的实际边界修正逐步 RMS 为固定预算。
#SBATCH -J E5_matched
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_condition_matched_%j.log
#SBATCH -e log_condition_matched_%j.err
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/eval_outputs/condition_response_50steps_6/112581}"
export BUDGET_ROOT="${BUDGET_ROOT:-${PROJECT_ROOT}/eval_outputs/condition_interventions_6/112600}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/condition_interventions_matched_3/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export SAMPLE_IDS='5 18 23'
export VARIANTS='baseline boundary weaken_texture_matched strengthen_sketch_matched'
cd "${PROJECT_ROOT}"
bash submit/compare_condition_interventions.sh
