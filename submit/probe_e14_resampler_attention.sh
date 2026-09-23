#!/bin/bash
#SBATCH -J E14_resampler
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_resampler_%j.log
#SBATCH -e log_e14_resampler_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
CMD=(python -m tools.e14_resampler_attention
 --features "${FEATURES_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_dropout_ab/113304/baseline/representations}"
 --checkpoint "${CHECKPOINT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
 --output "${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_resampler_attention/${SLURM_JOB_ID:-local}}")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
