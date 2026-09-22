#!/bin/bash
#SBATCH -J E14_dropout_gen
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_dropout_gen_%j.log
#SBATCH -e log_e14_dropout_gen_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd "${PROJECT_ROOT}"
CMD=(python -m tools.e14_dropout_generation run
 --baseline "${BASELINE:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin}"
 --ab-root "${AB_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_dropout_ab/113304}"
 --inputs "${INPUTS_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_orientation_inputs}"
 --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" --clip-model "${PROJECT_ROOT}/models/clip"
 --output "${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_dropout_generation/${SLURM_JOB_ID:-local}}")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
