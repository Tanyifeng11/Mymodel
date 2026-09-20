#!/bin/bash
# sbatch submit/probe_e14_generation.sh
# DRY_RUN=1 bash submit/probe_e14_generation.sh
#SBATCH -J E14_response
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_response_%j.log
#SBATCH -e log_e14_response_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_generation/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
cmd=(python -m tools.e14_generation_response run
  --split "${SPLIT_PATH:-${PROJECT_ROOT}/eval_outputs/full_condition_probe_6/112711/fixed_split.json}"
  --data-root "${DATA_ROOT_PATH:-/share/home/u2515283058/datasets/BF}"
  --checkpoint "${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
  --texture-checkpoint "${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
  --clip-model "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
  --output "${EVAL_ROOT}")
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
