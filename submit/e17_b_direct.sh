#!/bin/bash
#SBATCH -J E17_B_direct
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e17/e17_b_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e17/e17_b_%j.err
set -eo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
E17_ROOT="${E17_ROOT:-${PROJECT_ROOT}/e17}"
BASE_CKPT="${BASE_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
mkdir -p "${E17_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export BASE_CKPT TEXTURE_ADAPTER_CKPT="${TEXTURE_CKPT}"
export OUTPUT_DIR="${E17_ROOT}/direct_train"
export RESAMPLER_MODE=e17_direct RESAMPLER_LR="${RESAMPLER_LR:-1e-4}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}" CHECKPOINTING_STEPS=0
export START_GLOBAL_STEP=0 TGR_RESUME_CKPT="" REPORT_TO=none
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
bash scripts/train_phase1_text_guided_resampler.sh
DIRECT_CKPT="${OUTPUT_DIR}/checkpoint-final/joint_model.pt"
if [[ "${DRY_RUN:-0}" != 1 ]]; then test -f "${DIRECT_CKPT}"; fi
for TAG in baseline direct; do
  if [[ "${TAG}" == baseline ]]; then CKPT="${BASE_CKPT}"; else CKPT="${DIRECT_CKPT}"; fi
  CMD=(python -m tools.e15_d5_generation
    --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
    --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
    --checkpoint "${CKPT}" --texture-ckpt "${TEXTURE_CKPT}"
    --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
    --clip-model "${PROJECT_ROOT}/models/clip"
    --mask-root "${PROJECT_ROOT}/eval_outputs/e14_causal_inputs/regions"
    --count "${GEN_COUNT:-32}" --seed 42 --steps "${GEN_STEPS:-50}"
    --device cuda:0 --suite reference
    --output "${E17_ROOT}/unused_layers_${TAG}"
    --output-reference "${E17_ROOT}/generation_${TAG}")
  printf '%q ' "${CMD[@]}"; printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
done
CMD=(python -m tools.e17_direct_response
  --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
  --base-checkpoint "${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin"
  --direct-checkpoint "${DIRECT_CKPT}"
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
  --clip-model "${PROJECT_ROOT}/models/clip"
  --count "${GEN_COUNT:-32}" --device cuda:0 --output "${E17_ROOT}/direct_response")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
