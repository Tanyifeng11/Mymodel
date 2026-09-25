#!/bin/bash
#SBATCH -J E17_C_control
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e17/e17_c_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e17/e17_c_%j.err
# Only submit when A has no reliable direction/frequency signal and B also fails.
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
E17_ROOT="${E17_ROOT:-${PROJECT_ROOT}/e17}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
BASE_BF="${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin"
DIRECT="${E17_ROOT}/direct_train_select/checkpoint-final/joint_model.pt"
C_ROOT="${E17_ROOT}/c_encoder_positive_select"
TEXTURE="${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin"
python -m tools.e17_encoder_control \
  --manifest "${PROJECT_ROOT}/data/train_bf_texture.json" \
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}" \
  --direct-checkpoint "${DIRECT}" --base-checkpoint "${BASE_BF}" \
  --clip-model "${PROJECT_ROOT}/models/clip" \
  --output "${C_ROOT}" --device cuda:0
python -m tools.e17_probe \
  --manifest "${PROJECT_ROOT}/data/processed/bf_full_audit_v1/validation_clean.json" \
  --data-root "${VAL_DATA_ROOT:-/share/home/u2515283058/datasets/BF}" \
  --checkpoint "${C_ROOT}/pytorch_model.bin" \
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --clip-model "${PROJECT_ROOT}/models/clip" \
  --labels "${PROJECT_ROOT}/eval_outputs/e14_candidates/candidates/labels.csv" \
  --output "${E17_ROOT}/c_probe_select" --device cuda:0
python -m tools.e17_final_token_probe \
  --fused-dir "${E17_ROOT}/c_probe_select" \
  --checkpoint "${C_ROOT}/joint_model.pt" \
  --output "${E17_ROOT}/c_final_token_probe_select" --device cuda:0
python -m tools.e17_direct_response \
  --manifest "${PROJECT_ROOT}/data/train_bf_texture.json" \
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}" \
  --base-checkpoint "${BASE_BF}" \
  --direct-checkpoint "${C_ROOT}/joint_model.pt" \
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --clip-model "${PROJECT_ROOT}/models/clip" \
  --output "${E17_ROOT}/c_response_select" --device cuda:0
python -m tools.e15_d5_generation \
  --manifest "${PROJECT_ROOT}/data/train_bf_texture.json" \
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}" \
  --checkpoint "${C_ROOT}/joint_model.pt" \
  --texture-ckpt "${TEXTURE}" \
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --clip-model "${PROJECT_ROOT}/models/clip" \
  --mask-root "${PROJECT_ROOT}/eval_outputs/e14_causal_inputs/regions" \
  --count 32 --seed 42 --steps 50 --device cuda:0 --suite reference \
  --output "${E17_ROOT}/unused_layers_c_select" \
  --output-reference "${E17_ROOT}/generation_c_select"
python -m tools.e17_generation_audit \
  --reports "${E17_ROOT}/generation_c_select/report.json" \
  --mask-root "${PROJECT_ROOT}/eval_outputs/e14_causal_inputs/regions" \
  --output "${E17_ROOT}/c_generation_audit_select.json"
