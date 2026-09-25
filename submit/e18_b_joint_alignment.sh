#!/bin/bash
#SBATCH -J E18_B_joint
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e18/e18_b_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e18/e18_b_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUT="${ROOT}/e18"
cd "${ROOT}"
mkdir -p "${OUT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
GAM="${ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt"
DIRECT="${ROOT}/e17/direct_train_select/checkpoint-final/joint_model.pt"
FLAT="${ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin"
CLIP="${ROOT}/models/clip"
TRAIN="${ROOT}/data/train_bf_texture.json"
TRAIN_ROOT="${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
VAL="${ROOT}/data/processed/bf_full_audit_v1/validation_clean.json"
VAL_ROOT="${VAL_DATA_ROOT:-/share/home/u2515283058/datasets/BF}"
BASE="${ROOT}/models/stable-diffusion-v1-5"
GOLD="${OUT}/gold"
python -m tools.e18_pattern_gold \
  --candidates "${ROOT}/eval_outputs/e14_candidates/candidates" --output "${GOLD}"
python -m tools.e18_gold_probe --gold "${GOLD}" --checkpoint "${GAM}" \
  --clip-model "${CLIP}" --output "${OUT}/gold_probe_gam" --device cuda:0
python -m tools.e18_gold_probe --gold "${GOLD}" --checkpoint "${DIRECT}" \
  --clip-model "${CLIP}" --output "${OUT}/gold_probe_e17_select" --device cuda:0
for ARM in b0 b1; do
  python -m tools.e18_joint_alignment --arm "${ARM}" --manifest "${TRAIN}" \
    --data-root "${TRAIN_ROOT}" --start-checkpoint "${DIRECT}" --clip-model "${CLIP}" \
    --selection "${OUT}/train_selection.json" --output "${OUT}/${ARM}" \
    --steps "${E18_B_STEPS:-500}" --device cuda:0
  CKPT="${OUT}/${ARM}/joint_model.pt"
  python -m tools.e18_gold_probe --gold "${GOLD}" --checkpoint "${CKPT}" \
    --clip-model "${CLIP}" --output "${OUT}/gold_probe_${ARM}" --device cuda:0
  python -m tools.e17_gam_bf_checkpoint --flat "${FLAT}" --gam "${CKPT}" \
    --output "${OUT}/${ARM}/probe_flat.bin"
  python -m tools.e17_probe --manifest "${VAL}" --data-root "${VAL_ROOT}" \
    --checkpoint "${OUT}/${ARM}/probe_flat.bin" --base-model "${BASE}" \
    --clip-model "${CLIP}" --labels "${ROOT}/eval_outputs/e14_candidates/candidates/labels.csv" \
    --output "${OUT}/validation_${ARM}" --device cuda:0
  python -m tools.e17_final_token_probe --fused-dir "${OUT}/validation_${ARM}" \
    --checkpoint "${CKPT}" --output "${OUT}/final_probe_${ARM}" --device cuda:0
  python -m tools.e17_direct_response --manifest "${TRAIN}" --data-root "${TRAIN_ROOT}" \
    --base-checkpoint "${FLAT}" --direct-checkpoint "${CKPT}" \
    --base-model "${BASE}" --clip-model "${CLIP}" --count 32 \
    --output "${OUT}/response_${ARM}" --device cuda:0
done
