#!/bin/bash
#SBATCH -J E17_A_GAM
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e17/e17_a_gam_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e17/e17_a_gam_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
FLAT="${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin"
GAM="${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt"
CHECKPOINT="${PROJECT_ROOT}/e17/a_gam_bf_checkpoint.bin"
python -m tools.e17_gam_bf_checkpoint --flat "${FLAT}" --gam "${GAM}" --output "${CHECKPOINT}"
python -m tools.e17_probe \
  --manifest "${PROJECT_ROOT}/data/processed/bf_full_audit_v1/validation_clean.json" \
  --data-root "${VAL_DATA_ROOT:-/share/home/u2515283058/datasets/BF}" \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --clip-model "${PROJECT_ROOT}/models/clip" \
  --labels "${PROJECT_ROOT}/eval_outputs/e14_candidates/candidates/labels.csv" \
  --output "${PROJECT_ROOT}/e17/a_gam" --device cuda:0
python -m tools.e17_source_probe --probe-dir "${PROJECT_ROOT}/e17/a_gam" --device cuda:0
