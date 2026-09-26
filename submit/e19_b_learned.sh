#!/bin/bash
#SBATCH -J E19_B_pattern
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e19/e19_b_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e19/e19_b_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python -m tools.e19_learned --a-output "${ROOT}/e19/a" --clean "${ROOT}/e18_1/clean" \
  --checkpoint "${ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt" \
  --appearance-checkpoint "${ROOT}/e18_2/b2/joint_model.pt" \
  --base-model "${ROOT}/models/stable-diffusion-v1-5" --clip-model "${ROOT}/models/clip" \
  --manifest "${ROOT}/data/processed/bf_full_audit_v1/validation_clean.json" \
  --data-root /share/home/u2515283058/datasets/BF --output "${ROOT}/e19/b" --device cuda:0
