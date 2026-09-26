#!/bin/bash
#SBATCH -J E18_2_fused_only
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e18_2/e18_2_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e18_2/e18_2_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUT="${ROOT}/e18_2"
CLEAN="${ROOT}/e18_1/clean"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python -m tools.e18_2_fused_alignment --manifest "${CLEAN}/train.json" \
  --data-root "${CLEAN}" --selection "${CLEAN}/train_selection.json" \
  --start-checkpoint "${ROOT}/e17/direct_train_select/checkpoint-final/joint_model.pt" \
  --clip-model "${ROOT}/models/clip" --output "${OUT}/b2" --steps 500 --device cuda:0
python -m tools.e18_1_probe --data "${CLEAN}" --checkpoint "${OUT}/b2/joint_model.pt" \
  --clip-model "${ROOT}/models/clip" --output "${OUT}/probe_b2" --device cuda:0
python -m tools.e18_2_compare --controls "${ROOT}/e18_1" --output "${OUT}"
