#!/bin/bash
#SBATCH -J E18_1_direction
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e18_1/e18_1_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e18_1/e18_1_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUT="${ROOT}/e18_1"
cd "${ROOT}"
mkdir -p "${OUT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
CLIP="${ROOT}/models/clip"
START="${ROOT}/e17/direct_train_select/checkpoint-final/joint_model.pt"
CLEAN="${OUT}/clean"
python -m tools.e18_1_clean_patterns --output "${CLEAN}"
python -m tools.e18_1_probe --data "${CLEAN}" --checkpoint "${START}" \
  --clip-model "${CLIP}" --output "${OUT}/probe_start" --device cuda:0
for ARM in b0 b1; do
  python -m tools.e18_joint_alignment --arm "${ARM}" \
    --manifest "${CLEAN}/train.json" --data-root "${CLEAN}" \
    --start-checkpoint "${START}" --clip-model "${CLIP}" \
    --selection "${CLEAN}/train_selection.json" --output "${OUT}/${ARM}" \
    --steps 500 --device cuda:0
  python -m tools.e18_1_probe --data "${CLEAN}" \
    --checkpoint "${OUT}/${ARM}/joint_model.pt" --clip-model "${CLIP}" \
    --output "${OUT}/probe_${ARM}" --device cuda:0
done
