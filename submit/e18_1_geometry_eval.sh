#!/bin/bash
#SBATCH -J E18_1_geometry
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e18_1/geometry_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e18_1/geometry_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUT="${ROOT}/e18_1"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
for ARM in start b0 b1; do
  if [ "${ARM}" = start ]; then
    CHECKPOINT="${ROOT}/e17/direct_train_select/checkpoint-final/joint_model.pt"
  else
    CHECKPOINT="${OUT}/${ARM}/joint_model.pt"
  fi
  python -m tools.e18_1_probe --data "${OUT}/clean" --checkpoint "${CHECKPOINT}" \
    --clip-model "${ROOT}/models/clip" --output "${OUT}/probe_${ARM}" --device cuda:0
done
