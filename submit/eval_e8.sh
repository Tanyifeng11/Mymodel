#!/bin/bash
# 已有 E8a/E8b 权重时单独评测：sbatch submit/eval_e8.sh
# 正式测试集：BF_SPLIT=test sbatch submit/eval_e8.sh

#SBATCH -J E8_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e8_eval_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e8_eval_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export BF_SPLIT="${BF_SPLIT:-validation}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

echo "[E8] 单独评测，划分=${BF_SPLIT}，每组=${NUM_SAMPLES} 张"
DEVICE=cuda:0 bash scripts/eval_phase1_text_guided_resampler.sh
