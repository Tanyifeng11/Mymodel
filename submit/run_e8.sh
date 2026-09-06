#!/bin/bash
# 提交：sbatch submit/run_e8.sh
# 同一作业内依次训练 E8a、E8b，再评测 E5/E8a/E8b；默认各训练 1000 步、各评测 100 张。

#SBATCH -J E8
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e8_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e8_%j.err

# Conda 激活脚本可能访问未定义变量，因此激活后再启用 nounset。
set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
export BF_SPLIT="${BF_SPLIT:-validation}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export NUM_GPUS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

# 两组分别从同一 E5 初始化；输出目录分开，训练数据始终使用 training。
for mode in visual text; do
  if [[ "${mode}" == "visual" ]]; then stage=E8a; else stage=E8b; fi
  echo "[${stage}] 开始训练，模式=${mode}，步数=${MAX_TRAIN_STEPS:-1000}"
  RESAMPLER_MODE="${mode}" \
  OUTPUT_DIR="${OUTPUT_BASE}/phase1_resampler_${mode}" \
  DATA_ROOT_PATH="${DATASETS_ROOT}/BF/training" \
  TGR_RESUME_CKPT="" START_GLOBAL_STEP=0 \
  bash scripts/train_phase1_text_guided_resampler.sh
done

# set -e 保证任一训练失败时停止，只有两组均成功才进入评测。
echo "[E8] 开始评测 E5/E8a/E8b，划分=${BF_SPLIT}，每组=${NUM_SAMPLES} 张"
E5_CKPT="${BASE_CKPT}" \
VISUAL_CKPT="${OUTPUT_BASE}/phase1_resampler_visual/checkpoint-final/joint_model.pt" \
TEXT_CKPT="${OUTPUT_BASE}/phase1_resampler_text/checkpoint-final/joint_model.pt" \
TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" \
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" \
EXPERIMENTS=e5,resampler_visual,resampler_text DEVICE=cuda:0 \
bash scripts/eval_phase1_text_guided_resampler.sh

echo "[E8] 流程结束（DRY_RUN=${DRY_RUN:-0}）"
