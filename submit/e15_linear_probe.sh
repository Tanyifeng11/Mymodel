#!/bin/bash
#SBATCH -J E15_probe
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o log_e15_probe_%j.log
#SBATCH -e log_e15_probe_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
# E15 线性探针：冻结权重、不训练，检验 16 个 texture token 还能解码哪些参考图信息。
# 默认落盘到 e15/linear_probe，与 D1-D5 同属 e15 主文件夹。
E15_ROOT="${E15_ROOT:-${PROJECT_ROOT}/e15}"
CMD=(python -m tools.e15_linear_probe
  --manifest "${VAL_JSON:-${PROJECT_ROOT}/data/processed/bf_full_audit_v1/validation_clean.json}"
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF}"
  --checkpoint "${CHECKPOINT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
  --clip-model "${PROJECT_ROOT}/models/clip"
  --labels "${LABELS:-${PROJECT_ROOT}/eval_outputs/e14_candidates/candidates/labels.csv}"
  --count "${COUNT:-256}"
  --orientation-count "${ORIENTATION_COUNT:-120}"
  --components "${COMPONENTS:-64}"
  --permutations "${PERMUTATIONS:-5}"
  --seed "${SEED:-42}"
  --device "${DEVICE:-cuda:0}"
  --output "${E15_ROOT}/linear_probe")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi