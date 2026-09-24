#!/bin/bash
#SBATCH -J E16_probe
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o log_e16_probe_%j.log
#SBATCH -e log_e16_probe_%j.err
# 分层线性探针：同一套冻结表示探针跑多个检查点，层集含 cnn1-4，只做 color/orientation 两个任务。
# 目的：（1）定位方向信息在哪一级消失；（2）对比 A0/A1 是否把纹样信息送过 resampler。
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
E16_ROOT="${E16_ROOT:-${PROJECT_ROOT}/e16}"
ARMS="${ARMS:-baseline:${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin a0:${PROJECT_ROOT}/output/e16_a0_tokens16/checkpoint-final/pytorch_model.bin a1:${PROJECT_ROOT}/output/e16_a1_tokens64/checkpoint-final/pytorch_model.bin}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
cd "${PROJECT_ROOT}"
run() { printf '%q ' "$@"; printf '\n'; if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi; }
run mkdir -p "${E16_ROOT}"
for PAIR in ${ARMS}; do
 TAG="${PAIR%%:*}"; CKPT="${PAIR#*:}"
 [[ -f "${CKPT}" ]] || { echo "[SKIP] 缺少检查点 ${CKPT}"; continue; }
 run python -m tools.e15_linear_probe \
  --manifest "${VAL_JSON:-${PROJECT_ROOT}/data/processed/bf_full_audit_v1/validation_clean.json}" \
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF}" \
  --checkpoint "${CKPT}" \
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --clip-model "${PROJECT_ROOT}/models/clip" \
  --labels "${LABELS:-${PROJECT_ROOT}/eval_outputs/e14_candidates/candidates/labels.csv}" \
  --count "${COUNT:-256}" --orientation-count "${ORIENTATION_COUNT:-120}" \
  --components "${COMPONENTS:-64}" --permutations "${PERMUTATIONS:-5}" --seed "${SEED:-42}" \
  --tasks color,orientation \
  --device cuda:0 \
  --output "${E16_ROOT}/probe_${TAG}"
done
echo "完成：${E16_ROOT}/probe_*  （逐标签 SUMMARY.md）"
