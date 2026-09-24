#!/bin/bash
#SBATCH -J E15_diag
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o log_e15_diag_%j.log
#SBATCH -e log_e15_diag_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
# 五参考条件全链路诊断：D1 表示衰减、D2 residual 热图、D3 离线区域、D4 分组消融。
CMD=(python -m tools.e15_diagnosis
 --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
 --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
 --checkpoint "${CHECKPOINT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin}"
 --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
 --clip-model "${PROJECT_ROOT}/models/clip"
 --mask-root "${MASK_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_causal_inputs/regions}"
 --previous-report "${PREVIOUS_REPORT:-${PROJECT_ROOT}/eval_e14/causal_suite/113817/denoising_report.json}"
 --count "${COUNT:-32}"
 --stages "${STAGES:-d1,d2,d3,d4}"
 --device "${DEVICE:-cuda:0}"
 --output "${E15_ROOT:-${PROJECT_ROOT}/e15}")
if [[ "${FORCE_D4:-0}" == 1 ]]; then CMD+=(--force-d4); fi
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi