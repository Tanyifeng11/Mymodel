#!/bin/bash
#SBATCH -J E15_d5_gen
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o log_e15_d5_%j.log
#SBATCH -e log_e15_d5_%j.err
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
# D5 最终生成验证：固定 32 样本与 seed，只改 texture attention 层开关。
# 默认落盘到 e15/d5_generation，与 D1-D4 同属 e15 主文件夹。
E15_ROOT="${E15_ROOT:-${PROJECT_ROOT}/e15}"
CMD=(python -m tools.e15_d5_generation
 --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
 --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
 --checkpoint "${CHECKPOINT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
 --texture-ckpt "${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
 --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
 --clip-model "${PROJECT_ROOT}/models/clip"
 --mask-root "${MASK_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_causal_inputs/regions}"
 --count "${COUNT:-32}"
 --seed "${SEED:-42}"
 --steps "${STEPS:-50}"
 --device "${DEVICE:-cuda:0}"
 --output "${E15_ROOT}/d5_generation")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
