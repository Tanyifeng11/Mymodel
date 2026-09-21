#!/bin/bash
# sbatch submit/probe_e14_real_rotation.sh
# 提交前检查输入：CHECK_ONLY=1 bash submit/probe_e14_real_rotation.sh
#SBATCH -J E14_real_rotation
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_rotation_%j.log
#SBATCH -e log_e14_rotation_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" && "${CHECK_ONLY:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
INPUTS_ROOT="${INPUTS_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_rotation_inputs}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for file in inputs.json ref_0013_original.png ref_0013_rot90.png ref_0030_original.png ref_0030_rot90.png; do
    if [[ ! -f "${INPUTS_ROOT}/${file}" ]]; then
      echo "缺少输入文件：${INPUTS_ROOT}/${file}" >&2
      echo '请上传整个 e14_real_rotation_inputs 目录，或设置 INPUTS_ROOT 为包含 inputs.json 的目录。' >&2
      exit 1
    fi
  done
  echo "输入文件齐全：${INPUTS_ROOT}"
fi
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then exit 0; fi
cmd=(python -m tools.e14_real_rotation run
  --inputs "${INPUTS_ROOT}"
  --output "${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_rotation/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
  --data-root "${DATA_ROOT_PATH:-/share/home/u2515283058/datasets/BF}"
  --checkpoint "${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
  --texture-checkpoint "${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
  --clip-model "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}")
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
