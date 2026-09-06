#!/bin/bash
#SBATCH -J e5_e7a_response
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o log_response_%j.log
#SBATCH -e log_response_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E7A_CKPT="${E7A_CKPT:-${PROJECT_ROOT}/output/phase1_e7a/checkpoint-final/joint_model.pt}"
# 使用训练配套的 BF 适配器；不要用其他训练版本替代。
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
SKETCH_PATH="${SKETCH_PATH:-${PROJECT_ROOT}/test/sketch1.jpg}"
TEXTURE_PATHS="${TEXTURE_PATHS:-${PROJECT_ROOT}/test/texture1.jpg ${PROJECT_ROOT}/test/texture2.jpg ${PROJECT_ROOT}/test/texture3.png ${PROJECT_ROOT}/test/texture4.png}"
SEEDS="${SEEDS:-42}"
STEPS="${STEPS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_outputs/response_diagnosis_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

read -r -a textures <<< "${TEXTURE_PATHS}"
read -r -a seeds <<< "${SEEDS}"
for path in "${E5_CKPT}" "${E7A_CKPT}" "${TEXTURE_CKPT}" "${SKETCH_PATH}" "${textures[@]}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] missing file: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "[ERROR] output already exists: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

extra=()
# 默认由 checkpoint metadata 解析预训练模型路径；迁移服务器时可显式覆盖。
[[ -z "${BASE_MODEL_PATH:-}" ]] || extra+=(--base_model_path "${BASE_MODEL_PATH}")
[[ -z "${VAE_MODEL_PATH:-}" ]] || extra+=(--vae_model_path "${VAE_MODEL_PATH}")
[[ -z "${IMAGE_ENCODER_PATH:-}" ]] || extra+=(--image_encoder_path "${IMAGE_ENCODER_PATH}")

run_model() {
  local name="$1" ckpt="$2" fuse="$3"
  python tools/diagnose_text_texture_response.py \
    --GAM_model_ckpt "${ckpt}" --texture_ckpt "${TEXTURE_CKPT}" \
    --sketch_path "${SKETCH_PATH}" --texture_paths "${textures[@]}" \
    --output_dir "${OUTPUT_ROOT}/${name}" --device cuda:0 \
    --seeds "${seeds[@]}" --num_inference_steps "${STEPS}" \
    --prompt "a cloth" --text_prompts "a red cloth" "a blue cloth" \
    --texture_condition_mode token --texture_mode patch_resampled \
    --texture_preprocess_mode plain_resize --use_tcpm_lite 1 \
    --use_texture_gate 1 --layer_group_enabled 1 --use_aa_tcr_fuse "${fuse}" \
    "${extra[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${name}.log"
}

status=0
run_model e5 "${E5_CKPT}" 0 || status=1
run_model e7a "${E7A_CKPT}" 1 || status=1
echo "[done] results: ${OUTPUT_ROOT} ; status=${status}"
exit "${status}"
