#!/bin/bash

#SBATCH -J texture_plain
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_plain_texture_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_plain_texture_%j.err

# 固定草图、简易文本 a cloth 和随机种子；只更换纹理图，分别测试 E5 与 E7a。
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_outputs/plain_prompt_texture_control}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E7A_CKPT="${E7A_CKPT:-${PROJECT_ROOT}/output/phase1_e7a/joint_model.pt}"
SKETCH_PATH="${SKETCH_PATH:-${PROJECT_ROOT}/test/sketch1.jpg}"
DEVICE="${DEVICE:-cuda:0}"
FIXED_SEED="${FIXED_SEED:-42}"
MIN_PAIRWISE_MSE="${MIN_PAIRWISE_MSE:-0.002}"

# 可在 sbatch 时通过 TEXTURE_PATHS 覆盖，例如：
# TEXTURE_PATHS="/path/a.jpg /path/b.jpg /path/c.jpg" sbatch submit/...
TEXTURE_PATHS="${TEXTURE_PATHS:-${PROJECT_ROOT}/test/texture1.jpg ${PROJECT_ROOT}/test/texture2.jpg ${PROJECT_ROOT}/test/texture3.png ${PROJECT_ROOT}/test/texture4.png}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for path in "${TEXTURE_CKPT}" "${E5_CKPT}" "${E7A_CKPT}" "${SKETCH_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] required path does not exist: ${path}" >&2
    exit 1
  fi
done

read -r -a texture_args <<< "${TEXTURE_PATHS}"
if [[ ${#texture_args[@]} -lt 2 ]]; then
  echo "[ERROR] TEXTURE_PATHS must contain at least two texture images" >&2
  exit 1
fi
for path in "${texture_args[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] texture image does not exist: ${path}" >&2
    exit 1
  fi
done

run_test() {
  local name="$1"
  local checkpoint="$2"
  local use_aa_tcr="$3"

  echo "========== ${name} =========="
  python tools/test_plain_prompt_texture_control.py \
    --GAM_model_ckpt "${checkpoint}" \
    --texture_ckpt "${TEXTURE_CKPT}" \
    --sketch_path "${SKETCH_PATH}" \
    --texture_paths "${texture_args[@]}" \
    --output_dir "${OUTPUT_ROOT}/${name}" \
    --prompt "a cloth" \
    --device "${DEVICE}" \
    --fixed_seed "${FIXED_SEED}" \
    --min_pairwise_mse "${MIN_PAIRWISE_MSE}" \
    --texture_condition_mode token \
    --texture_preprocess_mode plain_resize \
    --use_tcpm_lite 1 \
    --use_texture_gate 1 \
    --layer_group_enabled 1 \
    --use_aa_tcr_fuse "${use_aa_tcr}"
}

status=0
run_test e5 "${E5_CKPT}" 0 || status=1
run_test e7a "${E7A_CKPT}" 1 || status=1

echo "========== done =========="
echo "E5 result:  ${OUTPUT_ROOT}/e5/result.json"
echo "E7a result: ${OUTPUT_ROOT}/e7a/result.json"
exit "${status}"
