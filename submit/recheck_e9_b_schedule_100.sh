#!/bin/bash
# 提交：sbatch submit/recheck_e9_b_schedule_100.sh
# 仅复核 E9-B 的关闭旁路、完整旁路、仅早期旁路三组；固定验证集前 100 张与 seed 42。

#SBATCH -J E9B_recheck
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_b_recheck_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_b_recheck_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export E9_B_CKPT="${E9_B_CKPT:-${OUTPUT_BASE}/phase1_local_detail_local_stage2_2000/checkpoint-final/joint_model.pt}"
export TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
export DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/validation}"
export DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_validation_resampler.json}"
export SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_resampler_100.json}"
export RECHECK_RUN_ID="${RECHECK_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export RECHECK_OUTPUT_DIR="${RECHECK_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/e9_b_schedule_recheck_validation_100/${RECHECK_RUN_ID}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${E5_CKPT}" "${E9_B_CKPT}" "${TEXTURE_CKPT}" "${DATASET_JSON}" "${SPLIT_PATH}"; do
    [[ -f "${path}" ]] || { echo "缺少必需文件：${path}" >&2; exit 1; }
  done
  python tools/check_e9_frozen.py --e5-ckpt "${E5_CKPT}" --e9-b-ckpt "${E9_B_CKPT}" \
    --output-dir "${RECHECK_OUTPUT_DIR}/frozen_check"
fi

cmd=(
  python tools/run_e9_b_diagnostics.py
  --dataset-json "${DATASET_JSON}" --data-root "${DATA_ROOT_PATH}" --split-path "${SPLIT_PATH}"
  --gam-ckpt "${E9_B_CKPT}" --texture-ckpt "${TEXTURE_CKPT}"
  --clip-model-path "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
  --output-dir "${RECHECK_OUTPUT_DIR}" --num-samples 100 --seed 42 --generation-seed 42
  --variants "${RECHECK_VARIANTS:-alpha_000,alpha_100,window_early}" --device "${DEVICE:-cuda:0}"
  --compute-fid "${COMPUTE_FID:-0}"
)
echo "[E9-B 100 张复核] 输出目录：${RECHECK_OUTPUT_DIR}"
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${cmd[@]}"
  if [[ "${RUN_OUTPUT_BLOCK_AUDIT:-0}" == "1" ]]; then
    python tools/check_e9_output_block.py --run-dir "${RECHECK_OUTPUT_DIR}/output_block/e9_b_diagnosis"
    python tools/diagnose_e9_existing_images.py --eval-root "${RECHECK_OUTPUT_DIR}" \
      --output-dir "${RECHECK_OUTPUT_DIR}/image_audit" --expected-count 100 \
      --variants alpha_000,alpha_100,output_block
  fi
fi
