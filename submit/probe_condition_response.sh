#!/bin/bash
# 提交：sbatch submit/probe_condition_response.sh
# 预览：DRY_RUN=1 bash submit/probe_condition_response.sh
# 冒烟：NUM_SAMPLES=2 sbatch submit/probe_condition_response.sh
#SBATCH -J E5_response_probe
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_condition_response_%j.log
#SBATCH -e log_condition_response_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/processed/bf_full_audit_v1/validation_clean.json}"
# 复用 E12 已固定的 validation 样本顺序，只读取前 N 张。
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_e12_200.json}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
GENERATION_SEED="${GENERATION_SEED:-42}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/condition_response_validation_${NUM_SAMPLES}/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${DATA_ROOT_PATH}" "${DATASET_JSON}" "${SPLIT_PATH}" "${E5_CKPT}" "${TEXTURE_CKPT}" "${CLIP_MODEL}"; do
    [[ -e "${path}" ]] || { echo "缺少必需路径：${path}" >&2; exit 1; }
  done
  [[ ! -e "${EVAL_ROOT}" ]] || { echo "请使用新的 EVAL_ROOT：${EVAL_ROOT}" >&2; exit 1; }
fi
# 复制清单，避免 benchmark 的规范化写回修改已有实验清单；不重采样、不改 caption。
run mkdir -p "${EVAL_ROOT}"
run cp "${DATASET_JSON}" "${EVAL_ROOT}/dataset.json"
run cp "${SPLIT_PATH}" "${EVAL_ROOT}/fixed_split.json"
common=(
  --dataset_json "${EVAL_ROOT}/dataset.json" --data_root "${DATA_ROOT_PATH}"
  --split_path "${EVAL_ROOT}/fixed_split.json" --num_samples "${NUM_SAMPLES}"
  --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}" --seed 42 --generation_seed "${GENERATION_SEED}"
  --gam_ckpt "${E5_CKPT}" --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}"
  --device cuda:0 --modes token --texture_preprocess_mode plain_resize
  --use_tcpm_lite 1 --use_texture_gate 1 --layer_group_enabled 1 --use_palette_tokens 0
  --use_aa_tcr_fuse 0 --use_text_guided_resampler 0 --use_local_detail_adapter 0 --disable_nexus_adapter
  --mask_policy sketch_only --evaluation_protocol original_image_size --compute_fid 0 --compute_kid 0
  --write_text_sidecars 1 --resume_generation 0 --skip_existing 0 --overwrite 0
)
echo '[1/3] E5 关闭探针对照'
run python tools/run_fixed_benchmark.py "${common[@]}" --run_name e5 \
  --condition_response_probe 0 --output_dir "${EVAL_ROOT}/off"
echo '[2/3] 同样本、同 seed，开启局部响应探针'
run python tools/run_fixed_benchmark.py "${common[@]}" --run_name e5 \
  --condition_response_probe 1 --condition_response_probe_steps 0 5 15 25 49 \
  --condition_response_probe_fractions 0.1 0.2 --condition_response_probe_region_kernel 9 \
  --output_dir "${EVAL_ROOT}/on"
echo '[3/3] 检查原尺寸输出逐像素一致性、探针完整性并汇总'
run python tools/report_condition_response_probe.py --baseline-dir "${EVAL_ROOT}/off/e5" \
  --probe-dir "${EVAL_ROOT}/on/e5" --expected-count "${NUM_SAMPLES}" --output-dir "${EVAL_ROOT}/report"
echo "结果：${EVAL_ROOT}/report；验证通过不代表投影方法有效。"
