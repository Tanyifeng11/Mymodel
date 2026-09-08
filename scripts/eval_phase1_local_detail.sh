#!/usr/bin/env bash
set -euo pipefail

# E9 首轮三组对照：E5 / E9-A(旁路读原 16 token) / E9-B(旁路读压缩前局部 token)。
# 同一官方验证划分、同一样本、同一生成 seed，三组之间只有旁路来源不同。
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_SPLIT="${BF_SPLIT:-validation}"
case "${BF_SPLIT}" in validation|test) ;; *) echo "BF_SPLIT 必须为 validation 或 test" >&2; exit 1 ;; esac
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/${BF_SPLIT}}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_${BF_SPLIT}_resampler.json}"
FULL_SPLIT_PATH="${FULL_SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_resampler_full.json}"
# EVAL_SCOPE=metrics 用完整随机子集看整体指标；pattern 用图案子集做定性对照。
EVAL_SCOPE="${EVAL_SCOPE:-metrics}"
case "${EVAL_SCOPE}" in metrics|pattern) ;; *) echo "EVAL_SCOPE 必须为 metrics 或 pattern" >&2; exit 1 ;; esac
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E9_A_CKPT="${E9_A_CKPT:-${OUTPUT_BASE}/phase1_local_detail_resampled/checkpoint-final/joint_model.pt}"
E9_B_CKPT="${E9_B_CKPT:-${OUTPUT_BASE}/phase1_local_detail_local/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
# 默认每次评测单独落盘，避免 checkpoint 更新后误复用旧生成图。
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_${EVAL_SCOPE}_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
EXPERIMENTS="${EXPERIMENTS:-e5,e9_a,e9_b}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
has_experiment() {
  [[ ",${EXPERIMENTS}," == *",$1,"* ]]
}

# 冻结检查：确认两组都只新增旁路，且除来源外训练配置一致。
# 用 if 而不是 [[ ]] && ...，避免 set -e 下条件不成立时直接退出。
frozen_args=()
if has_experiment e9_a; then
  frozen_args+=(--e9-a-ckpt "${E9_A_CKPT}")
fi
if has_experiment e9_b; then
  frozen_args+=(--e9-b-ckpt "${E9_B_CKPT}")
fi
if [[ "${#frozen_args[@]}" -gt 0 ]]; then
  run python tools/check_e9_frozen.py --e5-ckpt "${E5_CKPT}" "${frozen_args[@]}" \
    --output-dir "${EVAL_ROOT}/frozen_check"
fi

# 完整验证清单：官方划分，split_count=0 保留全部样本，并检查与训练集无重叠。
if [[ ! -f "${DATASET_JSON}" || ! -f "${FULL_SPLIT_PATH}" ]]; then
  run python tools/build_bf_test_manifest.py --data_root "${DATA_ROOT_PATH}" \
    --dataset_json "${DATASET_JSON}" --split_path "${FULL_SPLIT_PATH}" --split_count 0 \
    --seed 42 --train_json "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
fi

if [[ "${EVAL_SCOPE}" == "pattern" ]]; then
  # 图案子集只依据参考图自身统计挑选，与任何一组生成结果无关。
  SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_pattern_${NUM_SAMPLES}.json}"
  run python tools/prepare_e9_pattern_split.py --split_path "${FULL_SPLIT_PATH}" \
    --data_root "${DATA_ROOT_PATH}" --output_split "${SPLIT_PATH}" \
    --summary_json "${EVAL_ROOT}/pattern_split/pattern_summary.json" \
    --swap_json "${EVAL_ROOT}/pattern_split/color_swap_pairs.json" \
    --count "${NUM_SAMPLES}" --swap_count "${SWAP_COUNT:-8}" --seed 42
else
  # 指标组沿用与 E8c 同一个 100 张随机子集，便于跨轮次对照。
  SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_resampler_${NUM_SAMPLES}.json}"
  if [[ ! -f "${SPLIT_PATH}" ]]; then
    run python tools/build_bf_test_manifest.py --data_root "${DATA_ROOT_PATH}" \
      --dataset_json "${DATASET_JSON}" --split_path "${SPLIT_PATH}" --split_count "${NUM_SAMPLES}" \
      --seed 42 --train_json "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
  fi
fi

common=(
  --dataset_json "${DATASET_JSON}" --data_root "${DATA_ROOT_PATH}" --split_path "${SPLIT_PATH}"
  --num_samples "${NUM_SAMPLES}" --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}" --seed 42
  --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}" --device "${DEVICE:-cuda:0}"
  --modes token --texture_preprocess_mode plain_resize --use_tcpm_lite 1 --use_texture_gate 1
  --layer_group_enabled 1 --use_aa_tcr_fuse 0 --mask_policy sketch_only
  --evaluation_protocol original_image_size --compute_kid 1 --kid_subset_size "${NUM_SAMPLES}"
  --kid_subsets 50 --write_text_sidecars 1 --resume_generation 1 --skip_existing 1 --overwrite 0
)
IFS=',' read -r -a seeds <<< "${GENERATION_SEEDS}"
IFS=',' read -r -a experiments <<< "${EXPERIMENTS}"
for seed in "${seeds[@]}"; do
  for experiment in "${experiments[@]}"; do
    # use_local_detail_adapter=-1 表示按 checkpoint 元数据决定；E5 显式关闭。
    case "${experiment}" in
      e5) ckpt="${E5_CKPT}"; local_detail=0 ;;
      e9_a) ckpt="${E9_A_CKPT}"; local_detail=-1 ;;
      e9_b) ckpt="${E9_B_CKPT}"; local_detail=-1 ;;
      *) echo "未知实验：${experiment}" >&2; exit 1 ;;
    esac
    run python tools/run_fixed_benchmark.py "${common[@]}" \
      --generation_seed "${seed}" --gam_ckpt "${ckpt}" --run_name "${experiment}" \
      --use_text_guided_resampler 0 --use_local_detail_adapter "${local_detail}" \
      --output_dir "${EVAL_ROOT}/seed_${seed}"
  done
  run python tools/validate_benchmark_outputs.py --experiments_dir "${EVAL_ROOT}/seed_${seed}" \
    --experiment_names "${EXPERIMENTS}" --expected_count "${NUM_SAMPLES}"
  if has_experiment e5 && { has_experiment e9_a || has_experiment e9_b; }; then
    run python tools/check_e9_images.py --experiments-dir "${EVAL_ROOT}/seed_${seed}" \
      --experiment-names "${EXPERIMENTS}" --expected-count "${NUM_SAMPLES}" \
      --output-dir "${EVAL_ROOT}/seed_${seed}/image_check"
  fi
done
comparisons=""
add_comparison() { comparisons+="${comparisons:+,}$1"; }
if has_experiment e5 && has_experiment e9_a; then add_comparison e9_a:e5; fi
if has_experiment e5 && has_experiment e9_b; then add_comparison e9_b:e5; fi
if has_experiment e9_a && has_experiment e9_b; then add_comparison e9_b:e9_a; fi
if [[ -n "${comparisons}" ]]; then
  report_args=()
  if [[ "${#seeds[@]}" == 1 ]]; then report_args+=(--single_seed_reference); fi
  run python tools/report_e7a_control.py --eval_root "${EVAL_ROOT}" --output_dir "${EVAL_ROOT}/report" \
    --experiments "${EXPERIMENTS}" --generation_seeds "${GENERATION_SEEDS}" \
    --comparisons "${comparisons}" "${report_args[@]}"
fi
echo "[E9] scope=${EVAL_SCOPE}，评测目录：${EVAL_ROOT}"
if [[ "${EVAL_SCOPE}" == "pattern" ]]; then
  echo "[E9] 图案子集需人工逐图核对具体纹样是否保留；TPF-Patch 主要反映分组平均颜色，不作为图案判据。"
fi
