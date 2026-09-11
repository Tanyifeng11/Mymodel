#!/bin/bash
# sbatch submit/diagnose_e9_mask_propagation.sh
# 全部固定 100 张：重放 E9-B 全程轨迹，在 0/5/15/25/49 步做同 latent 开关对照。
#SBATCH -J E9_mask_probe
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_mask_probe_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_mask_probe_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E9_B_CKPT="${E9_B_CKPT:-${OUTPUT_BASE}/phase1_local_detail_local_stage2_2000/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/e9_mask_propagation_100/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${E5_CKPT}" "${E9_B_CKPT}" "${TEXTURE_CKPT}" \
    "${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_resampler_100.json}" \
    "${DATASET_JSON:-${PROJECT_ROOT}/data/bf_validation_resampler.json}"; do
    [[ -f "${path}" ]] || { echo "缺少必需文件：${path}" >&2; exit 1; }
  done
fi
run() {
  printf '%q ' "$@"; printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
run python tools/check_e9_frozen.py --e5-ckpt "${E5_CKPT}" --e9-b-ckpt "${E9_B_CKPT}" --output-dir "${PROBE_OUTPUT_DIR}/frozen_check"
run python tools/run_fixed_benchmark.py \
  --dataset_json "${DATASET_JSON:-${PROJECT_ROOT}/data/bf_validation_resampler.json}" \
  --split_path "${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_resampler_100.json}" \
  --data_root "${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/validation}" \
  --gam_ckpt "${E9_B_CKPT}" --texture_ckpt "${TEXTURE_CKPT}" \
  --clip_model_path "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}" \
  --num_samples 100 --sample_id_start 0 --sample_id_end 100 --seed 42 --generation_seed 42 \
  --modes token --texture_preprocess_mode plain_resize --use_tcpm_lite 1 --use_texture_gate 1 \
  --layer_group_enabled 1 --use_aa_tcr_fuse 0 --use_text_guided_resampler 0 --use_local_detail_adapter 1 \
  --mask_policy sketch_only --evaluation_protocol original_image_size --compute_fid 0 --compute_kid 0 \
  --local_detail_scale 1 --local_detail_step_start 0 --local_detail_step_end 49 \
  --save_local_detail_trace 1 --local_detail_propagation_probe 1 --overwrite 1 \
  --output_dir "${PROBE_OUTPUT_DIR}" --run_name e9_b_probe --device cuda:0
run python tools/report_e9_propagation.py --run-dir "${PROBE_OUTPUT_DIR}/e9_b_probe" \
  --output-dir "${PROBE_OUTPUT_DIR}/report" --expected-count 100
