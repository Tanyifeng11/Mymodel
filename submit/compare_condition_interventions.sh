#!/bin/bash
# 六张固定样本、五组对照：sbatch submit/compare_condition_interventions.sh
#SBATCH -J E5_intervene
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_condition_intervention_%j.log
#SBATCH -e log_condition_intervention_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-/share/home/u2515283058/datasets/BF}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/eval_outputs/condition_response_50steps_6/112581}"
DATASET_JSON="${SOURCE_RUN}/sample_000002/dataset.json"
SPLIT_PATH="${SOURCE_RUN}/sample_000002/fixed_split.json"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/condition_interventions_6/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
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
    [[ -e "${path}" ]] || { echo "缺少路径：${path}" >&2; exit 1; }
  done
  [[ ! -e "${EVAL_ROOT}" ]] || { echo "请使用新的 EVAL_ROOT：${EVAL_ROOT}" >&2; exit 1; }
fi
run mkdir -p "${EVAL_ROOT}"
run cp "${DATASET_JSON}" "${EVAL_ROOT}/dataset.json"
run cp "${SPLIT_PATH}" "${EVAL_ROOT}/fixed_split.json"
common=(
  --dataset_json "${EVAL_ROOT}/dataset.json" --data_root "${DATA_ROOT_PATH}"
  --split_path "${EVAL_ROOT}/fixed_split.json" --num_samples 1 --seed 42 --generation_seed 42
  --gam_ckpt "${E5_CKPT}" --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}"
  --device cuda:0 --modes token --texture_preprocess_mode plain_resize
  --use_tcpm_lite 1 --use_texture_gate 1 --layer_group_enabled 1 --use_palette_tokens 0
  --use_aa_tcr_fuse 0 --use_text_guided_resampler 0 --use_local_detail_adapter 0 --disable_nexus_adapter
  --mask_policy sketch_only --evaluation_protocol original_image_size --compute_fid 0 --compute_kid 0
  --write_text_sidecars 1 --resume_generation 0 --skip_existing 0 --overwrite 0
)
for sid in 2 5 14 18 22 23; do
  sample_name=$(printf 'sample_%06d' "${sid}")
  for variant in baseline boundary weaken_texture strengthen_sketch global; do
    echo "[干预对照] ${sample_name} ${variant}"
    run python tools/run_fixed_benchmark.py "${common[@]}" --run_name e5 \
      --sample_id_start "${sid}" --sample_id_end "$((sid + 1))" \
      --condition_intervention "${variant}" --condition_intervention_source_root "${SOURCE_RUN}" \
      --output_dir "${EVAL_ROOT}/${sample_name}/${variant}"
  done
done
run python tools/report_condition_interventions.py --run-dir "${EVAL_ROOT}" --source-run "${SOURCE_RUN}"
echo "结果：${EVAL_ROOT}/report"
