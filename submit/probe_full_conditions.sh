#!/bin/bash
# �ύ��sbatch submit/probe_full_conditions.sh
# Ԥ����DRY_RUN=1 bash submit/probe_full_conditions.sh
# ð�̣�NUM_SAMPLES=2 sbatch submit/probe_full_conditions.sh
#SBATCH -J E5_full_probe
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_full_condition_%j.log
#SBATCH -e log_full_condition_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF}"
LOCAL_RESPONSE_ROOT="${LOCAL_RESPONSE_ROOT:-${PROJECT_ROOT}/eval_outputs/condition_response_50steps_6/112581}"
DATASET_JSON="${DATASET_JSON:-${LOCAL_RESPONSE_ROOT}/sample_000002/dataset.json}"
# ���� E12 �ѹ̶��� validation ����˳��ֻ��ȡǰ N �š�
SPLIT_PATH="${SPLIT_PATH:-${LOCAL_RESPONSE_ROOT}/sample_000002/fixed_split.json}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
SAMPLE_ID_START="${SAMPLE_ID_START:-0}"
SAMPLE_ID_END=$((SAMPLE_ID_START + NUM_SAMPLES))
read -r -a probe_steps <<< "${PROBE_STEPS:-0 5 15 25 49}"
GENERATION_SEED="${GENERATION_SEED:-42}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/full_condition_probe_6/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
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
    [[ -e "${path}" ]] || { echo "ȱ�ٱ���·����${path}" >&2; exit 1; }
  done
  [[ ! -e "${EVAL_ROOT}" ]] || { echo "��ʹ���µ� EVAL_ROOT��${EVAL_ROOT}" >&2; exit 1; }
fi
# �����嵥������ benchmark �Ĺ淶��д���޸�����ʵ���嵥�����ز��������� caption��
run mkdir -p "${EVAL_ROOT}"
run cp "${DATASET_JSON}" "${EVAL_ROOT}/dataset.json"
run cp "${SPLIT_PATH}" "${EVAL_ROOT}/fixed_split.json"
common=(
  --dataset_json "${EVAL_ROOT}/dataset.json" --data_root "${DATA_ROOT_PATH}"
  --split_path "${EVAL_ROOT}/fixed_split.json" --num_samples 1
   --seed 42 --generation_seed "${GENERATION_SEED}"
  --gam_ckpt "${E5_CKPT}" --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}"
  --device cuda:0 --modes token --texture_preprocess_mode plain_resize
  --use_tcpm_lite 1 --use_texture_gate 1 --layer_group_enabled 1 --use_palette_tokens 0
  --use_aa_tcr_fuse 0 --use_text_guided_resampler 0 --use_local_detail_adapter 0 --disable_nexus_adapter
  --mask_policy sketch_only --evaluation_protocol original_image_size --compute_fid 0 --compute_kid 0
  --write_text_sidecars 1 --resume_generation 0 --skip_existing 0 --overwrite 0
)
for sid in 2 5 14 18 22 23; do
  sample_name=$(printf 'sample_%06d' "${sid}")
  for branch in off on; do
    enabled=0
    [[ "${branch}" == "on" ]] && enabled=1
    run python tools/run_fixed_benchmark.py "${common[@]}" --run_name e5 \
      --sample_id_start "${sid}" --sample_id_end "$((sid + 1))" \
      --full_condition_probe "${enabled}" --output_dir "${EVAL_ROOT}/${sample_name}/${branch}"
  done
done
run python tools/report_full_conditions.py --run-dir "${EVAL_ROOT}" --local-response-root "${LOCAL_RESPONSE_ROOT}"
