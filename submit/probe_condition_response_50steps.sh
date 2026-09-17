#!/bin/bash
# 提交：sbatch submit/probe_condition_response_50steps.sh
# 固定六张机制诊断样本；保留 112508 中的 sample_id 和 seed=42+sample_id。
#SBATCH -J E5_response_50
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_condition_response_50_%j.log
#SBATCH -e log_condition_response_50_%j.err

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/eval_outputs/condition_response_validation_32/112508}"
export DATASET_JSON="${SOURCE_RUN}/dataset.json"
export SPLIT_PATH="${SOURCE_RUN}/fixed_split.json"
export GENERATION_SEED=42
export NUM_SAMPLES=1
export PROBE_STEPS="$(seq -s ' ' 0 49)"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/eval_outputs/condition_response_50steps_6/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
cd "${PROJECT_ROOT}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ ! -e "${RESULT_ROOT}" ]] || { echo "请使用新的 RESULT_ROOT：${RESULT_ROOT}" >&2; exit 1; }
  mkdir -p "${RESULT_ROOT}"
fi
# 18/23：边界反向较强；5/22：全图出现反向；2/14：五个原观测点边界同向。
# 这是有意选择的诊断集，不能用来估计总体冲突发生率。
for sid in 2 5 14 18 22 23; do
  sample_name=$(printf 'sample_%06d' "${sid}")
  export SAMPLE_ID_START="${sid}"
  export EVAL_ROOT="${RESULT_ROOT}/${sample_name}"
  echo "[50步探针] sample_id=${sid}, generation_seed=$((42 + sid))"
  bash submit/probe_condition_response.sh
done
echo "完成：${RESULT_ROOT}/sample_*/report/response_timeline.json"
