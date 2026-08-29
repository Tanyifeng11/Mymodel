#!/bin/bash

# CTD Stage A 训练前验证：不加载 checkpoint，不启动训练，不占用 GPU。
# 执行：等价性/算子测试，以及小规模 S2/S3 评测集渲染。
# 提交：sbatch submit/verify_ctd_stage_a.sh
# 覆盖示例：CTD_PREP_LIMIT=64 sbatch submit/verify_ctd_stage_a.sh

#SBATCH -J ctd_verify
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o /share/home/u2515283058/Mymodel/log_ctd_verify_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_ctd_verify_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATA_ROOT="${CTD_DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
DATASET_JSON="${CTD_DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SD_PATH="${CTD_SD_PATH:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
# 仅供第 2 步构造 CTD 的验证样本清单；不用于 E7a 模型评测。
# 该路径来自 caption sweep 的既有 plan.json 记录。
PER_SAMPLE_CSV="${CTD_PER_SAMPLE_CSV:-${PROJECT_ROOT}/eval_outputs/phase1_e7a_500/report_e7a/metrics_per_sample.csv}"
CTD_SEED="${CTD_SEED:-42}"
CTD_PREP_LIMIT="${CTD_PREP_LIMIT:-64}"
CTD_MIN_DELTA_E="${CTD_MIN_DELTA_E:-15.0}"
VALIDATION_ROOT="${CTD_VALIDATION_ROOT:-${PROJECT_ROOT}/output_eval/ctd_validation_${SLURM_JOB_ID}}"

for path in "${PROJECT_ROOT}" "${DATA_ROOT}" "${DATASET_JSON}" "${SD_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] required path missing: ${path}" >&2
    exit 2
  fi
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CTD_DATA_ROOT="${DATA_ROOT}"
export CTD_DATASET_JSON="${DATASET_JSON}"
export CTD_SD_PATH="${SD_PATH}"
export PYTHONDONTWRITEBYTECODE=1

echo "========== [1/2] CTD 等价性与算子测试 =========="
cd "${PROJECT_ROOT}"
TEST_STATUS=0
if python -c "import pytest"; then
  python -m pytest -v test_ctd_equivalence.py || TEST_STATUS=$?
else
  echo "[ERROR] 当前 Python 环境未安装 pytest；等价性测试未执行。" >&2
  TEST_STATUS=127
fi

echo "========== [2/2] CTD 评测集小规模渲染 =========="
# prepare_ctd_eval_sets.py 当前会写相对 data/；切到独立目录可避免覆盖
# 项目 data/ 下的文件。图像、JSON、manifest 全部落在本次验证目录。
if [[ ! -e "${PER_SAMPLE_CSV}" ]]; then
  echo "[ERROR] CTD 验证样本清单缺失: ${PER_SAMPLE_CSV}" >&2
  echo "        用 CTD_PER_SAMPLE_CSV=/实际/metrics_per_sample.csv 覆盖后重提即可。" >&2
  exit 2
fi
mkdir -p "${VALIDATION_ROOT}"
cd "${VALIDATION_ROOT}"
python "${PROJECT_ROOT}/tools/prepare_ctd_eval_sets.py" \
  --per_sample_csv "${PER_SAMPLE_CSV}" \
  --data_root "${DATA_ROOT}" \
  --out_dir "${VALIDATION_ROOT}/eval_sets" \
  --ctd_seed "${CTD_SEED}" \
  --min_delta_e "${CTD_MIN_DELTA_E}" \
  --limit "${CTD_PREP_LIMIT}"

echo "[done] validation artifacts: ${VALIDATION_ROOT}"
if [[ "${TEST_STATUS}" -ne 0 ]]; then
  if [[ "${TEST_STATUS}" -eq 127 ]]; then
    echo "[failed] 等价性测试未执行（缺少 pytest），评测集渲染结果仅用于诊断；禁止启动训练。" >&2
  else
    echo "[failed] 等价性测试未通过，评测集渲染结果仅用于诊断；禁止启动训练。" >&2
  fi
  exit "${TEST_STATUS}"
fi
