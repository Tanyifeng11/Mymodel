#!/bin/bash

# caption 颜色响应实验 —— CTD 的立项闸门。
#
# 固定 sketch / 纹理图 / seed / 采样器, 只把 caption 里的颜色词扫过 12 档,
# 看生成图 mask 内主色跟不跟。判定结果决定下一阶段主线:
#
#   PINNED                     -> 输出被参考图钉住, 这是真缺陷, CTD 按规格开工
#   FOLLOWS                    -> 模型本来就服从文本颜色, 主线转向纹理侧余量
#   FOLLOWS_BUT_PATTERN_DRIFTS -> 颜色与图案纠缠, 直接进 Stage B 频带分离
#   PARTIAL                    -> 扩大样本量再判
#
# 背景见 docs/ctd_stage_a_spec.md §0.5 与 docs/e0_e7a_settlement.md §0.2:
# E5 的 TxtCF 已优于 GT floor, 所以"文本颜色被忽略"这个缺陷用现有指标测不出来。
# 但那只证明指标无法发现缺陷, 不证明模型会响应 caption 的颜色变化。
#
# 可在 sbatch 时用环境变量覆盖:
#   SWEEP_STAGES           默认 "preflight plan verify generate score"
#   NUM_SAMPLES            默认 48    (总生成量 = NUM_SAMPLES x 颜色档数)
#   SWEEP_COLORS           默认 12 档
#   GENERATION_SEED        默认 42
#   MASK_POLICY            默认 sketch_only
#   REQUIRE_MASK_BACKEND   默认 opencv (设 any 可在 Pillow fallback 下继续)
#   E5_CKPT / TEXTURE_CKPT / OUT_DIR / PER_SAMPLE_CSV / DEVICE
#
# 只重新评分(无需 GPU, 也可直接在登录节点跑):
#   SWEEP_STAGES="score" sbatch submit/sweep_caption_color.sh

#SBATCH -J color_sweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_color_sweep_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_color_sweep_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT="${DATA_ROOT:-${DATASETS_ROOT}/BF/training}"

TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
PER_SAMPLE_CSV="${PER_SAMPLE_CSV:-${PROJECT_ROOT}/eval_outputs/report_e7a/metrics_per_sample.csv}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/output_eval/caption_color_sweep/e5}"

SWEEP_STAGES="${SWEEP_STAGES:-preflight plan verify generate score}"
NUM_SAMPLES="${NUM_SAMPLES:-48}"
SWEEP_COLORS="${SWEEP_COLORS:-black,white,gray,red,blue,green,yellow,orange,purple,pink,brown,beige}"
GENERATION_SEED="${GENERATION_SEED:-42}"
MASK_POLICY="${MASK_POLICY:-sketch_only}"
REQUIRE_MASK_BACKEND="${REQUIRE_MASK_BACKEND:-opencv}"
RUN_NAME="${RUN_NAME:-e5_tcpm_lite}"
DEVICE="${DEVICE:-cuda:0}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

COMMON=(
  --per_sample_csv "${PER_SAMPLE_CSV}"
  --data_root "${DATA_ROOT}"
  --out_dir "${OUT_DIR}"
  --colors "${SWEEP_COLORS}"
  --num_samples "${NUM_SAMPLES}"
  --generation_seed "${GENERATION_SEED}"
  --mask_policy "${MASK_POLICY}"
  --gam_ckpt "${E5_CKPT}"
  --texture_ckpt "${TEXTURE_CKPT}"
  --run_name "${RUN_NAME}"
  --device "${DEVICE}"
  --require_mask_backend "${REQUIRE_MASK_BACKEND}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON+=(--overwrite)
fi

has_stage() {
  [[ " ${SWEEP_STAGES} " == *" $1 "* ]]
}

echo "============================================================"
echo "caption color response sweep"
echo "PROJECT_ROOT   = ${PROJECT_ROOT}"
echo "DATA_ROOT      = ${DATA_ROOT}"
echo "E5_CKPT        = ${E5_CKPT}"
echo "TEXTURE_CKPT   = ${TEXTURE_CKPT}"
echo "PER_SAMPLE_CSV = ${PER_SAMPLE_CSV}"
echo "OUT_DIR        = ${OUT_DIR}"
echo "STAGES         = ${SWEEP_STAGES}"
echo "NUM_SAMPLES    = ${NUM_SAMPLES}  (x $(tr -cd ',' <<< "${SWEEP_COLORS}" | wc -c | tr -d ' ') + 1 colors)"
echo "MASK_POLICY    = ${MASK_POLICY}   REQUIRE_MASK_BACKEND = ${REQUIRE_MASK_BACKEND}"
echo "============================================================"

# ---------------------------------------------------------------------------
# preflight: mask 后端 + 输入文件。不合格直接退出, 不浪费 GPU 时间。
#
# 这一步存在的理由: garment_mask_utils 在 cv2 缺失时会静默退回 Pillow 形态学,
# 两条路径产出的 mask 不同。既有全部报告都跑在 Pillow 路径上且从未被记录,
# 这是本项目里 mask 派生指标长期不可比的根因。
# ---------------------------------------------------------------------------
if has_stage preflight; then
  echo "########## [1/5] preflight ##########"
  python tools/sweep_caption_color.py --stage preflight --check_ckpt "${COMMON[@]}"
fi

# ---------------------------------------------------------------------------
# plan: 构造 (样本 x 颜色) 矩阵。无 GPU。
# ---------------------------------------------------------------------------
if has_stage plan; then
  echo
  echo "########## [2/5] plan ##########"
  python tools/sweep_caption_color.py --stage plan "${COMMON[@]}"
fi

# ---------------------------------------------------------------------------
# verify: 实验设计不变量。放在 generate 之前, 因为下面这些条件一旦被破坏,
# 生成图照样出得来但结论无效 —— 那就白烧几百次推理:
#   - 同一 base 的 12 个变体必须共用同一 seed(否则颜色效应与噪声混淆)
#   - sketch/texture/target 在 base 内必须完全不变
#   - 各变体 prompt 必须两两不同(替换若静默失败就成了 12 次重复实验)
#   - 传给 inference 的 flag 必须与 run_fixed_benchmark 逐项一致
# ---------------------------------------------------------------------------
if has_stage verify; then
  echo
  echo "########## [3/5] verify(设计不变量 + 后端) ##########"
  python test_mask_backend.py
  python test_caption_color_sweep.py
fi

# ---------------------------------------------------------------------------
# generate: 唯一需要 GPU 的阶段。已存在的图默认跳过, OVERWRITE=1 可强制重生成。
# ---------------------------------------------------------------------------
if has_stage generate; then
  echo
  echo "########## [4/5] generate(需要 GPU) ##########"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python tools/sweep_caption_color.py --stage generate "${COMMON[@]}"
fi

# ---------------------------------------------------------------------------
# score: 判定。无 GPU, 也可以把生成图拷回本地再跑。
# ---------------------------------------------------------------------------
if has_stage score; then
  echo
  echo "########## [5/5] score ##########"
  python tools/sweep_caption_color.py --stage score "${COMMON[@]}"
fi

echo
echo "============================================================"
echo "产出:"
echo "  ${OUT_DIR}/plan.json            (样本 x 颜色 矩阵, 含 seed 与 mask 后端)"
echo "  ${OUT_DIR}/sweep_per_item.csv   (逐 (样本,颜色) 的主色与 ΔE)"
echo "  ${OUT_DIR}/sweep_per_base.csv   (逐样本的 response_range / follow_rate)"
echo "  ${OUT_DIR}/sweep_summary.json   (汇总 + verdict)"
echo "  ${OUT_DIR}/gen/<base>/<color>/  (生成图)"
echo
echo "判定与下一步的对应关系见 docs/ctd_stage_a_spec.md §0.5"
echo "============================================================"
