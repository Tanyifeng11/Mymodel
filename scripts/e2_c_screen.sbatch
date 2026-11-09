#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_c_screen_20261109
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == prepare ]]; then
  mkdir -p "$RUN/tests"
  python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_e2_followup.py tests/test_e2_local_texture.py tests/test_e2_structure.py tests/test_e2_a_metrics.py tests/test_e2_region_router.py tests/test_e2_c_metrics.py -q | tee "$RUN/tests/pytest.txt"
  python -m tools.e2_c_screen prepare
elif [[ "$1" == pilot ]]; then
  python -m tools.e2_c_screen regression
  python -m tools.e2_c_screen generate --arm C1
  python -m tools.e2_c_screen generate --arm C2
  python -m tools.e2_c_screen evaluate --phase fixed
  python -m tools.e2_c_screen smoke
elif [[ "$1" == train || "$1" == generate ]]; then
  arms=(C3 C4 C5 C6)
  arm=${2:-${arms[${SLURM_ARRAY_TASK_ID:-0}]}}
  python -m tools.e2_c_screen "$1" --arm "$arm"
elif [[ "$1" == review ]]; then
  python -m tools.e2_c_screen evaluate
  python -m tools.e2_c_screen bundle
else
  python -m tools.e2_c_screen "$@"
fi
