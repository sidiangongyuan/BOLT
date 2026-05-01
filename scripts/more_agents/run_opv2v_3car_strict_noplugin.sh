#!/bin/bash
set -euo pipefail

ROOT="/mnt/sdb/public/data/yk/projects/Hetero-task"
OUT_ROOT="${1:-/mnt/sdb/public/data/yk/result/hetero-task}"
DEFAULT_PYTHON="/mnt/sdb/public/data/yk/conda_envs/quantv2x/bin/python"
if [ -x "${DEFAULT_PYTHON}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
LOG_DIR="${ROOT}/logs/more_agents"
MODEL_DIR="${OUT_ROOT}/opv2v_base_free_pp_eff_second_r50_4mod"

mkdir -p "${LOG_DIR}"
bash "${ROOT}/scripts/more_agents/assemble_opv2v_4modality.sh" "${OUT_ROOT}"
cd "${ROOT}"

echo "============================================================"
echo "[opv2v-3car-strict][no-plugin] use_cav=3, strict_n_car=3"
echo "============================================================"
"${PYTHON_BIN}" -m opencood.tools.inference \
  --model_dir "${MODEL_DIR}" \
  --fusion_method intermediate \
  --comm_range 100 \
  --use_cav 3 \
  --strict_n_car 3 \
  --assignment_path "opencood/modality_assign/opv2v_3car_strict.json" \
  --seed 42 \
  --note "_3car_strict_noplugin" \
  2>&1 | tee "${LOG_DIR}/opv2v_3car_strict_noplugin.log"

echo "[opv2v-3car-strict][no-plugin] done"
