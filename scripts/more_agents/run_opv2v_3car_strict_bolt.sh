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
OUT_DIR="${OUT_ROOT}/opv2v_3car_strict_bolt"

mkdir -p "${LOG_DIR}"
bash "${ROOT}/scripts/more_agents/assemble_opv2v_4modality.sh" "${OUT_ROOT}"
cd "${ROOT}"

V31_ARGS=(
  --epochs 1
  --teacher_conf_thresh 0.3
  --boost_weight 0.1
  --boost_lo 0.1
  --boost_hi 0.3
  --plugin_adain_alpha_init_logit -10
  --src_modalities m2,m3
)

echo "============================================================"
echo "[opv2v-3car-strict][bolt] use_cav=3, strict_n_car=3"
echo "============================================================"
"${PYTHON_BIN}" -m opencood.tools.online_adapt \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUT_DIR}" \
  --comm_range 100 \
  --use_cav 3 \
  --strict_n_car 3 \
  --assignment_path "opencood/modality_assign/opv2v_3car_strict.json" \
  --seed 42 \
  "${V31_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/opv2v_3car_strict_bolt.log"

echo "[opv2v-3car-strict][bolt] done"
