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

V31_ARGS=(
  --epochs 1
  --teacher_conf_thresh 0.3
  --boost_weight 0.1
  --boost_lo 0.1
  --boost_hi 0.3
  --plugin_adain_alpha_init_logit -10
  --src_modalities m2,m3,m4
)

for USE_CAV in 2 3 4; do
  OUT_DIR="${OUT_ROOT}/opv2v_4mod_multisrc_bolt_usecav${USE_CAV}"
  echo "============================================================"
  echo "[opv2v-4mod][multi-src-bolt] use_cav=${USE_CAV}"
  echo "============================================================"
  "${PYTHON_BIN}" -m opencood.tools.online_adapt \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${OUT_DIR}" \
    --comm_range 100 \
    --use_cav "${USE_CAV}" \
    --seed 42 \
    "${V31_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/opv2v_4mod_multisrc_bolt_usecav${USE_CAV}.log"
done

echo "[opv2v-4mod][multi-src-bolt] done"
