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

MODEL_DIR="${OUT_ROOT}/opv2v_base_free_pp_eff_second_r50_4mod"

cd "${ROOT}"

"${PYTHON_BIN}" -m opencood.tools.assemble_heter_checkpoint_multi \
  --output_dir "${MODEL_DIR}" \
  --name "opv2v_base_free_pp_eff_second_r50_4mod" \
  --ego_modality m1 \
  --label_type lidar \
  --max_cav 5 \
  --assignment_path "opencood/modality_assign/opv2v_4modality_in_order.json" \
  --m1_dir "${OUT_ROOT}/DirectHeter_OPV2V_ego_lidar_single" \
  --m1_src_modality m1 \
  --m2_dir "${OUT_ROOT}/DirectHeter_OPV2V_general_camera_single" \
  --m2_src_modality m2 \
  --m3_dir "${OUT_ROOT}/DirectHeter_OPV2V_general_lidar_second_single" \
  --m3_src_modality m1 \
  --m4_dir "${OUT_ROOT}/DirectHeter_OPV2V_general_camera_single_r50_20260306_004341" \
  --m4_src_modality m2

echo "[assemble-4mod] model_dir=${MODEL_DIR}"
