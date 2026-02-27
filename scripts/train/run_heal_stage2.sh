#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/mnt/sdb/public/data/yk/projects/Hetero-task:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES="${1:-2}"

python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_camera_pyramid_stage2_local.yaml \
  --stage1_model_dir /mnt/sdb/public/data/yk/result/hetero-task/HEAL_merged_stage2 \
  --output_dir /mnt/sdb/public/data/yk/result/hetero-task/HEAL_stage2_lidar_camera_pyramid
