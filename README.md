<div align="center">

# BOLT

### Base-Free Online Lightweight Adaptation for Heterogeneous Cooperative Perception

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-ee4c2c.svg)](https://pytorch.org/)

</div>

<p align="center">
  <img src="images/Teaser.png" width="100%"/>
</p>

## Overview

Existing heterogeneous cooperative perception methods universally require a collaboratively trained fusion module, assuming access to multi-agent data during training. In practice, vehicles from different manufacturers ship with independently trained detectors, making joint cooperative training infeasible.

**BOLT** addresses this *base-free* setting by inserting a lightweight adaptive plugin (~0.9M parameters) between the neighbor encoder and the frozen fusion module. The plugin is trained online at test time via ego-as-teacher distillation — no ground-truth labels, no cooperative training data.

<p align="center">
  <img src="images/Framework.png" width="100%"/>
</p>

### Highlights

| | Feature | Description |
|---|---|---|
| 1 | **Base-free** | No cooperative training required. Each agent uses its own independently trained detector. |
| 2 | **Online adaptation** | Plugin parameters are updated at test time via self-supervised distillation. |
| 3 | **Lightweight** | Only ~0.9M trainable parameters in the plugin module. |
| 4 | **Label-free** | Uses ego predictions as pseudo-labels — no annotations needed at adaptation time. |
| 5 | **Versatile** | Works across multiple encoder pairs (PointPillar, SECOND, Camera) and fusion strategies. |

## Results

Evaluated on three benchmarks: **DAIR-V2X**, **OPV2V**, and **V2X-Real**.

BOLT consistently recovers ego-only performance from severely degraded base-free baselines across all settings.

## Data Preparation

- **DAIR-V2X-C**: Download from [DAIR-V2X](https://thudair.baai.ac.cn/index) with [complemented annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).
- **OPV2V**: Download from [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). Also download `additional-001.zip` for camera data.
- **OPV2V-H**: Download from [Huggingface Hub](https://huggingface.co/datasets/yifanlu/OPV2V-H).
- **V2X-Real**: Download from the [official website](https://mobility-lab.seas.ucla.edu/v2x-real/).

## Installation

```bash
# Create environment
conda create -n bolt python=3.8 pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate bolt

# Install dependencies
pip install -r requirements.txt

# Install spconv (match your CUDA version)
pip install spconv-cu116

# Install project
python setup.py develop

# Compile IoU CUDA ops
python opencood/utils/setup.py build_ext --inplace
```

## Usage

### Step 1: Train Single-Agent Encoders (HEAL Stage 1)

Train each encoder independently on single-agent data:

```bash
python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_pyramid_local.yaml
```

### Step 2: Train HEAL Fusion Backbone (Stage 2)

Align encoders into a shared protocol space:

```bash
python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_pp_second_stage2.yaml \
  --stage1_model_dir <path_to_stage1_checkpoint>
```

### Step 3: Online Adaptation with BOLT

Run online test-time training with the plugin:

```bash
python -m opencood.tools.online_adapt \
  --model_dir <path_to_heal_checkpoint> \
  --output_dir <output_path> \
  --lr 1e-4 --epochs 3 \
  --teacher_conf_thresh 0.3 \
  --boost_weight 0.1 --boost_lo 0.1 --boost_hi 0.3
```

### Inference

```bash
python -m opencood.tools.inference \
  --model_dir <path_to_checkpoint>
```

## Project Structure

```
opencood/
├── models/
│   ├── plugin/              # BOLT adaptive plugin (AdaIN + residual blocks)
│   ├── heter_pyramid_collab.py   # Main heterogeneous pyramid model
│   ├── heter_encoders.py         # Multi-modality encoder registry
│   └── fuse_modules/             # Fusion strategies (pyramid, attention, etc.)
├── tools/
│   ├── online_adapt.py      # Online TTT with ego-as-teacher distillation
│   ├── train.py             # Standard training
│   ├── train_stage2.py      # HEAL stage-2 alignment training
│   └── inference.py         # Evaluation
├── hypes_yaml/              # Config files for DAIR-V2X, OPV2V, V2X-Real
└── data_utils/              # Dataset loaders and pre/post processors
```

## Acknowledgement

This codebase is built upon [HEAL](https://github.com/yifanlu0227/HEAL) and [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD).

## Citation

If you find this work useful, please cite:

```bibtex
@article{bolt2025,
  title={BOLT: Base-Free Online Lightweight Adaptation for Heterogeneous Cooperative Perception},
  year={2025}
}
```
