<div align="center">

# BOLT
### Online Lightweight Adaptation for Preparation-Free Heterogeneous Cooperative Perception

<!-- Badges: replace ARXIV_ID after upload -->
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/abs/ARXIV_ID)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-ee4c2c.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/sidiangongyuan/BOLT-Base-Free-Online-Lightweight-Adaptation-for-Heterogeneous-Cooperative-Perception?style=social)](https://github.com/sidiangongyuan/BOLT-Base-Free-Online-Lightweight-Adaptation-for-Heterogeneous-Cooperative-Perception)

<p align="center"><b>
Drop a 0.9M-parameter plugin between your detector and a stranger's detector,<br/>
adapt it on the fly with the ego's own predictions, and recover useful cooperation<br/>
without any pre-deployment joint training, without any labels, and without any cooperative data.
</b></p>

<p align="center">
  <img src="images/Teaser.png" width="92%"/>
</p>

</div>

---

## 📰 News

- **2026-05** — Paper submitted to a peer-reviewed venue. Preprint upcoming.
- **2026-05** — Code is publicly released.
- **2026-05** — arXiv preprint will be linked here once available.

## ✨ TL;DR

Most "heterogeneous cooperative perception" methods assume that all agents pre-train *together*: shared protocols, joint optimization, or collaborator-specific calibration data. This breaks the moment cars built by different vendors meet on the road.

**BOLT** assumes nothing. Each agent ships with its own independently trained single-agent detector. When two such agents meet, BOLT inserts a tiny adaptive plugin between the *neighbor's* feature stream and the *ego's* frozen fusion module, and adapts it online via **ego-as-teacher distillation**. No labels, no cooperative training data, no shared protocol.

Across DAIR-V2X and OPV2V, with multiple LiDAR/camera encoder pairs and multiple fusion strategies, BOLT consistently turns degraded preparation-free cooperation into useful cooperation that surpasses ego-only performance.

<p align="center">
  <img src="images/Framework.png" width="92%"/>
</p>

## 🚀 Highlights

| | Feature | Description |
|---|---|---|
| 1 | **Preparation-free** | Each agent uses its own independently trained detector. No joint training, no shared protocol. |
| 2 | **Online adaptation** | Plugin parameters are updated at deployment via single-pass test-time distillation. |
| 3 | **Lightweight** | About **0.9M trainable parameters** in the plugin. Encoders, fusion module, and detection head all stay frozen. |
| 4 | **Label-free, data-free** | Uses ego predictions as a teacher signal — no labels and no cooperative training data required. |
| 5 | **Versatile** | Works across PointPillars / SECOND / Lift-Splat-Shoot encoders and across multiple fusion strategies. |

## 📊 Results

Evaluated on **DAIR-V2X** (real V2I) and **OPV2V** (simulated V2V), with extensions on **V2X-Real**.

In the preparation-free setting, vanilla unadapted fusion typically falls *below* ego-only detection — cooperation actively hurts. BOLT reverses this:

- Up to **+32.3 AP@50** over unadapted fusion in the preparation-free setting.
- **Surpasses ego-only** on every evaluated encoder pair across DAIR-V2X and OPV2V.
- Trains only **~0.9M parameters** per plugin.

See the paper for full tables, ablations, and qualitative BEV comparisons.

## ⚡ Quick Start

```bash
# 1. Environment
conda create -n bolt python=3.8 pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 \
  -c pytorch -c conda-forge
conda activate bolt
pip install -r requirements.txt
pip install spconv-cu116                 # match your CUDA
python setup.py develop
python opencood/utils/setup.py build_ext --inplace

# 2. Run online adaptation (BOLT) on a HEAL checkpoint
python -m opencood.tools.online_adapt \
  --model_dir <path_to_heal_checkpoint> \
  --output_dir <output_path> \
  --lr 1e-4 --epochs 1 \
  --teacher_conf_thresh 0.3 \
  --boost_weight 0.1 --boost_lo 0.1 --boost_hi 0.3
```

## 📦 Data Preparation

| Dataset | Source |
|---|---|
| **DAIR-V2X-C** | [DAIR-V2X](https://thudair.baai.ac.cn/index) with [complemented annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/) |
| **OPV2V** | [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) (also `additional-001.zip` for camera) |
| **OPV2V-H** | [Hugging Face](https://huggingface.co/datasets/yifanlu0227/OPV2V-H) |
| **V2X-Real** | [Official site](https://mobility-lab.seas.ucla.edu/v2x-real/) |

## 🛠 Installation

```bash
conda create -n bolt python=3.8 pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate bolt
pip install -r requirements.txt
pip install spconv-cu116                 # adjust suffix to match your CUDA
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
```

## 📋 Usage

### Step 1 — Train single-agent encoders (HEAL Stage 1)

```bash
python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_pyramid_local.yaml
```

### Step 2 — Train HEAL fusion backbone (Stage 2)

```bash
python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_pp_second_stage2.yaml \
  --stage1_model_dir <path_to_stage1_checkpoint>
```

### Step 3 — Online adaptation with BOLT

```bash
python -m opencood.tools.online_adapt \
  --model_dir <path_to_heal_checkpoint> \
  --output_dir <output_path> \
  --lr 1e-4 --epochs 1 \
  --teacher_conf_thresh 0.3 \
  --boost_weight 0.1 --boost_hi 0.3
```

### Inference

```bash
python -m opencood.tools.inference \
  --model_dir <path_to_checkpoint>
```

## 🗂 Project Structure

```
opencood/
├── models/
│   ├── plugin/                       # BOLT adaptive plugin (AdaIN + residual + gate)
│   ├── heter_pyramid_collab.py       # Heterogeneous pyramid model
│   ├── heter_encoders.py             # Multi-modality encoder registry
│   └── fuse_modules/                 # Fusion strategies (pyramid, attention, ...)
├── tools/
│   ├── online_adapt.py               # Online TTT with ego-as-teacher distillation
│   ├── train.py                      # Standard training
│   └── inference.py                  # Evaluation
├── hypes_yaml/                       # Configs for DAIR-V2X, OPV2V, V2X-Real
└── data_utils/                       # Dataset loaders + pre/post processors
```

## 🙏 Acknowledgements

This codebase is built on top of [HEAL](https://github.com/yifanlu0227/HEAL) and [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). We thank the authors of DAIR-V2X, OPV2V, OPV2V-H, and V2X-Real for releasing the datasets that made this work possible.

## 📖 Citation

If you find BOLT useful, please consider citing:

```bibtex
@article{bolt2026,
  title   = {BOLT: Online Lightweight Adaptation for Preparation-Free Heterogeneous Cooperative Perception},
  author  = {Yang, Kang and Bu, Tianci and Wang, Peng and Li, Deying and Wang, Yongcai},
  journal = {arXiv preprint arXiv:ARXIV_ID},
  year    = {2026}
}
```

(BibTeX will be updated once the arXiv ID is assigned.)

## 📄 License

This project is released under the [MIT License](LICENSE).
