<div align="center">

# 🚦 BOLT
### Base-free Online Lightweight Adaptation<br/>for Preparation-Free Heterogeneous Cooperative Perception

<!-- Badges: replace ARXIV_ID after the arXiv upload -->
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/abs/ARXIV_ID)
[![Project Status: Code Released](https://img.shields.io/badge/status-code%20released-brightgreen.svg)](#-news)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-ee4c2c.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/sidiangongyuan/BOLT?style=social)](https://github.com/sidiangongyuan/BOLT)

<p align="center"><b>
Drop a ~0.9M-parameter plugin between your detector and a stranger's detector,<br/>
adapt it on the fly with the ego's own predictions, and recover useful cooperation<br/>
without any pre-deployment joint training, without any labels, and without any cooperative data.
</b></p>

<p align="center">
  <img src="images/Teaser.png" width="92%"/>
</p>

</div>

---

## 📰 News

- **2026-05** &nbsp; Paper submitted to a peer-reviewed venue.
- **2026-05** &nbsp; **Code is publicly released** 🎉
- **2026-05** &nbsp; arXiv preprint will be linked here once the ID is assigned.

---

## ✨ TL;DR

Most "heterogeneous cooperative perception" methods assume that all agents pre-train *together* — shared protocols, joint optimization, or collaborator-specific calibration data. **This breaks the moment two agents from different vendors meet on the road.**

**BOLT** assumes nothing of the sort. Each agent ships with its own independently trained single-agent detector. When two agents meet, BOLT inserts a tiny adaptive plugin between the *neighbor's* feature stream and the *ego's* frozen fusion module, and adapts the plugin **online** via **ego-as-teacher distillation**. No labels. No cooperative training data. No shared protocol.

<p align="center">
  <img src="images/Framework.png" width="92%"/>
</p>

Across **DAIR-V2X** and **OPV2V**, with multiple LiDAR/camera encoder pairs and multiple fusion strategies, BOLT consistently turns degraded preparation-free cooperation into useful cooperation that **surpasses ego-only performance**.

---

## ✅ Highlights

| | Feature | Description |
|---|---|---|
| 1 | 🆓 **Preparation-free** | Each agent uses its own independently trained detector. No joint training. No shared protocol. |
| 2 | 🔄 **Online adaptation** | Plugin parameters are updated at deployment by single-pass test-time distillation. |
| 3 | 🪶 **Lightweight** | About **0.9M trainable parameters**. Encoders, fusion module, and detection head are all frozen. |
| 4 | 🏷️ **Label-free, data-free** | Ego predictions act as teacher — no labels, no cooperative training data. |
| 5 | 🧩 **Plug-and-play** | Works across PointPillars / SECOND / Lift-Splat-Shoot encoders and across multiple fusion strategies. |

---

## 📊 Results

Evaluated on **DAIR-V2X** (real V2I), **OPV2V** (simulated V2V), and **V2X-Real**.

In the preparation-free setting, vanilla unadapted fusion typically falls *below* ego-only detection — cooperation actively *hurts*. BOLT reverses this:

- 🚀 Up to **+32.3 AP@50** over unadapted fusion in the preparation-free setting.
- 📈 **Surpasses ego-only** on every evaluated encoder pair across DAIR-V2X and OPV2V.
- 🪶 Trains only **~0.9M parameters** per plugin.

See the paper for full tables, ablations, and qualitative BEV comparisons.

---

## 🧠 Method at a Glance

BOLT is a thin, frozen-everything-except-the-plugin design.

```
                                Ego stream                                  Frozen
   ┌──────────┐   F_e     ┌──────────────────────────┐
   │ Ego enc. │ ───────►  │                          │     ┌────────┐
   └──────────┘           │   Frozen Fusion Module   │ ──► │ Frozen │ ──► detections
   ┌──────────┐   F_n     │                          │     │  Head  │
   │ Nbr enc. │ ──┐  ┌──► │                          │     └────────┘
   └──────────┘   │  │    └──────────────────────────┘
                  ▼  │
            ┌────────────┐                              Ego logits
            │  Plugin θ  │ ◄──── ego-as-teacher ────────  (no labels)
            └────────────┘            (online)
            (~0.9M params,
             only trainable)
```

The plugin has three stages — **statistical alignment → semantic transformation → selective gating** — each designed so its own no-op configuration recovers an exact identity at initialization. This guarantees that an *un*adapted plugin never makes cooperation worse than it already is, while a *trained* plugin closes the cross-agent feature gap.

---

## ⚙️ Installation

> **TL;DR:** the codebase is built on top of [**HEAL**](https://github.com/yifanlu0227/HEAL). Please follow HEAL's environment setup first; BOLT has no extra system dependencies on top of HEAL.

```bash
# 1. Follow HEAL's installation instructions:
#    https://github.com/yifanlu0227/HEAL#installation
#    (creates the conda env, installs pytorch / spconv / cumm / etc.)

# 2. Then clone this repo and finalize:
git clone https://github.com/sidiangongyuan/BOLT.git
cd BOLT
pip install -r requirements.txt
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
```

If you can run HEAL's training and inference scripts, you can run BOLT.

---

## 📦 Data Preparation

Same datasets and same preprocessing as HEAL — see [HEAL's data preparation](https://github.com/yifanlu0227/HEAL#data-preparation) for the full instructions.

| Dataset | Source |
|---|---|
| **DAIR-V2X-C** | [DAIR-V2X](https://thudair.baai.ac.cn/index) with [complemented annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/) |
| **OPV2V** | [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) (also `additional-001.zip` for camera) |
| **OPV2V-H** | [Hugging Face](https://huggingface.co/datasets/yifanlu0227/OPV2V-H) |
| **V2X-Real** | [Official site](https://mobility-lab.seas.ucla.edu/v2x-real/) |

---

## 🚀 Quick Start

The full BOLT pipeline has three stages. The first two come straight from HEAL; only the third is BOLT-specific.

### Stage 1 — Train each agent's single-agent detector (HEAL Stage 1)

Each agent independently trains its own detector. No collaboration involved.

```bash
python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_pyramid_local.yaml
```

### Stage 2 — Train a HEAL-style fusion backbone (HEAL Stage 2)

This produces the *frozen* base model that BOLT will adapt around. Consult HEAL for details.

```bash
python -m opencood.tools.train \
  -y opencood/hypes_yaml/dairv2x/HEAL/lidar_pp_second_stage2.yaml \
  --stage1_model_dir <path_to_stage1_checkpoint>
```

### Stage 3 — Online adaptation with BOLT 🟢 *(this repo's contribution)*

At deployment, encoders / fusion / head all stay frozen. Only the plugin updates, online, with one gradient step per incoming sample, supervised by the ego detector's own high-confidence predictions.

```bash
python -m opencood.tools.online_adapt \
  --model_dir <path_to_heal_checkpoint> \
  --output_dir <output_path> \
  --lr 1e-4 --epochs 1 \
  --teacher_conf_thresh 0.3 \
  --boost_weight 0.1 --boost_lo 0.1 --boost_hi 0.3
```

### Evaluate

```bash
python -m opencood.tools.inference --model_dir <path_to_checkpoint>
```

For ready-made pipelines, see [`scripts/`](scripts/) — including `scripts/inference/inference.sh` and the multi-agent assembly scripts under `scripts/more_agents/`.

---

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
│   ├── train.py                      # Standard training (HEAL stages)
│   └── inference.py                  # Evaluation
├── hypes_yaml/                       # Configs for DAIR-V2X, OPV2V, V2X-Real
└── data_utils/                       # Dataset loaders + pre/post processors

scripts/                              # Reproducible runners
├── inference/                        # Off-the-shelf inference scripts
├── train/                            # End-to-end training scripts
└── more_agents/                      # Multi-agent (3+ cars) assembly + adaptation
```

---

## 🙏 Acknowledgements

This codebase is built on top of [**HEAL**](https://github.com/yifanlu0227/HEAL) and [**OpenCOOD**](https://github.com/DerrickXuNu/OpenCOOD). We thank the authors of DAIR-V2X, OPV2V, OPV2V-H, and V2X-Real for releasing the datasets that made this work possible.

---

## 📖 Citation

If you find BOLT useful, please cite:

```bibtex
@article{bolt2026,
  title   = {BOLT: Base-free Online Lightweight Adaptation for Preparation-Free Heterogeneous Cooperative Perception},
  author  = {Yang, Kang and Bu, Tianci and Wang, Peng and Li, Deying and Wang, Yongcai},
  journal = {arXiv preprint arXiv:ARXIV_ID},
  year    = {2026}
}
```

*(BibTeX will be updated once the arXiv ID is assigned.)*

---

## 📄 License

Released under the [MIT License](LICENSE).
