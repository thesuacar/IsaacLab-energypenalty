# IsaacLab — Energy-Penalty Extension

> **Thesis project** — *How do different energy-efficient reward functions impact the learning speed and success rate of robotic systems in grasping tasks?*
> Su Acar · Tilburg University · Cognitive Science & Artificial Intelligence · 2026

This repository is a fork of [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) extended with energy-aware reward shaping for robotic grasping research. It trains a **Franka Emika Panda** arm using **Proximal Policy Optimization (PPO)** and compares three reward configurations that progressively introduce energy-consumption penalties derived from joint torque and motion smoothness (jerk).

---

## Table of Contents

- [Overview](#overview)
- [Reward Configurations](#reward-configurations)
- [Requirements](#requirements)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Most reinforcement learning approaches for robotic manipulation optimise purely for task success. This project embeds simple, simulation-based energy proxies directly into the reward signal to investigate whether efficiency can be improved without significantly reducing grasp performance.

**Research question:** How can energy-aware reward shaping influence learning efficiency, task success rate, and energy consumption of a Franka Panda arm in object-grasping tasks?

**Metrics tracked per configuration:**
- Grasp success rate (% of episodes with a stable, completed grasp)
- Learning speed (training steps to reach a stable success-rate threshold)
- Energy proxy (average integrated joint torque + jerk per episode)

---

## Reward Configurations

| ID | Name | Description |
|----|------|-------------|
| **C1** | Baseline | Grasp success reward only |
| **C2** | Effort penalty | Grasp success + integrated joint-torque penalty |
| **C3** | Effort + jerk penalty | Grasp success + joint-torque penalty + jerk-based smoothness penalty |

Each configuration is trained multiple times to reduce variance. Averaged results are used for comparison.


---

## Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux 64-bit or Windows 11 64-bit |
| GPU | NVIDIA GPU with ≥ 12 GB VRAM (RTX 3080 or better recommended) |
| RAM | ≥ 32 GB |
| Storage | ≥ 30 GB free (REQUIRED for IsaacSim and conda environment)|
| NVIDIA Driver | ≥ 528.33 |
| Python | 3.11.x (REQUIRED for latest IsaacSim) |
| Conda | Miniconda (recommended) |

---

## Setup Instructions
Detailed setup instructions are provided in [SETUP_GUIDE.md](./SETUP_GUIDE.md). Follow the steps to install Isaac Sim, set up the conda environment, and run the training scripts for each reward configuration.

## Results

Full results, training curves, and analysis are presented in the thesis document. The `thesis_plots/` directory contains the figures used in the final report.

---

## Acknowledgements
The baseline training code was developed together with Andrei Vasile at Tilburg University (2026).

## Citation

If you use this work, please cite both this project and the upstream Isaac Lab framework:

**This project:**
```bibtex
@misc{acar2026energypenalty,
  author       = {Su Acar},
  title        = {IsaacLab-energypenalty: Energy-Aware Reward Shaping for Robotic Grasping},
  year         = {2026},
  howpublished = {\url{https://github.com/thesuacar/IsaacLab-energypenalty}},
  note         = {BSc Thesis, Tilburg University}
}
```

**Upstream Isaac Lab:**
```bibtex
@article{mittal2025isaaclab,
  title   = {Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author  = {Mayank Mittal et al.},
  journal = {arXiv preprint arXiv:2511.04831},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.04831}
}
```

---

## License

The Isaac Lab framework (and this fork) is released under the [BSD-3-Clause License].
Thesis-specific additions in this fork are also released under BSD-3-Clause.

Note: this project depends on NVIDIA Isaac Sim, which has its own proprietary licensing terms — see Isaac Sim documentation.
