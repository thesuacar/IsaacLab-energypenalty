# IsaacLab — Energy-Penalty Extension

> **Thesis project** — *How do different energy-efficient reward functions impact the learning speed and success rate of robotic systems in grasping tasks?*
> Su Acar · Tilburg University · Cognitive Science & Artificial Intelligence · 2026

This repository is a fork of [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) extended with energy-aware reward shaping for robotic grasping research. It trains a **Franka Emika Panda** arm using **Proximal Policy Optimization (PPO)** and compares three reward configurations that progressively introduce energy-consumption penalties derived from joint torque and motion smoothness (jerk).

---

## Table of Contents

- [Overview](#overview)
- [Reward Configurations](#reward-configurations)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Visualisation](#visualisation)
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

## Repository Structure

```
IsaacLab-energypenalty/
├── source/                  # Core IsaacLab source (upstream)
│   └── ...
├── scripts/                 # Training entry-points and experiment runners
├── thesis_plots/
│   └── baseline/            # Saved plots and training curves
├── visualisation/           # Matplotlib-based analysis and plotting scripts
├── docs/                    # Documentation (upstream + thesis additions)
├── tools/                   # Helper utilities
├── isaaclab.sh / .bat       # Environment launch scripts
└── README.md
```

---

## Requirements

- **OS:** Ubuntu 22.04 (Linux x86-64)
- **Python:** 3.11.x
- **NVIDIA Isaac Sim:** 4.5 / 5.0 / 5.1 (see [version table](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- **GPU:** NVIDIA GPU with CUDA support (required for Isaac Sim)

Python dependencies (see `environment.yml` or `pyproject.toml`):
- `torch`
- `matplotlib`
- `pandas`
- `numpy`

---

## Installation

1. **Install Isaac Sim** following the [official guide](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).

2. **Clone this repository:**
   ```bash
   git clone https://github.com/thesuacar/IsaacLab-energypenalty.git
   cd IsaacLab-energypenalty
   ```

3. **Set up the Isaac Lab environment:**
   ```bash
   # Linux
   ./isaaclab.sh --install

   # Windows
   isaaclab.bat --install
   ```

4. **Install additional dependencies:**
   ```bash
   pip install matplotlib pandas
   ```

---

## Running Experiments

Train each reward configuration from the `scripts/` directory. Replace `<config>` with `baseline`, `effort`, or `effort_jerk`.

```bash
./isaaclab.sh -p scripts/train.py --task FrankaCubeGrasp-<config> --num_envs 4096
```

To run all three configurations sequentially:
```bash
for cfg in baseline effort effort_jerk; do
    ./isaaclab.sh -p scripts/train.py --task FrankaCubeGrasp-$cfg --num_envs 4096
done
```

Training logs and checkpoints are saved to `logs/` by default.

---

## Visualisation

After training, generate comparison plots using the scripts in `visualisation/`:

```bash
python visualisation/plot_results.py --log_dir logs/
```

Pre-generated plots from thesis runs are available in `thesis_plots/baseline/`.

---

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
