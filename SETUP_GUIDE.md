# Reproducibility Tutorial — Windows 11 + Conda
## Energy-Aware Reward Shaping for Robotic Grasping in Isaac Lab
**Repository:** https://github.com/thesuacar/IsaacLab-energypenalty  
**Robot:** Franka Emika Panda | **Algorithm:** PPO (via RSL-RL) | **Simulator:** NVIDIA Isaac Lab  
**Platform:** Windows 11 (64-bit) | **Environment manager:** Miniconda

---

## Prerequisites

Make sure your system meets the following before starting:

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


## Step 1 — Install NVIDIA Isaac Sim

Isaac Lab runs on top of Isaac Sim. Install it via the official guide: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html

---

## Step 3 — Install Git and Clone the Repository

Open powershell and run:

```
git clone https://github.com/thesuacar/IsaacLab-energypenalty.git
cd IsaacLab-energypenalty
```

---

## Step 4 — Create the Conda Environment

Isaac Sim requires **Python 3.11**. Create a dedicated conda environment using the `isaaclab.bat` helper script:

```bat
isaaclab.bat --conda env_isaaclab
```

This creates a conda environment named `env_isaaclab` with the correct Python version and links it to your Isaac Sim installation.

Then activate it:

```bat
conda activate env_isaaclab
```

> You should now see `(env_isaaclab)` at the start of your prompt. Always activate this environment before running any commands in this tutorial.

---

## Step 5 — Install Isaac Lab Extensions

With the conda environment active, install all Isaac Lab extensions and RL frameworks (RSL-RL, RL Games, SKRL, Stable Baselines3):

```bat
isaaclab.bat --install
```

This installs all packages inside `source/` in editable mode. It may take **10–20 minutes** on first run.

To verify:

```bat
isaaclab.bat --help
```

You should see the list of available flags printed without errors.

---

## Step 6 — Link Isaac Sim (if not already linked)

If `isaaclab.bat` cannot find Isaac Sim automatically, create a symbolic link manually. To resolve this common error, run this in **Command Prompt as Administrator**:

```bat
:: Replace the path below with your actual Isaac Sim installation path
mklink /D _isaac_sim "C:\Users\<YourName>\AppData\Local\ov\pkg\isaac-sim-4.5.0"
```

Then re-run Step 5.

---

## Step 7 — Understand the Three Reward Configurations

This thesis compares three reward setups for the Franka Panda grasping task:

| Config | Description |
|--------|-------------|
| **Baseline** | Grasp success reward only |
| **Effort Penalty** | Grasp success + integrated joint torque penalty |
| **Effort + Jerk Penalty** | Grasp success + torque penalty + motion smoothness (jerk) penalty |

Pre-generated result plots from the thesis runs are in the `thesis_plots\` folder.

---

## Step 8 — Run Training

Make sure your conda environment is activated (`conda activate env_isaaclab`) before every session.

### Baseline (no energy penalty)

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-v0 --headless
```

### With Joint Effort Penalty

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-JointEffort-v0 --headless
```

### With Effort + Jerk Penalty

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-JointEffortAndJerk-v0 --headless
```

> Remove `--headless` if you want to watch the simulation in the GUI — but headless mode trains significantly faster and prevents "Not Responding" error.

**For reproducibility, always fix the random seed:**

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-v0 --headless --seed 42
```

> The default seed is 42. The experiment uses 3 seeds throughout different conditions: 42,123,456.

**To stop training:** press `Ctrl+Break` (or `Ctrl+Fn+B` on laptops). Do **not** use `Ctrl+C` on Windows, as it may leave background processes running.

---

## Step 9 — Monitor Training with TensorBoard

Training logs are saved to:
```
logs\rl_games\franka_lift\<timestamp>\
```

Each run produces:
- `params\` — environment and agent config snapshots
- `nn\` — saved model checkpoints
- `summaries\` — TensorBoard logs for metrics and scalars

Launch TensorBoard (with env active) to monitor training progress:

```bat
isaaclab.bat -p -m tensorboard.main --logdir logs\rl_games\franka_lift\
```

Then open http://localhost:6006 in your browser.

Key metrics to watch:

- `Episode/rew_mean` — average reward per episode (should increase over time)
- `Train/mean_reward` — reward during training rollouts
- `Episode/ep_len_mean` — episode length (should stabilise)

---

## Step 10 — Evaluate a Trained Policy

After training, load a saved checkpoint and run the policy:

```bat
.\isaaclab.bat -p scripts/reinforcement_learning/rl_games/play.py --task <taskname> --num_envs 1 --video --video_length 300 --checkpoint logs/rl_games/franka_lift/<timestamp>/nn/<iteration>.pt
```

Replace `<timestamp>` and `<iteration>` with the actual folder name and checkpoint file from your run. Task names are specified in the `--task` argument also during training. This launches the simulation GUI so you can visually inspect the robot's motion and record evaluation metrics. A video (`.mp4`) is saved to `logs\rl_games\franka_lift\<timestamp>\videos\` for later review.

---

## Step 11 — Visualise Results

The `evaluation\` folder contains the plotting and energy proxy calculation scripts used for the thesis figures.

```bat
cd ~/IsaacLab
./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 --checkpoint <path to checkpoint> --label <label of output file> --num_envs 16 --num_episodes 50 --headless

cd evaluation
python plot_effort.py
python plot_effortjerk.py
python plot_baseline.py
```

This generates comparison figures for:
- Total Reward per Iteration
- Lifting Object Reward
- Reaching Object Reward
- Action Rate Penalty
- Joint Velocity Penalty
- Curriculum: Action Rate Weight
- Curriculum: Joint Velocity Weight
- Average energy usage (integrated joint torque) per episode

Plots are saved to `thesis_plots\` for reference. You can modify the plotting scripts to generate custom visualisations or compare different runs.
---


## File Structure Reference

```
IsaacLab-energypenalty\
├── scripts\
│   └── reinforcement_learning\
│       └── rl_games\
|           └── franka_lift\                 ← PPO training scripts for Franka Lift task
│               ├── train.py          ← Main training entry point
│               └── play.py           ← Policy evaluation / playback
├── source\                       ← Isaac Lab source & task definitions
├── evaluation\
|   ├── thesis_plots\                 ← Pre-generated plots from thesis runs (for reference)
|   |   ├── baseline\          ← Baseline reward config results
|   |   ├── jointeffort\       ← Joint effort penalty config results
|   |   └── jointeffortjerk\   ← Effort + jerk penalty
│   ├── evaluate_energy.py  ← Script to calculate energy proxy metrics from logs
|   ├── plot_baseline.py       ← Plotting script for baseline results
|   ├── plot_effort.py         ← Plotting script for joint effort penalty results
|   └── plot_effortjerk.py     ← Plotting script for joint effort + jerk penalty 

```