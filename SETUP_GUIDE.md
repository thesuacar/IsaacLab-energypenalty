# Reproducibility Tutorial — Windows 11 + Conda
## Energy-Aware Reward Shaping for Robotic Grasping in Isaac Lab
**Repository:** https://github.com/thesuacar/IsaacLab-energypenalty  
**Robot:** Franka Emika Panda | **Algorithm:** PPO (via rl_games) | **Simulator:** NVIDIA Isaac Lab  
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

Troubleshooting: If conda cannot be found, try in powershell:
`<path to conda.exe in miniconda/anaconda> init powershell`
then test conda again.
---

## Step 5 — Install Isaac Lab Extensions

With the conda environment active, install all Isaac Lab extensions and RL frameworks (RSL-RL, RL Games, SKRL, Stable Baselines3), we will only use RL Games for this thesis but the others are available for future experimentation:

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
:: Replace the path below with your actual Isaac Sim installation path, using Command Prompt and Administrator
mklink /D "<path to your working folder>" "<path to isaacsim installation>"
```

Then re-run Step 5.

---

## Step 7 — Understand the Three Reward Configurations

This thesis compares three reward setups for the Franka Panda grasping task:

| Config | Description |
|--------|-------------|
| **Baseline** | Grasp success reward only |
| **Effort Penalty** | Grasp success + integrated joint torque penalty |
| **Effort + Acceleration Penalty** | Grasp success + torque penalty + motion smoothness (jerk) penalty |

Pre-generated result plots from the thesis runs are in the `thesis_plots\` folder.
Videos of trained policies are in `logs\rl_games\franka_lift\<taskname>42\videos\`.

---

## Step 8 — Run Training

Make sure your conda environment is activated (`conda activate env_isaaclab`) before every session.

### Baseline (no energy penalty)

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-v0 --headless --seed <n> --num_envs 4096
```

### With Joint Effort Penalty

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-JointEffort-v0 --headless --seed <n> --num_envs 4096
```

### With Effort + Acceleration Penalty

```bat
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac-Lift-Cube-Franka-JointEffortAndAcceleration-v0 --headless --seed <n> --num_envs 4096
```

> Remove `--headless` if you want to watch the simulation in the GUI — but headless mode trains significantly faster and prevents "Not Responding" error.
> Note: in some folder names, comments, and plot labels, "JointEffortAndJerk" may be used instead of "JointEffortAndAcceleration" due to an earlier naming choice. These refer to the same Config 3 with the acceleration penalty. This was later renamed for clarity but some legacy names remain in the codebase. Please be wary of name errors when navigating the code and logs.

**For reproducibility, always fix the random seed:**

> The default seed is 42. The experiment uses 5 seeds throughout different conditions: 42, 123, 456, 789, 999.

**To stop training:** press `Ctrl+Break` (or `Ctrl+Fn+B` on laptops). Do **not** use `Ctrl+C` on Windows, as it may leave background processes running.

---

## Step 9 — Monitor Training with TensorBoard

Training logs are saved to:
```
logs\rl_games\franka_lift\<timestamp>\
```
You can rename it later for clarity (e.g. `logs\rl_games\franka_lift\baseline42\`) but keep the internal structure intact for TensorBoard compatibility.

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

Replace `<timestamp>` and `<iteration>` with the actual folder name and checkpoint file from your run. Task names are specified in the `--task` argument also during training. This launches the simulation GUI so you can visually inspect the robot's motion and record evaluation metrics. If your computer is not powerful enough, use the headless mode (--headless). Regardless, a video (`.mp4`) is saved to `logs\rl_games\franka_lift\<timestamp>\videos\` for later review.

---

## Step 11 — Visualise Results

The `evaluation\` folder contains the plotting and energy proxy calculation scripts used for the thesis figures.

Troubleshooting: install packages required into your environment via powershell if not already installed:
`pip install numpy matplotlib tensorboard`

```bat
cd evaluation

python evaluate_energy.py evaluate_energy.py --task <task name (e.g Isaac-Lift-Cube-Franka-v0)> --checkpoint <path to checkpoint> --label <label of output file (e.g. baselinexxx)> --num_envs 16 --num_episodes 50 --headless
python extract_metrics.py --label <label of output file>

cd evaluation
python plot_effort.py
python plot_effortacceleration.py
python plot_baseline.py
python plot_metrics.py
```

This generates comparison figures for success metrics, energy proxies, and training curves across the three reward configurations.

there are plots are saved to `thesis_plots\` for reference. You can modify the plotting scripts to generate custom visualisations or compare different runs.
---

The codebase contains training logs, plots, videos and scripts for all three reward configurations, results from the thesis experiment, for your reference.

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
|   |   └── jointeffortacceleration\   ← Effort + acceleration penalty
│   ├── evaluate_energy.py  ← Script to calculate energy proxy metrics from logs
|   ├── plot_baseline.py       ← Plotting script for baseline results
|   ├── plot_effort.py         ← Plotting script for joint effort penalty results
|   └── plot_effortacceleration.py     ← Plotting script for joint effort + acceleration penalty 

```