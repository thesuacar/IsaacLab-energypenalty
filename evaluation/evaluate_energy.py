# =============================================================================
# Energy Evaluation Script
# Thesis: Energy-Aware Reward Shaping for Robotic Grasping
# Author: Su Acar, Tilburg University, 2026
#
# Loads trained policy checkpoints (baseline and config2) and runs them in
# evaluation mode, recording joint torques each step to compute an energy
# proxy (integrated torque L2 norm) per episode.
# =============================================================================

'''
To run baseline:
cd ~/IsaacLab
./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 \
  --checkpoint logs/rl_games/franka_lift/baseline/SEED42/nn/franka_lift.pth \
  --label baseline_seed42 --num_envs 16 --num_episodes 50 --headless

./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 \
  --checkpoint logs/rl_games/franka_lift/baseline/SEED123/nn/franka_lift.pth \
  --label baseline_seed123 --num_envs 16 --num_episodes 50 --headless

./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 \
  --checkpoint logs/rl_games/franka_lift/baseline/SEED456/nn/franka_lift.pth \
  --label baseline_seed456 --num_envs 16 --num_episodes 50 --headless
'''

'''
To run config2:
cd ~/IsaacLab
./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 \
  --checkpoint logs/rl_games/franka_lift/config2/SEED42/nn/franka_lift.pth \
  --label config2_seed42 --num_envs 16 --num_episodes 50 --headless

./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 \
  --checkpoint logs/rl_games/franka_lift/config2/SEED123/nn/franka_lift.pth \
  --label config2_seed123 --num_envs 16 --num_episodes 50 --headless

./isaaclab.sh -p evaluate_energy.py --task Isaac-Lift-Cube-Franka-v0 \
  --checkpoint logs/rl_games/franka_lift/config2/SEED456/nn/franka_lift.pth \
  --label config2_seed456 --num_envs 16 --num_episodes 50 --headless
'''

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate energy consumption of trained policies.")
parser.add_argument("--task",         type=str, required=True,  help="Task name.")
parser.add_argument("--checkpoint",   type=str, required=True,  help="Path to .pth checkpoint.")
parser.add_argument("--label",        type=str, required=True,  help="Label for output file (e.g. baseline_seed42).")
parser.add_argument("--num_envs",     type=int, default=16,     help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=50,     help="Number of episodes to evaluate.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim launch ---
import json
import yaml
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

# ── Load environment ──────────────────────────────────────────────────────────

device = args_cli.device if args_cli.device else "cuda:0"

env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
env_cfg.episode_length_s = 5.0

env = gym.make(args_cli.task, cfg=env_cfg)

# ── Load agent config and policy ─────────────────────────────────────────────

params_path = os.path.join(
    os.path.dirname(args_cli.checkpoint), "..", "params", "agent.yaml"
)
with open(params_path, "r") as f:
    agent_cfg = yaml.safe_load(f)

vecenv.register(
    "IsaacRlgWrapper",
    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
)
env_configurations.register(
    "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
)

agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
agent_cfg["params"]["config"]["device"]      = device
agent_cfg["params"]["config"]["device_name"] = device

runner = Runner()
runner.load(agent_cfg)
runner.reset()

player = runner.create_player()
player.restore(args_cli.checkpoint)
player.reset()

# ── Evaluation loop ───────────────────────────────────────────────────────────

print(f"\n[INFO] Evaluating '{args_cli.label}' for {args_cli.num_episodes} episodes...")

robot = env.unwrapped.scene["robot"]

episode_energies  = []
episode_successes = []
episode_lengths   = []

obs, _ = env.reset()
obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)

completed       = 0
current_energy  = torch.zeros(args_cli.num_envs, device=device)
current_steps   = torch.zeros(args_cli.num_envs, device=device)
current_success = torch.zeros(args_cli.num_envs, device=device)

while completed < args_cli.num_episodes:
    with torch.no_grad():
        action = player.get_action(obs_tensor, is_deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
    obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)

    # energy proxy: L2 norm of joint torques each step
    torques      = robot.data.applied_torque          # (num_envs, num_joints)
    torque_norm  = torch.norm(torques, dim=-1)        # (num_envs,)
    current_energy += torque_norm
    current_steps  += 1

    # success detection via lifting reward
    if "Episode_Reward/lifting_object" in info:
        lift = torch.tensor(info["Episode_Reward/lifting_object"], device=device)
        current_success = torch.maximum(current_success, (lift > 0).float())

    # handle resets
    done = torch.tensor(
        np.array(terminated) | np.array(truncated), dtype=torch.bool, device=device
    )
    for i in done.nonzero(as_tuple=True)[0]:
        if completed >= args_cli.num_episodes:
            break
        episode_energies.append(current_energy[i].item())
        episode_lengths.append(current_steps[i].item())
        episode_successes.append(current_success[i].item())
        current_energy[i]  = 0.0
        current_steps[i]   = 0.0
        current_success[i] = 0.0
        completed += 1
        if completed % 10 == 0:
            print(f"  Episodes completed: {completed}/{args_cli.num_episodes}")

# ── Summary ───────────────────────────────────────────────────────────────────

mean_energy  = float(np.mean(episode_energies))
std_energy   = float(np.std(episode_energies))
success_rate = float(np.mean(episode_successes)) * 100
mean_length  = float(np.mean(episode_lengths))

print(f"\n{'='*55}")
print(f"  Label:            {args_cli.label}")
print(f"  Episodes:         {args_cli.num_episodes}")
print(f"  Mean energy:      {mean_energy:.2f} ± {std_energy:.2f}")
print(f"  Success rate:     {success_rate:.1f}%")
print(f"  Mean ep length:   {mean_length:.1f} steps")
print(f"{'='*55}\n")

# ── Save results ──────────────────────────────────────────────────────────────

output_dir = os.path.expanduser("~/IsaacLab/thesis_plots/energy_eval")
os.makedirs(output_dir, exist_ok=True)

results = {
    "label":             args_cli.label,
    "num_episodes":      args_cli.num_episodes,
    "mean_energy":       mean_energy,
    "std_energy":        std_energy,
    "success_rate":      success_rate,
    "mean_ep_length":    mean_length,
    "episode_energies":  episode_energies,
    "episode_successes": episode_successes,
}
out_path = os.path.join(output_dir, f"{args_cli.label}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"[INFO] Results saved to: {out_path}")

env.close()
simulation_app.close()
