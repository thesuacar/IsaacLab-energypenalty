# =============================================================================
# Energy Evaluation Script
# =============================================================================
'''
Thesis: Energy-Aware Reward Shaping for Robotic Grasping
Author: Su Acar, Tilburg University, 2026

'''

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--task",         type=str, required=True)
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--label",        type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=50)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ──────────────────────────────────────────────────

import json
import yaml
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

# ── Helpers ───────────────────────────────────────────────────────────────────

def unwrap_obs(obs, device):
    """Extract a flat float tensor from whatever the env returns."""
    if isinstance(obs, dict):
        for key in ("policy", "obs"):
            if key in obs:
                obs = obs[key]
                break
        else:
            obs = next(iter(obs.values()))
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    return obs.float().to(device)


def get_action(player, obs_tensor):
    """
    Call the rl_games player robustly.
    rl_games expects input_dict['obs'] to be a plain tensor — but internally
    some wrappers re-wrap it. We bypass that by calling the model directly.
    """
    with torch.no_grad():
        try:
            # rl_games get_action expects plain tensor, not dict
            action = player.get_action(obs_tensor, is_deterministic=True)
        except Exception:
            # Fallback: call model directly
            input_dict = {"obs": obs_tensor, "is_train": False}
            res    = player.model(input_dict)
            action = res["mus"] if "mus" in res else res["mu"]
        return action

# ── Environment ──────────────────────────────────────────────────────────────

device = getattr(args_cli, "device", None) or "cuda:0"

env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
env_cfg.episode_length_s = 5.0

base_env = gym.make(args_cli.task, cfg=env_cfg)

env = RlGamesVecEnvWrapper(
    base_env,
    rl_device=device,
    clip_obs=10.0,
    clip_actions=1.0,
)

# ── Load agent config ─────────────────────────────────────────────────────────

checkpoint_dir = os.path.dirname(args_cli.checkpoint)
seed_dir       = os.path.dirname(checkpoint_dir)
config_dir     = os.path.dirname(seed_dir)

candidate_paths = [
    os.path.join(seed_dir,    "params", "agent.yaml"),
    os.path.join(config_dir,  "params", "agent.yaml"),
]

params_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if params_path is None:
    raise FileNotFoundError("agent.yaml not found in expected locations.")

print(f"[INFO] Using agent config: {params_path}")

with open(params_path, "r") as f:
    agent_cfg = yaml.safe_load(f)

# ── RL-Games setup ────────────────────────────────────────────────────────────

vecenv.register(
    "IsaacRlgWrapper",
    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
)
env_configurations.register(
    "rlgpu",
    {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
)

agent_cfg["params"]["config"]["num_actors"]  = env.num_envs
agent_cfg["params"]["config"]["device"]      = device
agent_cfg["params"]["config"]["device_name"] = device

runner = Runner()
runner.load(agent_cfg)
runner.reset()

player = runner.create_player()
player.restore(args_cli.checkpoint)
player.reset()
player.model.eval()   # ensure eval mode

# ── Evaluation loop ───────────────────────────────────────────────────────────

print(f"\n[INFO] Evaluating '{args_cli.label}' for {args_cli.num_episodes} episodes...")

robot = base_env.unwrapped.scene["robot"]

episode_energy_proxy = []
episode_energy_true  = []
episode_lengths      = []

energy_proxy = torch.zeros(env.num_envs, device=device)
energy_true  = torch.zeros(env.num_envs, device=device)
steps        = torch.zeros(env.num_envs, device=device)

# Reset
raw_obs     = env.reset()
obs_tensor  = unwrap_obs(raw_obs, device)

completed = 0

while completed < args_cli.num_episodes:

    action = get_action(player, obs_tensor)

    # Step — handle both old (4-tuple) and new (5-tuple) gym API
    result = env.step(action)
    if len(result) == 5:
        raw_obs, reward, terminated, truncated, info = result
        done = terminated | truncated
    else:
        raw_obs, reward, done, info = result

    obs_tensor = unwrap_obs(raw_obs, device)

    # ── Energy ──
    torques    = robot.data.applied_torque   # (N, joints)
    velocities = robot.data.joint_vel        # (N, joints)

    energy_proxy += torch.norm(torques, dim=-1)
    energy_true  += torch.abs((torques * velocities).sum(dim=-1))
    steps        += 1

    # ── Episode bookkeeping ──
    if isinstance(done, torch.Tensor):
        done_bool = done.bool()
    else:
        done_bool = torch.tensor(done, dtype=torch.bool, device=device)

    for i in done_bool.nonzero(as_tuple=True)[0]:
        if completed >= args_cli.num_episodes:
            break

        episode_energy_proxy.append(energy_proxy[i].item())
        episode_energy_true.append(energy_true[i].item())
        episode_lengths.append(steps[i].item())

        energy_proxy[i] = 0.0
        energy_true[i]  = 0.0
        steps[i]        = 0.0

        completed += 1
        if completed % 10 == 0:
            print(f"  Episodes: {completed}/{args_cli.num_episodes}")

# ── Results ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Label:    {args_cli.label}")
print(f"Episodes: {args_cli.num_episodes}")
print(f"\nEnergy Proxy (||τ||):")
print(f"  Mean: {np.mean(episode_energy_proxy):.2f}")
print(f"  Std:  {np.std(episode_energy_proxy):.2f}")
print(f"\nTrue Energy (|τ·ω|):")
print(f"  Mean: {np.mean(episode_energy_true):.2f}")
print(f"  Std:  {np.std(episode_energy_true):.2f}")
print("=" * 60 + "\n")

# ── Save ──────────────────────────────────────────────────────────────────────

output_dir = os.path.expanduser("~/IsaacLab/thesis_plots/energy_eval")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"{args_cli.label}.json")

with open(out_path, "w") as f:
    json.dump({
        "label":        args_cli.label,
        "proxy_energy": episode_energy_proxy,
        "true_energy":  episode_energy_true,
        "lengths":      episode_lengths,
    }, f, indent=2)

print(f"[INFO] Saved to: {out_path}")

env.close()
simulation_app.close()