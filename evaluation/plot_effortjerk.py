"""
Config 3 — Joint Effort and Jerk Penalty Results Plotter
Reads TensorBoard event files from Isaac Lab training runs and produces
publication-ready comparison graphs for all 3 seeds.

Usage:
    python plot_config3.py

Output:
    Plots saved to ~/IsaacLab/thesis_plots/config3/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Configuration ─────────────────────────────────────────────────────────────

LOG_BASE = os.path.expanduser("~/IsaacLab-energypenalty/logs/rl_games/franka_lift")

RUNS = {
    "Seed 42":  "jointeffortjerk42",
    "Seed 123": "jointeffortjerk123",
    "Seed 456": "jointeffortjerk456",
}

# Metrics to plot — (TensorBoard tag, plot title, y-axis label)
METRICS = [
    ("rewards/iter",                                "Total Reward per Iteration",        "Reward"),
    ("Episode/Episode_Reward/lifting_object",       "Lifting Object Reward",             "Reward"),
    ("Episode/Episode_Reward/reaching_object",      "Reaching Object Reward",            "Reward"),
    ("Episode/Episode_Reward/action_rate",          "Action Rate Penalty",               "Penalty"),
    ("Episode/Episode_Reward/joint_vel",            "Joint Velocity Penalty",            "Penalty"),
    ("Episode/Episode_Reward/joint_effort",         "Joint Effort Penalty (Energy)",     "Penalty"),
    ("Episode/Episode_Reward/joint_acc",            "Joint Acceleration Penalty (Jerk)", "Penalty"),
    ("Episode/Curriculum/action_rate",              "Curriculum: Action Rate Weight",    "Weight"),
    ("Episode/Curriculum/joint_vel",                "Curriculum: Joint Velocity Weight", "Weight"),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_plots", "jointeffortjerk")
COLORS     = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green
SMOOTH_WEIGHT = 0.85

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scalar(run_path, tag):
    summaries_path = os.path.join(run_path, "summaries")
    if not os.path.isdir(summaries_path):
        print(f"  [WARN] summaries folder not found: {summaries_path}")
        return None, None
    ea = EventAccumulator(summaries_path, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        return None, None
    events  = ea.Scalars(tag)
    steps   = np.array([e.step  for e in events])
    values  = np.array([e.value for e in events])
    return steps, values

def smooth(values, weight=SMOOTH_WEIGHT):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)

def plot_metric(tag, title, ylabel, runs_data):
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, color), (steps, values) in zip(
        [(l, c) for l, c in zip(runs_data.keys(), COLORS)],
        runs_data.values()
    ):
        if steps is None:
            continue
        ax.plot(steps, values,        color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth(values), color=color, linewidth=2.0, label=label)

    ax.set_title(f"Config 3 (Joint Effort + Jerk) — {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    filename  = tag.replace("/", "_") + ".png"
    filepath  = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nReading logs from: {LOG_BASE}")
    print(f"Saving plots to:   {OUTPUT_DIR}/\n")

    for tag, title, ylabel in METRICS:
        print(f"Plotting: {title}")
        runs_data = {}
        for seed_label, folder in RUNS.items():
            run_path = os.path.join(LOG_BASE, folder)
            steps, values = load_scalar(run_path, tag)
            if steps is None:
                print(f"  [{seed_label}] tag '{tag}' not found — skipping")
            runs_data[seed_label] = (steps, values)
        plot_metric(tag, title, ylabel, runs_data)

    # ── Mean ± std band for total reward ──────────────────────────────────────
    print("\nPlotting mean ± std band for total reward...")
    all_values, ref_steps = [], None

    for folder in RUNS.values():
        run_path = os.path.join(LOG_BASE, folder)
        steps, values = load_scalar(run_path, "rewards/iter")
        if steps is not None:
            all_values.append(smooth(values))
            if ref_steps is None:
                ref_steps = steps

    if len(all_values) > 1:
        min_len    = min(len(v) for v in all_values)
        all_values = np.array([v[:min_len] for v in all_values])
        ref_steps  = ref_steps[:min_len]
        mean       = np.mean(all_values, axis=0)
        std        = np.std(all_values,  axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ref_steps, mean, color="#4C72B0", linewidth=2.5, label="Mean (3 seeds)")
        ax.fill_between(ref_steps, mean - std, mean + std,
                        color="#4C72B0", alpha=0.25, label="± 1 std dev")
        ax.set_title("Config 3 (Joint Effort + Jerk) — Total Reward (Mean ± Std, 3 Seeds)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(OUTPUT_DIR, "rewards_mean_std_band.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filepath}")
    else:
        print("  Not enough seeds with data to plot mean ± std band yet.")

    print(f"\nDone! All plots saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
