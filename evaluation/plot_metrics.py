# =============================================================================
# Plot Metrics Script
# Thesis: Energy-Aware Reward Shaping for Robotic Grasping
# Author: Su Acar, Tilburg University, 2026
#
# Loads energy JSONs and TB-metrics JSONs from logs/energy_eval, groups them
# by condition (baseline / jointeffort / jointeffortjerk), aggregates across
# seeds, and produces thesis-quality plots.
#
# Expected files in --data_dir:
#   Energy JSONs  : baseline42_energyproxy.json,jointeffort42_energyproxy.json, jointeffortjerk42_energyproxy.json...
#   TB JSONs      : baseline42_tb_metrics.json, jointeffort42_tb_metrics.json, jointeffortjerk42_tb_metrics.json...
#
# Usage:
#   python plot_metrics.py
#   python plot_metrics.py --data_dir logs/energy_eval --out_dir evaluation/thesisplots
# =============================================================================

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "jointeffort", "jointeffortjerk"]

CONDITION_LABELS = {
    "baseline":         "Baseline",
    "jointeffort":      "Joint Effort",
    "jointeffortjerk":  "Joint Effort + Jerk",
}

PALETTE = {
    "baseline":         "#2E86AB",
    "jointeffort":      "#E84855",
    "jointeffortjerk":  "#3BB273",
}

GREY        = "#AAAAAA"
LIGHT_GREY  = "#F0F0F0"
TEXT_COLOUR = "#1A1A2E"


# ── File discovery & loading ──────────────────────────────────────────────────

def identify_condition(filename: str):
    name = Path(filename).stem.lower()
    # Match longest condition first so 'jointeffortjerk' beats 'jointeffort'
    for cond in sorted(CONDITIONS, key=len, reverse=True):
        if name.startswith(cond):
            return cond
    return None


def load_all(data_dir: str):
    data_dir = Path(os.path.expanduser(data_dir))
    if not data_dir.exists():
        sys.exit(f"[ERROR] Data directory not found: {data_dir}")

    energy = defaultdict(list)
    tb     = defaultdict(list)

    for f in sorted(data_dir.glob("*.json")):
        cond = identify_condition(f.name)
        if cond is None:
            print(f"[WARN] Skipping unrecognised file: {f.name}")
            continue
        with open(f) as fh:
            data = json.load(fh)
        if "_tb_metrics" in f.name:
            tb[cond].append(data)
        else:
            energy[cond].append(data)

    for cond in CONDITIONS:
        print(f"[INFO] {CONDITION_LABELS[cond]}: "
              f"{len(energy[cond])} energy file(s), {len(tb[cond])} TB file(s)")

    return energy, tb


# ── Aggregation helpers ───────────────────────────────────────────────────────

def agg(values):
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std())


def agg_series(series_list):
    """Trim all series to the same length, return (xs, mean, std)."""
    if not series_list:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(s) for s in series_list)
    trimmed = np.array([s[:min_len] for s in series_list], dtype=float)
    return np.arange(min_len), trimmed.mean(axis=0), trimmed.std(axis=0)


# ── Style helpers ─────────────────────────────────────────────────────────────

def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOUR, pad=8)
    ax.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOUR)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOUR)
    ax.tick_params(colors=TEXT_COLOUR, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.set_facecolor(LIGHT_GREY)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="white", zorder=0)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_dir / stem}.png")


# ── Plot 1: Energy comparison (proxy + true) ──────────────────────────────────

def plot_energy(energy, out_dir):
    conds  = [c for c in CONDITIONS if energy[c]]
    labels = [CONDITION_LABELS[c] for c in conds]
    x      = np.arange(len(conds))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, key, title, ylabel in [
        (axes[0], "proxy_energy",
         "Proxy Energy per Episode\n(mean ± std across seeds)",
         "Integrated Torque L2 Norm (N·m·steps)"),
        (axes[1], "true_energy",
         "True Energy per Episode\n(mean ± std across seeds)",
         "True Energy (J)"),
    ]:
        means, stds = [], []
        for cond in conds:
            vals = [v for run in energy[cond] for v in run.get(key, [])]
            m, s = agg(vals)
            means.append(m); stds.append(s)

        bars = ax.bar(x, means, yerr=stds, color=[PALETTE[c] for c in conds],
                      width=0.5, zorder=3, capsize=5,
                      error_kw={"elinewidth": 1.5, "ecolor": TEXT_COLOUR},
                      edgecolor="white", linewidth=1.2)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + max(means) * 0.02,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOUR)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
        apply_style(ax, title, "", ylabel)

    fig.tight_layout()
    save(fig, out_dir, "energy_comparison")


# ── Plot 2: Energy distributions (violin) ────────────────────────────────────

def plot_energy_distribution(energy, out_dir):
    conds  = [c for c in CONDITIONS if energy[c]]
    labels = [CONDITION_LABELS[c] for c in conds]
    rng    = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, key, title, ylabel in [
        (axes[0], "proxy_energy", "Proxy Energy Distribution", "Torque L2 (N·m·steps)"),
        (axes[1], "true_energy",  "True Energy Distribution",  "True Energy (J)"),
    ]:
        data = [[v for run in energy[cond] for v in run.get(key, [])] for cond in conds]

        parts = ax.violinplot(data, positions=range(len(conds)),
                              showmedians=True, showextrema=True)
        for pc, cond in zip(parts["bodies"], conds):
            pc.set_facecolor(PALETTE[cond]); pc.set_alpha(0.7)
            pc.set_edgecolor(TEXT_COLOUR);   pc.set_linewidth(0.8)
        for pn in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[pn].set_edgecolor(TEXT_COLOUR); parts[pn].set_linewidth(1.2)

        for i, (vals, cond) in enumerate(zip(data, conds)):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       alpha=0.25, s=6, color=PALETTE[cond], zorder=4)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=10, ha="right")
        apply_style(ax, title, "", ylabel)

    fig.tight_layout()
    save(fig, out_dir, "energy_distribution")


# ── Plot 3: Success rate ──────────────────────────────────────────────────────

def plot_success_rate(tb, out_dir):
    conds  = [c for c in CONDITIONS if tb[c]]
    labels = [CONDITION_LABELS[c] for c in conds]
    x      = np.arange(len(conds))

    means, stds = [], []
    for cond in conds:
        rates = [r["success_rate"] for r in tb[cond] if r.get("success_rate") is not None]
        m, s  = agg(rates) if rates else (0.0, 0.0)
        means.append(m); stds.append(s)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(x, means, yerr=stds, color=[PALETTE[c] for c in conds],
                  width=0.5, zorder=3, capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": TEXT_COLOUR},
                  edgecolor="white", linewidth=1.2)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 1.5, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT_COLOUR)

    ax.set_ylim(0, 115)
    ax.axhline(100, color=GREY, linestyle=":", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
    apply_style(ax, "Grasp Success Rate\n(mean ± std across seeds)", "", "Success Rate (%)")
    fig.tight_layout()
    save(fig, out_dir, "success_rate")


# ── Plot 4: Reward curves over training ──────────────────────────────────────

def plot_reward_curves(tb, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key, title, ylabel in [
        (axes[0], "rewards",        "Total Reward over Training",  "Mean Episode Reward"),
        (axes[1], "shaped_rewards", "Shaped Reward over Training", "Mean Shaped Reward"),
    ]:
        for cond in CONDITIONS:
            if not tb[cond]: continue
            series = [r[key]["values"] for r in tb[cond]
                      if r.get(key) and r[key].get("values")]
            if not series: continue
            xs, mean, std = agg_series(series)
            col = PALETTE[cond]
            ax.plot(xs, mean, color=col, linewidth=1.8, label=CONDITION_LABELS[cond])
            ax.fill_between(xs, mean - std, mean + std, color=col, alpha=0.15)
        apply_style(ax, title, "Iteration", ylabel)
        ax.legend(fontsize=8, framealpha=0.85)

    fig.tight_layout()
    save(fig, out_dir, "reward_curves")


# ── Plot 5: Episode reward breakdown ─────────────────────────────────────────

def plot_reward_breakdown(tb, out_dir):
    reward_keys = [
        ("reaching_object",                   "Reaching"),
        ("lifting_object",                    "Lifting"),
        ("object_goal_tracking",              "Goal Tracking"),
        ("object_goal_tracking_fine_grained", "Goal Tracking (fine)"),
        ("action_rate",                       "Action Rate"),
        ("joint_vel",                         "Joint Velocity"),
    ]

    conds  = [c for c in CONDITIONS if tb[c]]
    labels = [CONDITION_LABELS[c] for c in conds]
    n_cols = 3
    n_rows = (len(reward_keys) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()
    x    = np.arange(len(conds))

    for i, (rkey, rlabel) in enumerate(reward_keys):
        ax = axes[i]
        means, stds = [], []
        for cond in conds:
            vals = []
            for run in tb[cond]:
                series = run.get("episode_rewards", {}).get(rkey, {}).get("values", [])
                if series:
                    vals.append(float(np.mean(series)))
            m, s = agg(vals) if vals else (0.0, 0.0)
            means.append(m); stds.append(s)

        ax.bar(x, means, yerr=stds, color=[PALETTE[c] for c in conds],
               width=0.5, zorder=3, capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": TEXT_COLOUR},
               edgecolor="white", linewidth=1.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=7)
        apply_style(ax, rlabel, "", "Mean Reward")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Episode Reward Breakdown (mean ± std across seeds)",
                 fontsize=13, fontweight="bold", color=TEXT_COLOUR, y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "reward_breakdown")


# ── Plot 6: Pose error ────────────────────────────────────────────────────────

def plot_pose_error(tb, out_dir):
    conds  = [c for c in CONDITIONS if tb[c]]
    labels = [CONDITION_LABELS[c] for c in conds]
    x      = np.arange(len(conds))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for ax, mkey, title, ylabel in [
        (axes[0], "position_error",
         "Position Error\n(mean ± std across seeds)", "Error (m)"),
        (axes[1], "orientation_error",
         "Orientation Error\n(mean ± std across seeds)", "Error (rad)"),
    ]:
        means, stds = [], []
        for cond in conds:
            vals = []
            for run in tb[cond]:
                series = run.get("pose_metrics", {}).get(mkey, {}).get("values", [])
                if series:
                    vals.append(float(np.mean(series)))
            m, s = agg(vals) if vals else (0.0, 0.0)
            means.append(m); stds.append(s)

        bars = ax.bar(x, means, yerr=stds, color=[PALETTE[c] for c in conds],
                      width=0.5, zorder=3, capsize=5,
                      error_kw={"elinewidth": 1.5, "ecolor": TEXT_COLOUR},
                      edgecolor="white", linewidth=1.2)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + max(means) * 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOUR)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
        apply_style(ax, title, "", ylabel)

    fig.tight_layout()
    save(fig, out_dir, "pose_error")


# ── Plot 7: Episode length ────────────────────────────────────────────────────

def plot_episode_length(tb, out_dir):
    conds  = [c for c in CONDITIONS if tb[c]]
    labels = [CONDITION_LABELS[c] for c in conds]

    data = [[v for run in tb[cond]
               for v in run.get("episode_lengths", {}).get("values", [])]
            for cond in conds]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bplot = ax.boxplot(data, patch_artist=True, notch=False,
                       medianprops={"color": TEXT_COLOUR, "linewidth": 1.8},
                       whiskerprops={"linewidth": 1.2},
                       capprops={"linewidth": 1.2},
                       flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for patch, cond in zip(bplot["boxes"], conds):
        patch.set_facecolor(PALETTE[cond]); patch.set_alpha(0.75)

    ax.set_xticklabels(labels, rotation=10, ha="right")
    apply_style(ax, "Episode Length Distribution\n(all seeds pooled)", "", "Steps")
    fig.tight_layout()
    save(fig, out_dir, "episode_length")


# ── Plot 8: Summary grid ──────────────────────────────────────────────────────

def plot_summary(energy, tb, out_dir):
    conds  = [c for c in CONDITIONS if tb[c] or energy[c]]
    labels = [CONDITION_LABELS[c] for c in conds]
    x      = np.arange(len(conds))
    cols   = [PALETTE[c] for c in conds]
    kw_bar = dict(zorder=3, edgecolor="white", linewidth=1.1)
    kw_err = dict(capsize=4, error_kw={"elinewidth": 1.2, "ecolor": TEXT_COLOUR})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Evaluation Summary — All Conditions × Seeds",
                 fontsize=14, fontweight="bold", color=TEXT_COLOUR, y=1.01)

    # (0,0) Success rate
    ax = axes[0, 0]
    means, stds = [], []
    for cond in conds:
        rates = [r["success_rate"] for r in tb[cond] if r.get("success_rate") is not None]
        m, s  = agg(rates) if rates else (0.0, 0.0)
        means.append(m); stds.append(s)
    ax.bar(x, means, yerr=stds, color=cols, width=0.5, **kw_bar, **kw_err)
    ax.set_ylim(0, 115); ax.axhline(100, color=GREY, linestyle=":", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=7)
    apply_style(ax, "Success Rate (%)", "", "%")

    # (0,1) Proxy energy
    ax = axes[0, 1]
    means, stds = [], []
    for cond in conds:
        vals = [v for run in energy[cond] for v in run.get("proxy_energy", [])]
        m, s = agg(vals) if vals else (0.0, 0.0)
        means.append(m); stds.append(s)
    ax.bar(x, means, yerr=stds, color=cols, width=0.5, **kw_bar, **kw_err)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=7)
    apply_style(ax, "Proxy Energy (N·m·steps)", "", "")

    # (0,2) True energy
    ax = axes[0, 2]
    means, stds = [], []
    for cond in conds:
        vals = [v for run in energy[cond] for v in run.get("true_energy", [])]
        m, s = agg(vals) if vals else (0.0, 0.0)
        means.append(m); stds.append(s)
    ax.bar(x, means, yerr=stds, color=cols, width=0.5, **kw_bar, **kw_err)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=7)
    apply_style(ax, "True Energy (J)", "", "")

    # (1,0) Total reward curve
    ax = axes[1, 0]
    for cond in conds:
        series = [r["rewards"]["values"] for r in tb[cond]
                  if r.get("rewards") and r["rewards"].get("values")]
        if not series: continue
        xs, mean, std = agg_series(series)
        ax.plot(xs, mean, color=PALETTE[cond], linewidth=1.5, label=CONDITION_LABELS[cond])
        ax.fill_between(xs, mean - std, mean + std, color=PALETTE[cond], alpha=0.15)
    ax.legend(fontsize=7, framealpha=0.85)
    apply_style(ax, "Total Reward (training)", "Iteration", "Reward")

    # (1,1) Position error
    ax = axes[1, 1]
    means, stds = [], []
    for cond in conds:
        vals = []
        for run in tb[cond]:
            s = run.get("pose_metrics", {}).get("position_error", {}).get("values", [])
            if s: vals.append(float(np.mean(s)))
        m, s = agg(vals) if vals else (0.0, 0.0)
        means.append(m); stds.append(s)
    ax.bar(x, means, yerr=stds, color=cols, width=0.5, **kw_bar, **kw_err)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=7)
    apply_style(ax, "Position Error (m)", "", "m")

    # (1,2) Episode length violin
    ax = axes[1, 2]
    data  = [[v for run in tb[cond]
                for v in run.get("episode_lengths", {}).get("values", [])]
             for cond in conds]
    parts = ax.violinplot(data, positions=range(len(conds)), showmedians=True)
    for pc, cond in zip(parts["bodies"], conds):
        pc.set_facecolor(PALETTE[cond]); pc.set_alpha(0.7); pc.set_edgecolor(TEXT_COLOUR)
    for pn in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[pn].set_edgecolor(TEXT_COLOUR); parts[pn].set_linewidth(1.0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=7)
    apply_style(ax, "Episode Length", "", "Steps")

    handles = [mpatches.Patch(color=PALETTE[c], label=CONDITION_LABELS[c]) for c in conds]
    fig.legend(handles=handles, loc="lower center", ncol=len(conds),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    save(fig, out_dir, "summary")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot thesis metrics grouped by condition, aggregated across seeds."
    )
    parser.add_argument("--data_dir", default="logs/energy_eval",
                        help="Directory containing all JSON files.")
    parser.add_argument("--out_dir",  default="evaluation/thesis_plots",
                        help="Output directory for plots.")
    return parser.parse_args()


def main():
    args           = parse_args()
    out_dir        = Path(os.path.expanduser(args.out_dir))
    energy, tb     = load_all(args.data_dir)

    print(f"\n[INFO] Output directory: {out_dir}\n")

    plot_energy(energy, out_dir)
    plot_energy_distribution(energy, out_dir)
    plot_success_rate(tb, out_dir)
    plot_reward_curves(tb, out_dir)
    plot_reward_breakdown(tb, out_dir)
    plot_pose_error(tb, out_dir)
    plot_episode_length(tb, out_dir)
    plot_summary(energy, tb, out_dir)

    print(f"\n[DONE] All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()