# =============================================================================
# Extract Metrics from TensorBoard Events File
# Thesis: Energy-Aware Reward Shaping for Robotic Grasping
# Author: Su Acar, Tilburg University, 2026
#
# Reads a TensorBoard events file and extracts training metrics into a JSON
# file that can be used by plot_metrics.py or inspected directly.
#
# Extracted metrics:
#   - Episode rewards (reaching, lifting, goal tracking, action_rate, joint_vel)
#   - Episode lengths
#   - Episode terminations (time_out, object_dropping)
#   - Position / orientation error
#   - Training losses and entropy
#   - Total / shaped rewards
#
# Usage:
#   # Auto-discover events file from label (recommended):
#   python extract_tb_metrics.py --label baseline_seed42
#
#   # With explicit logs root if your folder isn't the default:
#   python extract_tb_metrics.py --label baseline_seed42 --logs_root C:\Users\user\Desktop\IsaacLab-energypenalty\logs\rl_games
#
#   # Override output dir:
#   python extract_tb_metrics.py --label baseline_seed42 --out_dir logs\energy_eval
#
# Folder structure assumed:
#   <logs_root>/<taskname><seed>/summaries/events.out.tfevents.*
#   e.g. logs/rl_games/franka_lift/baseline42/summaries/events.out.tfevents.*
#
#   The label "baseline_seed42" maps to folder name "baseline42"
#   (i.e. underscores and the word "seed" are stripped for matching).
# =============================================================================

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("[ERROR] tensorboard not installed. Run: pip install tensorboard")

import numpy as np


# ── Tags we care about ────────────────────────────────────────────────────────

EPISODE_REWARD_TAGS = [
    "Episode/Episode_Reward/reaching_object",
    "Episode/Episode_Reward/lifting_object",
    "Episode/Episode_Reward/object_goal_tracking",
    "Episode/Episode_Reward/object_goal_tracking_fine_grained",
    "Episode/Episode_Reward/action_rate",
    "Episode/Episode_Reward/joint_vel",
]

METRIC_TAGS = [
    "Episode/Metrics/object_pose/position_error",
    "Episode/Metrics/object_pose/orientation_error",
]

TERMINATION_TAGS = [
    "Episode/Episode_Termination/time_out",
    "Episode/Episode_Termination/object_dropping",
]

LENGTH_TAGS = [
    "episode_lengths/iter",
]

REWARD_TAGS = [
    "rewards/iter",
    "shaped_rewards/iter",
]

LOSS_TAGS = [
    "losses/a_loss",
    "losses/c_loss",
    "losses/entropy",
]


def extract_tag(ea, tag: str) -> dict:
    """Return {steps: [...], values: [...]} for a scalar tag, or empty if missing."""
    try:
        events = ea.Scalars(tag)
        return {
            "steps":  [e.step  for e in events],
            "values": [e.value for e in events],
        }
    except KeyError:
        return {"steps": [], "values": []}


def summarise(series: dict) -> dict:
    """Add mean/std/min/max to a series dict."""
    vals = series["values"]
    if not vals:
        return {**series, "mean": None, "std": None, "min": None, "max": None}
    arr = np.array(vals, dtype=float)
    return {
        **series,
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


def infer_success_rate(ea) -> float | None:
    """
    Estimate grasp success rate from lifting_object reward.
    A step where lifting_object > 0 is treated as a successful grasp.
    Returns percentage, or None if tag unavailable.
    """
    try:
        events = ea.Scalars("Episode/Episode_Reward/lifting_object")
        vals   = np.array([e.value for e in events])
        return float((vals > 0).mean() * 100)
    except KeyError:
        return None


DEFAULT_LOGS_ROOT = r"C:\Users\user\Desktop\IsaacLab-energypenalty\logs\rl_games\franka_lift"


def label_to_folder_name(label: str) -> str:
    """
    Convert a label like 'baseline_seed42' or 'config2_seed7' to the
    folder name used on disk, e.g. 'baseline42' or 'config27'.
    Rules:
      - Remove the word 'seed' (case-insensitive)
      - Remove underscores
    """
    import re
    folder = re.sub(r"_?seed_?", "", label, flags=re.IGNORECASE)
    folder = folder.replace("_", "")
    return folder


def find_events_file(logs_root: str, label: str) -> Path:
    """
    Search <logs_root> recursively for a folder whose name matches the label
    (after normalisation), then find the events file inside its summaries/.
    Falls back to a direct recursive search for any tfevents file under a
    folder that contains the normalised label name.
    """
    root        = Path(os.path.expanduser(logs_root))
    folder_name = label_to_folder_name(label)

    print(f"[INFO] Searching for folder matching '{folder_name}' under: {root}")

    # Walk all subdirs looking for a name match
    candidates = []
    for dirpath in root.rglob("*"):
        if dirpath.is_dir() and dirpath.name.lower() == folder_name.lower():
            candidates.append(dirpath)

    if not candidates:
        # Looser match: folder name contains the normalised label
        for dirpath in root.rglob("*"):
            if dirpath.is_dir() and folder_name.lower() in dirpath.name.lower():
                candidates.append(dirpath)

    if not candidates:
        sys.exit(
            f"[ERROR] Could not find a folder matching '{folder_name}' under {root}.\n"
            f"        Check --logs_root or rename your folder to match the label."
        )

    if len(candidates) > 1:
        print(f"[WARN] Multiple matching folders found, using the first:")
        for c in candidates:
            print(f"  {c}")

    run_dir = candidates[0]
    print(f"[INFO] Found run directory: {run_dir}")

    # Look for tfevents file in summaries/ subdir first, then anywhere under run_dir
    for search_dir in [run_dir / "summaries", run_dir]:
        for f in search_dir.rglob("events.out.tfevents*"):
            print(f"[INFO] Found events file: {f}")
            return f

    sys.exit(f"[ERROR] No events.out.tfevents file found under: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract metrics from a TensorBoard events file into JSON."
    )
    parser.add_argument("--label", required=True,
                        help="Config label, e.g. baseline_seed42 or config2_seed7.")
    parser.add_argument("--logs_root", default=DEFAULT_LOGS_ROOT,
                        help="Root folder containing all run subdirectories.")
    parser.add_argument("--out_dir", default="logs/energy_eval",
                        help="Directory to save the output JSON.")
    parser.add_argument("--list_tags", action="store_true",
                        help="Print all available tags and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    events_path = find_events_file(args.logs_root, args.label)
    if not events_path.is_file():
        sys.exit(f"[ERROR] Events file not found: {events_path}")

    print(f"[INFO] Loading events file: {events_path}")
    ea = EventAccumulator(str(events_path))
    ea.Reload()
    print("[INFO] Done loading.")

    if args.list_tags:
        print("\nAll scalar tags:")
        for t in sorted(ea.Tags().get("scalars", [])):
            print(f"  {t}")
        return

    out: dict = {"label": args.label}

    # ── Episode rewards ───────────────────────────────────────────────────────
    out["episode_rewards"] = {}
    for tag in EPISODE_REWARD_TAGS:
        short_key = tag.split("/")[-1]
        out["episode_rewards"][short_key] = summarise(extract_tag(ea, tag))

    # ── Pose metrics ──────────────────────────────────────────────────────────
    out["pose_metrics"] = {}
    for tag in METRIC_TAGS:
        short_key = tag.split("/")[-1]
        out["pose_metrics"][short_key] = summarise(extract_tag(ea, tag))

    # ── Terminations ──────────────────────────────────────────────────────────
    out["terminations"] = {}
    for tag in TERMINATION_TAGS:
        short_key = tag.split("/")[-1]
        out["terminations"][short_key] = summarise(extract_tag(ea, tag))

    # ── Episode lengths ───────────────────────────────────────────────────────
    out["episode_lengths"] = summarise(extract_tag(ea, "episode_lengths/iter"))

    # ── Total / shaped rewards ────────────────────────────────────────────────
    out["rewards"]         = summarise(extract_tag(ea, "rewards/iter"))
    out["shaped_rewards"]  = summarise(extract_tag(ea, "shaped_rewards/iter"))

    # ── Training losses ───────────────────────────────────────────────────────
    out["losses"] = {}
    for tag in LOSS_TAGS:
        short_key = tag.split("/")[-1]
        out["losses"][short_key] = summarise(extract_tag(ea, tag))

    # ── Inferred success rate ─────────────────────────────────────────────────
    out["success_rate"] = infer_success_rate(ea)

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Label:              {out['label']}")
    if out["success_rate"] is not None:
        print(f"  Inferred success:   {out['success_rate']:.1f}%")
    for k, v in out["episode_rewards"].items():
        if v["mean"] is not None:
            print(f"  Reward/{k:<30s} mean={v['mean']:.4f}")
    for k, v in out["pose_metrics"].items():
        if v["mean"] is not None:
            print(f"  Metric/{k:<30s} mean={v['mean']:.4f}")
    ep_len = out["episode_lengths"]
    if ep_len["mean"] is not None:
        print(f"  Episode length:     {ep_len['mean']:.1f} ± {ep_len['std']:.1f} steps")
    print(f"{'='*55}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"logs/energy_eval/{args.label}_metrics.json"

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved to: {out_path}")


if __name__ == "__main__":
    main()