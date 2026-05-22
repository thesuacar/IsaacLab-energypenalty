# =============================================================================
# Generates summary statistics (mean, std, min, max) from evaluation results
# for each metric and condition, and saves to a JSON file.
# =============================================================================
'''
Thesis: Energy-Aware Reward Shaping for Robotic Grasping
Author: Su Acar, Tilburg University, 2026

'''
import json
import numpy as np

# ── Configure these ────────────────────────────────────────────────────────────
CONDITIONS = {
    "Baseline": [
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline42_energyproxy.json",       # from evaluate_energy.py
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline42_tb_metrics.json", # from extract_tb_metrics.py
        },
        # repeat for seeds 2-5
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline123_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline123_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline456_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline456_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline789_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline789_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline999_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\baseline999_tb_metrics.json",
        }
    ],
    "Joint Effort": [
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort42_energyproxy.json",       # from evaluate_energy.py
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort42_tb_metrics.json", # from extract_tb_metrics.py
        },
        # repeat for seeds 2-5
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort123_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort123_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort456_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort456_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort789_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort789_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort999_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffort999_tb_metrics.json",
        }
    ],
    "Joint Effort + Jerk": [
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk42_energyproxy.json",       # from evaluate_energy.py
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk42_tb_metrics.json", # from extract_tb_metrics.py
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk123_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk123_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk456_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk456_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk789_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk789_tb_metrics.json",
        },
        {
            "energy": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk999_energyproxy.json",
            "metrics": "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\jointeffortjerk999_tb_metrics.json",
        }
    ],
}

OUTPUT_PATH = "C:\\Users\\user\\Desktop\\IsaacLab-energypenalty\\logs\\energy_eval\\summary_statistics.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_metrics(energy_path, metrics_path):
    with open(energy_path) as f:
        energy_data = json.load(f)
    with open(metrics_path) as f:
        metrics_data = json.load(f)

    return {
        "success_rate":   metrics_data.get("success_rate"),
        "proxy_energy":   float(np.mean(energy_data.get("proxy_energy", []))),
        "true_energy":    float(np.mean(energy_data.get("true_energy",  []))),
        "episode_length": metrics_data["episode_lengths"].get("mean"),
        "position_error": metrics_data["pose_metrics"]["position_error"].get("mean"),
    }

def summarise(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std":  round(float(arr.std()),  4),
        "min":  round(float(arr.min()),  4),
        "max":  round(float(arr.max()),  4),
        "n":    len(arr),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

metrics_keys = ["success_rate", "proxy_energy", "true_energy",
                "episode_length", "position_error"]

all_data = {}
for condition, seeds in CONDITIONS.items():
    all_data[condition] = [
        extract_metrics(s["energy"], s["metrics"]) for s in seeds
    ]

# Build summary
summary = {}
for condition in CONDITIONS:
    summary[condition] = {}
    for key in metrics_keys:
        values = [d[key] for d in all_data[condition]]
        summary[condition][key] = summarise(values)

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[INFO] Saved to: {OUTPUT_PATH}")

# Print table
print(f"\n{'Metric':<25} {'Baseline':<30} {'Joint Effort':<30} {'Joint Effort + Jerk'}")
print("-" * 110)
for key in metrics_keys:
    row = f"{key:<25}"
    for condition in CONDITIONS:
        s = summary[condition][key]
        row += f" {s['mean']:.2f} ± {s['std']:.2f}{'':>10}"
    print(row)
print()