"""从 E23-M 完成的数据生成机制曲线和便于核验的精简 JSON。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.e20_utilization import case_stat
from tools.e15_common import write_json


def curve(ax, steps, stats, label):
    mean = np.array([v["mean"] for v in stats])
    low = np.array([v["ci95"][0] for v in stats])
    high = np.array([v["ci95"][1] for v in stats])
    line, = ax.plot(steps, mean, marker=".", label=label)
    ax.fill_between(steps, low, high, alpha=.15, color=line.get_color())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    out = Path(args.root)/"e23_m"
    report = json.loads((out/"report.json").read_text())
    folder = out/"previews"
    folder.mkdir(exist_ok=True)
    groups = report["groups"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for j, region in enumerate(("interior", "boundary", "background")):
        for arm, group in groups.items():
            steps = sorted(map(int, group["by_step"]))
            curve(axes[0, j], steps, [group["by_step"][str(i)]["D"][region] for i in steps], arm)
            all_steps = sorted(map(int, group["F_by_step"]))
            curve(axes[1, j], all_steps, [group["F_by_step"][str(i)][region] for i in all_steps], arm)
        axes[0, j].set_title("Common-state texture-response deviation: "+region)
        axes[1, j].set_title("Free-running latent deviation: "+region)
        for ax in axes[:, j]:
            ax.set_xlabel("Inference step (high -> low noise)")
            ax.set_ylabel("RMS relative to E5")
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(folder/"score_and_trajectory.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for arm, group in groups.items():
        steps = sorted(map(int, group["by_step"]))
        for ax, metric in zip(axes, ("I", "rotation_rms")):
            curve(ax, steps, [group["by_step"][str(i)][metric]["background" if metric == "I" else "interior"] for i in steps], arm)
            ax.set_xlabel("Inference step")
            ax.grid(alpha=.2)
    axes[0].set_title("Teacher-forced one-step background injection")
    axes[1].set_title("Common-state interior rotation response")
    axes[0].legend()
    fig.savefig(folder/"injection_and_rotation.png", dpi=160)
    plt.close(fig)
    condition = json.loads((out/"condition_audit.json").read_text())
    summary = {"cases": report["cases"], "first_failure_candidate": report["first_failure_candidate"],
               "H1_supported": report["H1_supported"], "H2_candidate": report["H2_candidate"],
               "time_interval_difference": report["time_interval_difference"],
               "E17_minus_S0_D_background": report["E17_minus_S0_D_background"],
               "E17_I_background_by_window": report["E17_I_background_by_window"],
               "freeze_pass": report.get("freeze_pass"), "same_initial_latent": report.get("same_initial_latent"),
               "groups": {}, "scope": report["scope"], "windows": report.get("windows")}
    for arm, group in groups.items():
        subset = [v for v in condition["records"] if v["arm"] == arm]
        summary["groups"][arm] = {k: group[k] for k in ("D", "conditional_D", "unconditional_D", "LR", "I", "A", "rotation_rms", "D_over_E5_score_rms", "D_over_E5_texture_rms", "final")}
        summary["groups"][arm]["condition"] = {k: case_stat([(v["case"], v[k]) for v in subset]) for k in ("rms", "pairwise_cosine", "effective_rank", "empirical_E5_affine_OOR")}
    write_json(out/"decision_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
