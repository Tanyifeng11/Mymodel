"""从 E23-M 完成的数据生成机制曲线和便于核验的精简 JSON。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from tools.e20_utilization import case_stat
from tools.e15_common import write_json


def curve(ax, steps, stats, label):
    mean = np.array([v["mean"] for v in stats])
    low = np.array([v["ci95"][0] for v in stats])
    high = np.array([v["ci95"][1] for v in stats])
    line, = ax.plot(steps, mean, marker=".", label=label)
    ax.fill_between(steps, low, high, alpha=.15, color=line.get_color())


def attention_summary(out, count):
    # 按 case 聚合 seed、timestep；逐层保留，避免层宽差异被总平均掩盖。
    values, relative = {}, {}
    for case in range(count):
        for seed in (42, 43):
            data = json.loads((out/"common_state"/("c%02d_s%d.json" % (case, seed))).read_text())
            for row in data["common_state"]:
                baseline = row["models"]["E5"]["attention"]
                for arm, model in row["models"].items():
                    for branch, layers in model["attention"].items():
                        for layer, stats in layers.items():
                            if not stats:
                                continue  # 未调用的 processor 不属于本次 active layers。
                            for metric in ("k_norm", "v_norm", "attention_output_rms", "texture_residual_rms"):
                                key = arm, branch, layer, metric
                                values.setdefault(key, []).append((case, stats[metric]))
                                denominator = baseline[branch][layer][metric]
                                relative.setdefault(key, []).append((case, stats[metric]/max(denominator, 1e-8)))
    detail, compact = {}, {}
    for arm in ("E5", "E17_direct", "S0"):
        detail[arm], compact[arm] = {}, {}
        for branch in ("conditional", "unconditional"):
            detail[arm][branch], compact[arm][branch] = {}, {}
            layer_ids = sorted({key[2] for key in values if key[:2] == (arm, branch)}, key=int)
            for layer in layer_ids:
                detail[arm][branch][layer] = {
                    metric: {"value": case_stat(values[arm, branch, layer, metric]),
                             "ratio_to_E5": case_stat(relative[arm, branch, layer, metric])}
                    for metric in ("k_norm", "v_norm", "attention_output_rms", "texture_residual_rms")}
            for metric in ("k_norm", "v_norm", "attention_output_rms", "texture_residual_rms"):
                compact[arm][branch][metric] = {
                    "layer_mean": case_stat([v for layer in layer_ids for v in values[arm, branch, layer, metric]]),
                    "ratio_to_E5": case_stat([v for layer in layer_ids for v in relative[arm, branch, layer, metric]])}
    write_json(out/"attention_statistics.json", {"bootstrap_unit": "case", "per_layer": detail})
    return compact


def decoded_summary(out, count):
    values = {}
    for case in range(count):
        for seed in (42, 43):
            data = json.loads((out/"common_state"/("c%02d_s%d.json" % (case, seed))).read_text())
            for row in data["common_state"]:
                for arm, model in row["models"].items():
                    for kind in ("common_x0", "free_x0"):
                        if kind not in model:
                            continue
                        for metric in ("sketch_iou", "edge_f1", "leakage", "background_white_mae", "boundary_mse_vs_E5_final"):
                            key = arm, kind, str(row["step_index"]), metric
                            values.setdefault(key, []).append((case, model[kind][metric]))
    result = {arm: {} for arm in ("E5", "E17_direct", "S0")}
    for (arm, kind, step, metric), observations in values.items():
        result[arm].setdefault(kind, {}).setdefault(step, {})[metric] = case_stat(observations)
    return result


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
        axes[0, j].set_title("Common-state texture deviation\n"+region)
        axes[1, j].set_title("Free-running latent deviation\n"+region)
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
    cases = json.loads((out/"cases.json").read_text())
    for seed in (42, 43):
        sheet = Image.new("RGB", (8*128, 8*184), "white")
        draw = ImageDraw.Draw(sheet)
        for ref in cases["references"][:8]:
            i = ref["id"]
            sources = [(out/v["path"], v["variant"]) for v in ref["variants"]]
            sources += [(out/arm/("c%02d_s%d_%s.png" % (i, seed, variant)), arm+" "+variant)
                        for arm in ("E5", "E17_direct", "S0") for variant in ("original", "rot90")]
            for j, (path, label) in enumerate(sources):
                image = Image.open(path).convert("RGB")
                image.thumbnail((128, 160))
                sheet.paste(image, (j*128, i*184))
                draw.text((j*128+1, i*184+161), label, fill="black")
        sheet.save(folder/("pilot_seed%d.jpg" % seed), quality=92)
    condition = json.loads((out/"condition_audit.json").read_text())
    attention = attention_summary(out, report["cases"])
    decoded = decoded_summary(out, report["cases"])
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
        summary["groups"][arm]["attention"] = attention[arm]
        summary["groups"][arm]["decoded_x0"] = decoded[arm]
    write_json(out/"decision_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
