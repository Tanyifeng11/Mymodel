"""E23-M2：冻结权重下验证 CFG 分支关系与 texture token collapse。"""

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools import e23_mechanism as old
from tools.e15_common import write_json
from tools.e20_utilization import case_stat, load_pipeline
from tools.e22_4_generation import paired
from tools.e22_4_genealogy import file_hash


ARMS = ("E5", "E17_direct", "S0")
GROUPS = ("G0", "G1", "G2", "G3")
TOKEN_GROUPS = ("E5", "E5_collapse", "E17_direct", "E17_norm", "E17_self_decollapse", "S0")


def cosine(a, b, mask):
    a, b = a.float(), b.float()
    numerator = (a * b * mask).sum()
    denominator = ((a.square() * mask).sum() * (b.square() * mask).sum()).sqrt()
    return float(numerator / denominator.clamp_min(1e-12))


def region_cosine(a, b, masks):
    return {name: cosine(a, b, mask) for name, mask in masks.items()}


def token_rms(tokens):
    return tokens.float().square().mean().sqrt()


def token_metrics(tokens):
    x = tokens[0].float()
    center = x - x.mean(0)
    normalized = F.normalize(x, dim=-1)
    indices = torch.triu_indices(len(x), len(x), 1)
    singular = torch.linalg.svdvals(center)
    weights = singular.square() / singular.square().sum().clamp_min(1e-12)
    return {"rms": float(token_rms(x)), "centered_rms": float(token_rms(center)),
            "pairwise_cosine": float((normalized @ normalized.T)[indices[0], indices[1]].mean()),
            "effective_rank": float(torch.exp(-(weights[weights > 0] * weights[weights > 0].log()).sum()))}


def change_positive(pair, positive):
    assert positive.shape == pair[0].shape
    return positive.to(pair[0].dtype), pair[1]


def token_interventions(e5, e17):
    a, b = e5[0].float(), e17[0].float()
    e5_rms, e17_rms = token_rms(a), token_rms(b)
    mean_a = a.mean(1, keepdim=True)
    collapse = mean_a.expand_as(a) * (e5_rms / token_rms(mean_a).clamp_min(1e-12))
    norm = b * (e5_rms / e17_rms.clamp_min(1e-12))
    mean_b, centered_b = b.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    target_center = token_rms(a - mean_a)
    # 总 RMS 和中心化 RMS 同时匹配；若目标大于 E17 总 RMS，实验协议不成立。
    assert target_center < e17_rms
    mean_scale = (e17_rms.square() - target_center.square()).sqrt() / token_rms(mean_b).clamp_min(1e-12)
    spread_scale = target_center / token_rms(centered_b).clamp_min(1e-12)
    assert spread_scale / mean_scale > 1, "E17 centered spread is not collapsed relative to E5"
    decollapse = mean_b * mean_scale + centered_b * spread_scale
    assert abs(float(token_rms(decollapse) / e17_rms) - 1) < 1e-4
    return ({"E5": e5, "E5_collapse": change_positive(e5, collapse),
            "E17_direct": e17, "E17_norm": change_positive(e17, norm),
            "E17_self_decollapse": change_positive(e17, decollapse)},
            {"mean_scale": float(mean_scale), "spread_scale": float(spread_scale),
             "lambda_before_total_rms_calibration": float(spread_scale / mean_scale)})


def load_inputs(root, out):
    source = root / "output_eval/e23_m"
    assert (source / "cases.json").exists() and (source / "tokens.pt").exists()
    cases = json.loads((source / "cases.json").read_text())
    assert len(cases["references"]) == 32
    shutil.copytree(source / "inputs", out / "inputs", dirs_exist_ok=True)
    write_json(out / "cases.json", cases)
    banks = torch.load(source / "tokens.pt", map_location="cpu", weights_only=False)
    assert set(banks) == set(ARMS)
    return source, cases, banks


def bank_for(banks, group, count):
    result = {}
    for ref in range(count):
        for variant in ("original", "rot90"):
            e5, e17 = banks["E5"][ref, variant], banks["E17_direct"][ref, variant]
            if group in GROUPS:
                cond = e17[0] if group in ("G1", "G3") else e5[0]
                uncond = e17[1] if group in ("G2", "G3") else e5[1]
                result[ref, variant] = cond, uncond
            else:
                result[ref, variant] = token_interventions(e5, e17)[0][group]
    return result


def masked_stats(delta, masks):
    values = old.regional(delta, masks)
    values["GDR_bg"] = values["background"] / max(values["interior"], 1e-8)
    return values


@torch.inference_mode()
def common_state(pipe, source, cases, banks, out, start, stop, include_decollapse):
    target = out / "common_state"
    target.mkdir(exist_ok=True)
    pipe.scheduler.set_timesteps(50, device=pipe.device)
    probe = old.AttentionStats(pipe)
    try:
        for ref, sk in old.selected(cases, start, stop):
            case = ref["id"]
            for seed in old.gen.SEEDS:
                path = target / f"c{case:02d}_s{seed}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    if include_decollapse and "E17_self_decollapse" not in saved["rows"][0]["tokens"]:
                        pass
                    else:
                        continue
                trace = torch.load(source / "E5" / f"c{case:02d}_s{seed}_original.pt",
                                   map_location="cpu", weights_only=False)
                context = old.tree(trace["context"], pipe.device)
                masks = old.latent_regions(out, sk, trace["latent"].shape[-2:], pipe.device)
                rowset = []
                for step in old.POSITIONS:
                    z = trace["latent"][step].to(pipe.device)
                    timestep = pipe.scheduler.timesteps[step]
                    prediction = {}
                    intervention_info = {}
                    for variant in ("original", "rot90"):
                        pairs = {}
                        e5, e17 = banks["E5"][case, variant], banks["E17_direct"][case, variant]
                        altered, calibration = token_interventions(e5, e17)
                        intervention_info[variant] = calibration
                        for arm in ARMS:
                            pairs[arm] = banks[arm][case, variant]
                        for group in TOKEN_GROUPS:
                            if group != "E17_self_decollapse" or include_decollapse:
                                pairs[group] = banks["S0"][case, variant] if group == "S0" else altered[group]
                        prediction[variant] = {}
                        for group, pair in pairs.items():
                            prediction[variant][group] = old.forward(pipe, z, timestep, context,
                                old.tree(pair, pipe.device), probe=probe if group in TOKEN_GROUPS else None)
                    baseline = prediction["original"]["E5"]
                    replay_error = old.rms(baseline[0] - trace["epsilon"][step].to(pipe.device))
                    assert replay_error < 1e-5, f"E5 replay differs at case={case} seed={seed} step={step}: {replay_error}"
                    row = {"case": case, "seed": seed, "step_index": step,
                           "timestep": int(timestep), "arms": {}, "tokens": {},
                           "calibration": intervention_info}
                    for arm in ARMS:
                        normal = prediction["original"][arm]
                        rotated = prediction["rot90"][arm]
                        guidance = normal[1] - normal[2]
                        delta_c = normal[1] - baseline[1]
                        delta_u = normal[2] - baseline[2]
                        delta_g = guidance - (baseline[1] - baseline[2])
                        row["arms"][arm] = {
                            "delta_c": old.regional(delta_c, masks),
                            "delta_u": old.regional(delta_u, masks),
                            "delta_g": masked_stats(delta_g, masks),
                            "guidance_cosine_vs_E5": region_cosine(guidance, baseline[1] - baseline[2], masks),
                            "branch_delta_cosine": region_cosine(delta_c, delta_u, masks) if arm != "E5" else None,
                            "rotation_guidance": masked_stats((rotated[1] - rotated[2]) - guidance, masks)}
                    for group in TOKEN_GROUPS:
                        if group == "E17_self_decollapse" and not include_decollapse:
                            continue
                        normal = prediction["original"][group]
                        rotated = prediction["rot90"][group]
                        positive = (banks["S0"][case, "original"][0] if group == "S0" else
                            token_interventions(banks["E5"][case, "original"],
                            banks["E17_direct"][case, "original"])[0][group][0])
                        row["tokens"][group] = {
                            "token": token_metrics(positive),
                            "D": old.regional(normal[0] - baseline[0], masks),
                            "guidance_deviation": masked_stats((normal[1] - normal[2]) -
                                (baseline[1] - baseline[2]), masks),
                            "interior_rotation_response": old.regional(rotated[0] - normal[0], masks)["interior"],
                            "attention": normal[3]}
                    rowset.append(row)
                write_json(path, {"case": case, "seed": seed, "rows": rowset})
                print("[e23-m2-common]", case, seed, "decollapse", include_decollapse, flush=True)
    finally:
        probe.close()


def records(out, count):
    return [row for case in range(count) for seed in old.gen.SEEDS
            for row in json.loads((out / "common_state" / f"c{case:02d}_s{seed}.json").read_text())["rows"]]


def difference(rows, arm_a, arm_b, metric, region):
    return case_stat([(row["case"], row["tokens"][arm_a][metric][region] -
                       row["tokens"][arm_b][metric][region]) for row in rows])


def summarize_common(rows):
    summary = {"A1": {}, "B": {}}
    for arm in ARMS:
        summary["A1"][arm] = {metric: {region: case_stat([(r["case"], r["arms"][arm][metric][region])
                                         for r in rows]) for region in old.REGIONS}
                                for metric in ("delta_c", "delta_u", "delta_g", "guidance_cosine_vs_E5",
                                               "rotation_guidance")}
        summary["A1"][arm]["GDR_bg"] = case_stat([(r["case"], r["arms"][arm]["delta_g"]["GDR_bg"]) for r in rows])
        if arm != "E5":
            summary["A1"][arm]["branch_delta_cosine"] = {
                region: case_stat([(r["case"], r["arms"][arm]["branch_delta_cosine"][region]) for r in rows])
                for region in old.REGIONS}
        summary["A1"][arm]["rotation_bg_over_in"] = case_stat([
            (r["case"], r["arms"][arm]["rotation_guidance"]["GDR_bg"]) for r in rows])
    for group in rows[0]["tokens"]:
        summary["B"][group] = {
            "token": {key: case_stat([(r["case"], r["tokens"][group]["token"][key]) for r in rows])
                      for key in ("rms", "centered_rms", "pairwise_cosine", "effective_rank")},
            "D": {region: case_stat([(r["case"], r["tokens"][group]["D"][region]) for r in rows])
                  for region in old.REGIONS},
            "GDR_bg": case_stat([(r["case"], r["tokens"][group]["guidance_deviation"]["GDR_bg"])
                                 for r in rows]),
            "rotation_interior": case_stat([(r["case"], r["tokens"][group]["interior_rotation_response"])
                                            for r in rows]),
            "attention": {branch: {key: case_stat([(r["case"], layer[key]) for r in rows
                for layer in r["tokens"][group]["attention"][branch].values() if key in layer])
                for key in ("k_norm", "v_norm", "attention_output_rms")}
                for branch in ("conditional", "unconditional")}}
    b1 = summary["B"]["E5_collapse"]["D"]["background"]
    e17 = summary["B"]["E17_direct"]["D"]["background"]
    summary["B"]["E17_norm_minus_E17_D_bg"] = difference(rows, "E17_norm", "E17_direct", "D", "background")
    b1_gate = b1["ci95"][0] > .1 * e17["mean"]
    summary["gates"] = {"B1_positive": bool(b1_gate),
                        "B1_definition": "case-bootstrap lower CI of E5-collapse D_bg exceeds 10% of E17 D_bg mean"}
    if "E17_self_decollapse" in summary["B"]:
        b3 = difference(rows, "E17_self_decollapse", "E17_direct", "D", "background")
        summary["B"]["B3_minus_E17_D_bg"] = b3
        summary["gates"]["B3_positive"] = bool(b3["ci95"][1] < -.1 * e17["mean"])
    return summary


def existing_baseline(source, out, group, ref, seed, variant):
    arm = "E5" if group == "G0" else "E17_direct"
    name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
    target = out / group
    target.mkdir(exist_ok=True)
    metadata = target / (name + ".json")
    if metadata.exists():
        return json.loads(metadata.read_text())
    original = source / arm / (name + ".json")
    row = json.loads(original.read_text())
    shutil.copy2(source / arm / (name + ".png"), target / (name + ".png"))
    row["arm"], row["path"] = group, f"{group}/{name}.png"
    write_json(metadata, row)
    return row


@torch.inference_mode()
def generate_groups(pipe, source, cases, banks, out, start, stop, groups, scale=7.):
    width = json.loads((source / "stage0_audit.json").read_text())["E5"]["resolution"][0]
    height = json.loads((source / "stage0_audit.json").read_text())["E5"]["resolution"][1]
    result = {}
    for group in groups:
        bank = bank_for(banks, group if group in GROUPS + TOKEN_GROUPS else group.split("_s")[0], stop)
        rows = []
        for ref, sk in old.selected(cases, start, stop):
            for seed in old.gen.SEEDS:
                for variant in ref["variants"]:
                    if group in ("G0", "G3") and start > 0:
                        row = existing_baseline(source, out, group, ref, seed, variant)
                    else:
                        row = old.generate(pipe, group, ref, sk, seed, variant, bank, out,
                                           width, height, trace=False, guidance_scale=scale)
                    if group in ("G0", "G3") and start == 0:
                        original_arm = "E5" if group == "G0" else "E17_direct"
                        name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}.png"
                        assert np.array_equal(np.asarray(Image.open(out / group / name)),
                                              np.asarray(Image.open(source / original_arm / name)))
                    rows.append(row)
        result[group] = rows
    return result


def final_summary(out, groups, count):
    result = {}
    for group in groups:
        rows = [json.loads(path.read_text()) for path in (out / group).glob("c*.json")
                if int(path.name[1:3]) < count]
        assert len(rows) == count * len(old.gen.SEEDS) * 2, (group, len(rows))
        pairs = paired(rows)
        result[group] = {key: case_stat([(r["case"], r[key]) for r in rows])
                         for key in ("sketch_iou", "edge_f1", "leakage")}
        result[group]["flip_accuracy"] = case_stat([(r["reference"], r["flip_correct"]) for r in pairs])
        result[group]["orientation_error"] = case_stat([(r["case"], r["direction"]["error"]) for r in rows])
        result[group]["orientation_valid_rate"] = case_stat([(r["case"], float(r["direction"]["valid"])) for r in rows])
        valid = [(r["case"], r["direction"]["error"]) for r in rows if r["direction"]["valid"]]
        result[group]["orientation_error_valid_only"] = case_stat(valid) if valid else None
    if all(group in result for group in GROUPS):
        indexed = {group: {(r["case"], r["seed"], r["variant"]): r for r in
                           [json.loads(path.read_text()) for path in (out / group).glob("c*.json")
                            if int(path.name[1:3]) < count]} for group in GROUPS}
        result["branch_swap_contrasts"] = {}
        for metric in ("leakage", "sketch_iou", "edge_f1"):
            result["branch_swap_contrasts"][metric] = {}
            for group in GROUPS[1:]:
                result["branch_swap_contrasts"][metric][group + "_minus_G0"] = case_stat([
                    (key[0], row[metric] - indexed["G0"][key][metric])
                    for key, row in indexed[group].items()])
            result["branch_swap_contrasts"][metric]["G3_interaction"] = case_stat([
                (key[0], indexed["G3"][key][metric] - indexed["G1"][key][metric] -
                 indexed["G2"][key][metric] + indexed["G0"][key][metric])
                for key in indexed["G0"]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "output_eval/e23_m2"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    source, cases, banks = load_inputs(root, out)
    pipe, _, _, modules, width, height = load_pipeline(root, "cuda:0")
    pipe.set_progress_bar_config(disable=True)
    frozen = old.module_hashes(modules)
    assert pipe.scheduler.__class__.__name__ == "DDIMScheduler"
    assert pipe.scheduler.config.prediction_type == "epsilon"
    write_json(out / "protocol.json", {"source": str(source), "source_tokens_sha256": file_hash(source / "tokens.pt"),
        "training_steps": 0, "steps": 50, "native_cfg": 7, "mid_cfg": 4, "seeds": list(old.gen.SEEDS),
        "pilot_cases": 8, "confirmation_cases": 32, "positions": old.POSITIONS,
        "token_interventions": "positive texture tokens only; native negative branch unchanged",
        "bootstrap_unit": "reference case; seeds, steps and rotation variants averaged within case",
        "source_frozen_module_hashes": frozen})

    for start, stop in ((0, 8), (8, 32)):
        if start and not report["gates"]["confirm"]:
            break
        common_state(pipe, source, cases, banks, out, start, stop, include_decollapse=False)
        a3 = generate_groups(pipe, source, cases, banks, out, start, stop, GROUPS)
        common = summarize_common(records(out, stop))
        if common["gates"]["B1_positive"]:
            common_state(pipe, source, cases, banks, out, 0, stop, include_decollapse=True)
            common = summarize_common(records(out, stop))
        a3_final = final_summary(out, GROUPS, stop)
        cfg_signal = a3_final["branch_swap_contrasts"]["leakage"]["G3_minus_G0"]["ci95"][0] > 0
        cfg_signal = cfg_signal or common["A1"]["E17_direct"]["delta_g"]["background"]["ci95"][0] > .1
        if cfg_signal:
            # s=7 已由 G0/G3 给出，只补 s=1 与 s=4。
            for scale in (1, 4):
                for arm, label in (("G0", "E5"), ("G3", "E17_direct")):
                    group = f"{arm}_s{scale}"
                    generate_groups(pipe, source, cases, banks, out, start, stop, (group,), float(scale))
        token_full = []
        if common["gates"]["B1_positive"]:
            token_full.append("E5_collapse")
        if common["gates"].get("B3_positive"):
            token_full.append("E17_self_decollapse")
        if token_full:
            generate_groups(pipe, source, cases, banks, out, start, stop, token_full)
        groups = list(GROUPS)
        if cfg_signal:
            groups += [f"{arm}_s{scale}" for scale in (1, 4) for arm in ("G0", "G3")]
        groups += token_full
        final = final_summary(out, groups, stop)
        report = {"cases": stop, "training_steps": 0, "common_state": common, "final": final,
                  "gates": {"CFG_sweep": bool(cfg_signal), "B1_positive": common["gates"]["B1_positive"],
                            "B3_positive": common["gates"].get("B3_positive", False),
                            "confirm": bool(cfg_signal or common["gates"]["B1_positive"] or
                                            common["gates"].get("B3_positive", False))},
                  "interpretation_limit": "score sensitivity and leakage do not establish correct pattern control"}
        assert old.module_hashes(modules) == frozen, "Model weights changed"
        report["frozen_pass"] = True
        write_json(out / ("pilot_report.json" if stop == 8 else "confirmation_report.json"), report)
        write_json(out / "report.json", report)
        print("[e23-m2-gates]", stop, report["gates"], flush=True)
    del pipe, modules
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
