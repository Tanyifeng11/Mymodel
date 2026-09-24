"""E15 诊断各步实现：D1 参考轨迹、D2 residual 热图、D3 离线区域、D4 分组消融。"""

import json
from pathlib import Path

import numpy as np

from tools.e15_common import (BOTTLENECK_RATIO, CONDITIONS, CONTRASTS, FULL_STEPS,
                              GROUP_RANGES, REFERENCE_LAYERS, REGIONS, LayerCapture,
                              compact_table, flatten_representation, format_heatmap,
                              gram_statistics, normalized_response, relative_change,
                              summary_stats, write_json)


def texture_processors(unet):
    """按 UNet 处理顺序返回 16 个 texture cross-attention processor。"""
    return [proc for proc in unet.attn_processors.values() if hasattr(proc, "to_k_ip")]


def reference_batch(condition, index, ctx):
    """返回 (clip_tensor, cnn_tensor)；zero_image 为全零参考输入。"""
    import torch
    from PIL import Image
    from tools.e14_pattern_probe import training_preprocess

    dataset = ctx["dataset"]
    if condition == "zero_image":
        batch = dataset[index]
        return (torch.zeros_like(batch["clip_texture_image"]),
                torch.zeros_like(batch["texture_image"]))
    if condition == "rot90":
        row = dataset.data[index]
        path = Path(ctx["args"]["data_root"]) / row.get("texture", row.get("color"))
        with Image.open(path) as image:
            rotation = image.convert("RGB").transpose(Image.ROTATE_90)
        cnn, clip = training_preprocess(rotation, dataset.clip_image_processor,
                                        ctx["height"], ctx["width"])
        return clip, cnn
    donor = index
    if condition == "color_nearest":
        donor = ctx["colors"][index]["index"]
    elif condition == "random":
        donor = ctx["pairs"][index][0]
    batch = dataset[donor]
    return batch["clip_texture_image"], batch["texture_image"]


def condition_tokens(ctx, index):
    """113817 口径的五个条件 tokens；zero_tokens 为真实全零 token。"""
    import torch

    bank = ctx["token_bank"]
    tokens = {
        "matched": bank["matched"][index],
        "rot90": bank["rot90"][index],
        "color_nearest": bank["matched"][ctx["colors"][index]["index"]],
        "random": bank["matched"][ctx["pairs"][index][0]],
        "zero_tokens": torch.zeros_like(bank["matched"][index]),
    }
    for name, value in tokens.items():
        if not torch.isfinite(value).all():
            raise ValueError("条件 tokens 非有限：%s" % name)
    return tokens


def build_token_bank(ctx):
    """对全部样本预计算 matched 与 rot90 的 16 个 texture tokens。"""
    import torch

    bank = {"matched": {}, "rot90": {}}
    with torch.inference_mode():
        for index in ctx["indices"]:
            for condition in ("matched", "rot90"):
                clip_input, cnn_input = reference_batch(condition, index, ctx)
                visual = ctx["vision"](clip_input[None].to(ctx["device"], ctx["dtype"]),
                                       output_hidden_states=True)
                bank[condition][index] = ctx["model"].get_texture_condition_tokens(
                    visual, cnn_input[None].to(ctx["device"], ctx["dtype"])).detach()
    ctx["token_bank"] = bank
    return bank


def d1_reference_trace(ctx):
    """参考信息衰减曲线：分层 normalized response、谱指标与来源 attention。"""
    import torch

    bf = ctx["bf"]
    capture = LayerCapture(bf)
    bf.probe_capture_resampler_attention = True
    per_layer = {layer: {condition: [] for condition in CONDITIONS} for layer in REFERENCE_LAYERS}
    attention_rows = []
    source_lengths = None
    source_names = ["clip", "cnn1", "cnn2", "cnn3", "cnn4"]
    try:
        with torch.inference_mode():
            for index in ctx["indices"]:
                for condition in CONDITIONS:
                    clip_input, cnn_input = reference_batch(condition, index, ctx)
                    capture.reset()
                    visual = ctx["vision"](clip_input[None].to(ctx["device"], ctx["dtype"]),
                                           output_hidden_states=True)
                    tokens = ctx["model"].get_texture_condition_tokens(
                        visual, cnn_input[None].to(ctx["device"], ctx["dtype"]))
                    if not torch.isfinite(tokens).all():
                        raise ValueError("参考 tokens 非有限")
                    captured = dict(capture.current)
                    captured["clip_patch"] = visual.hidden_states[-1][:, 1:, :]
                    for layer in REFERENCE_LAYERS:
                        per_layer[layer][condition].append(
                            flatten_representation(captured[layer]).astype(np.float16))
                    if source_lengths is None:
                        fused_tokens = int(captured["fused"].shape[1])
                        pooled = int(bf.stage_token_hw[0] * bf.stage_token_hw[1])
                        if fused_tokens <= 4 * pooled:
                            raise ValueError("fused token 数不足以分出五路来源")
                        source_lengths = [fused_tokens - 4 * pooled] + [pooled] * 4
                    attention = getattr(bf, "last_resampler_attention", None)
                    if attention is not None:
                        weights = attention.float()[0].mean(dim=0)
                        shares, start = [], 0
                        for length in source_lengths:
                            shares.append(float(weights[start:start + length].sum()))
                            start += length
                        attention_rows.append(shares)
                print("d1 sample %d done" % index, flush=True)
    finally:
        capture.remove()
        bf.probe_capture_resampler_attention = False

    report = {
        "protocol": {"samples": len(ctx["indices"]), "conditions": CONDITIONS,
                     "layers": REFERENCE_LAYERS, "note": "zero_image 为全零参考图"},
        "layers": {},
        "source_attention": {"names": source_names,
                             "mean_share": [float(np.mean([row[c] for row in attention_rows]))
                                            for c in range(len(source_names))]},
        "bottlenecks": [],
    }
    for layer in REFERENCE_LAYERS:
        entries = per_layer[layer]
        contrasts = {name: [] for name in CONTRASTS}
        rotation_relative = []
        for position in range(len(ctx["indices"])):
            reference = entries["matched"][position].astype(np.float32)
            zero = entries["zero_image"][position].astype(np.float32)
            for name in CONTRASTS:
                contrasts[name].append(
                    normalized_response(reference, entries[name][position].astype(np.float32), zero))
            rotation_relative.append(
                relative_change(reference, entries["rot90"][position].astype(np.float32)))
        correct_matrix = np.stack(entries["matched"]).astype(np.float32)
        rotated_matrix = np.stack(entries["rot90"]).astype(np.float32)
        report["layers"][layer] = {
            "dimension": int(correct_matrix.shape[1]),
            "d_rot90": summary_stats(contrasts["rot90"]),
            "d_color_nearest": summary_stats(contrasts["color_nearest"]),
            "d_random": summary_stats(contrasts["random"]),
            "d_zero": summary_stats([relative_change(entries["matched"][p].astype(np.float32),
                                                    entries["zero_image"][p].astype(np.float32))
                                     for p in range(len(ctx["indices"]))]),
            "rotation_relative_change": summary_stats(rotation_relative),
            "correct_spectrum": gram_statistics(correct_matrix),
            "rotated_spectrum": gram_statistics(rotated_matrix),
        }
        del per_layer[layer]

    tracked = ["rot90", "color_nearest", "random"]
    for previous, current in zip(REFERENCE_LAYERS, REFERENCE_LAYERS[1:]):
        before = report["layers"][previous]
        after = report["layers"][current]
        ratios = {}
        for name in tracked:
            left = before["d_%s" % name]["mean"]
            right = after["d_%s" % name]["mean"]
            ratios[name] = None if not left or right is None else right / left
        hit = all(value is not None and value < BOTTLENECK_RATIO for value in ratios.values())
        if hit:
            report["bottlenecks"].append({"from": previous, "to": current, "ratios": ratios})
    report["bottleneck_ratio_threshold"] = BOTTLENECK_RATIO
    write_json(Path(ctx["output"]) / "d1_reference_trace" / "report.json", report)

    rows = []
    for layer in REFERENCE_LAYERS:
        item = report["layers"][layer]
        rows.append([layer, item["dimension"],
                     item["d_rot90"]["mean"] or 0.0, item["d_color_nearest"]["mean"] or 0.0,
                     item["d_random"]["mean"] or 0.0, item["d_zero"]["mean"] or 0.0,
                     item["correct_spectrum"]["effective_rank"],
                     item["correct_spectrum"]["pc_variance"][0] if item["correct_spectrum"]["pc_variance"] else 0.0])
    text = compact_table(["layer", "dim", "D_rot", "D_col", "D_rnd", "D_zero", "erank", "PC1"], rows)
    attention_text = "source attention share: " + " ".join(
        "%s=%.3f" % (name, value) for name, value in zip(source_names, report["source_attention"]["mean_share"]))
    print(text, flush=True)
    print(attention_text, flush=True)
    print("candidate bottlenecks:", report["bottlenecks"], flush=True)
    (Path(ctx["output"]) / "d1_reference_trace").mkdir(parents=True, exist_ok=True)
    (Path(ctx["output"]) / "d1_reference_trace" / "trace.txt").write_text(
        text + "\n" + attention_text + "\n", encoding="utf-8")
    return report

BASE_STEPS = [1, 181, 481, 781, 981]


class ResidualProbe:
    """捕获 16 层 texture residual；turn 结束前必须 remove。"""

    def __init__(self, unet):
        import torch

        self.torch = torch
        self.processors = texture_processors(unet)
        for index, proc in enumerate(self.processors):
            proc.texture_probe_observer = self._make(index)
        self.store = {}

    def _make(self, index):
        def observer(residual):
            self.store[index] = residual.detach().to("cpu", dtype=self.torch.float16)
        return observer

    def reset(self):
        self.store = {}

    def remove(self):
        for proc in self.processors:
            proc.texture_probe_observer = None


class LayerDisable:
    """按层关闭 texture residual；transform 返回零即该层不注入。"""

    def __init__(self, unet, disabled):
        import torch

        disabled = set(disabled)
        self.processors = texture_processors(unet)
        for index, proc in enumerate(self.processors):
            if index in disabled:
                def zero(residual):
                    return torch.zeros_like(residual)
                proc.texture_probe_transform = zero

    def remove(self):
        for proc in self.processors:
            proc.texture_probe_transform = None


def tensor_rms(tensor):
    return float(tensor.float().square().mean().sqrt())


def sample_targets(ctx, index):
    """与 113817 相同的固定目标 latent、噪声与文本。"""
    import torch

    batch = ctx["dataset"][index]
    posterior = ctx["vae"].encode(batch["image"][None].to(ctx["device"], ctx["dtype"])).latent_dist
    eta = torch.randn(posterior.mean.shape,
                      generator=torch.Generator().manual_seed(100000 + index)).to(ctx["device"], ctx["dtype"])
    target = (posterior.mean + posterior.std * eta) * ctx["vae"].config.scaling_factor
    if not torch.isfinite(target).all():
        raise ValueError("目标 latent 非有限")
    noise = torch.randn(target.shape,
                        generator=torch.Generator().manual_seed(42 + index * 1000)).to(ctx["device"], ctx["dtype"])
    text = ctx["text"](batch["text_input_ids"].to(ctx["device"]))[0]
    return target, noise, text


def d2_residual_trace(ctx):
    """16 层 texture residual 的参考敏感性热图，同时记录 epsilon 级响应。"""
    import torch

    unet, scheduler = ctx["unet"], ctx["scheduler"]
    probe = ResidualProbe(unet)
    conditions = ["matched", "rot90", "color_nearest", "random", "zero_tokens"]
    contrasts = ["rot90", "color_nearest", "random"]
    steps = len(FULL_STEPS)
    layers = len(probe.processors)
    if layers != 16:
        raise ValueError("texture 层数不是 16：%d" % layers)
    ratio = {name: np.zeros((steps, layers)) for name in contrasts}
    absolute = {name: np.zeros((steps, layers)) for name in conditions}
    epsilon = {name: np.zeros(steps) for name in conditions}
    count = 0
    try:
        with torch.inference_mode():
            for index in ctx["indices"]:
                tokens = condition_tokens(ctx, index)
                target, noise, text = sample_targets(ctx, index)
                for t_position, timestep in enumerate(FULL_STEPS):
                    t = torch.tensor([timestep], device=ctx["device"], dtype=torch.long)
                    noisy = scheduler.add_noise(target, noise, t)
                    residuals, predictions = {}, {}
                    for name in conditions:
                        probe.reset()
                        pred = unet(noisy, t,
                                    encoder_hidden_states=torch.cat([text, tokens[name]], 1)).sample
                        if not torch.isfinite(pred).all():
                            raise ValueError("预测非有限：%s" % name)
                        if len(probe.store) != layers:
                            raise ValueError("residual 捕获层数不足")
                        residuals[name] = dict(probe.store)
                        predictions[name] = pred.float()
                    for name in conditions:
                        for layer in range(layers):
                            absolute[name][t_position, layer] += tensor_rms(residuals[name][layer])
                        epsilon[name][t_position] += tensor_rms(
                            predictions[name] - predictions["matched"])
                    for name in contrasts + ["zero_tokens"]:
                        for layer in range(layers):
                            numerator = tensor_rms(residuals["matched"][layer].float()
                                                   - residuals[name][layer].float())
                            if name in contrasts:
                                denominator = tensor_rms(residuals["matched"][layer].float()
                                                         - residuals["zero_tokens"][layer].float())
                                ratio[name][t_position, layer] += (
                                    numerator / denominator if denominator > 1e-12 else np.nan)
                    del residuals, predictions
                count += 1
                print("d2 sample %d done" % index, flush=True)
    finally:
        probe.remove()

    result = {"protocol": {"samples": count, "timesteps": FULL_STEPS, "layers": layers,
                           "conditions": conditions,
                           "note": "R 为同一 timestep 下 RMS 差分比；zero_tokens 为真实全零条件"},
              "layer_order": [_layer_name(unet, index) for index in range(layers)],
              "r_rot90": (ratio["rot90"] / count).tolist(),
              "r_color_nearest": (ratio["color_nearest"] / count).tolist(),
              "r_random": (ratio["random"] / count).tolist(),
              "epsilon_relative_response": {name: (epsilon[name] / count).tolist() for name in conditions},
              "absolute_residual_rms": {name: (absolute[name] / count).tolist() for name in conditions}}
    write_json(Path(ctx["output"]) / "d2_residual_trace" / "report.json", result)
    for name in contrasts:
        print("d2 R_%s (rows=timestep, cols=layer1..16)" % name, flush=True)
        print(format_heatmap(FULL_STEPS, ["L%d" % (i + 1) for i in range(layers)], result["r_%s" % name]),
              flush=True)
    print("d2 epsilon relative response vs matched:", flush=True)
    for name in conditions:
        values = result["epsilon_relative_response"][name]
        print("  %-14s %s" % (name, " ".join("%.5f" % v for v in values)), flush=True)
    return result


def _layer_name(unet, index):
    names = [name for name, proc in unet.attn_processors.items() if hasattr(proc, "to_k_ip")]
    return names[index]


def d3_timestep_region(ctx):
    """离线重算 timestep × 区域 × 条件的 matched 对照差。"""
    source = Path(ctx["args"]["previous_report"])
    data = json.loads(source.read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise ValueError("前次报告未完成：%s" % source)
    mapping = {"random": "wrong_1", "color_nearest": "color_nearest",
               "rot90": "rot90", "zero_tokens": "zero_tokens"}
    samples = {}
    for record in data["records"]:
        index = record["sample_index"]
        for name, key in mapping.items():
            for region in REGIONS:
                delta = record["region_losses"][region][key] - record["region_losses"][region]["matched"]
                samples.setdefault((name, region, record["timestep"]), {}).setdefault(index, []).append(delta)
    timesteps = sorted({key[2] for key in samples})
    result = {"source": str(source), "samples": len(data["samples"]), "timesteps": timesteps,
              "mapping": mapping, "regions": REGIONS,
              "delta": {}, "interior_positive_samples": {}}
    for name in mapping:
        result["delta"][name] = {}
        for region in REGIONS:
            per_step = []
            for timestep in timesteps:
                per_sample = samples[(name, region, timestep)]
                per_step.append(float(np.mean([np.mean(v) for v in per_sample.values()])))
            result["delta"][name][region] = per_step
        interior = []
        for timestep in timesteps:
            per_sample = samples[(name, "interior", timestep)]
            interior.append(int(sum(np.mean(v) > 0 for v in per_sample.values())))
        result["interior_positive_samples"][name] = interior
    write_json(Path(ctx["output"]) / "d3_timestep_region" / "report.json", result)
    for name in mapping:
        print("d3 delta L(%s - matched), positive = correct reference better" % name, flush=True)
        rows = [[region] + result["delta"][name][region] for region in REGIONS]
        print(format_heatmap([str(r[0]) for r in rows], timesteps, [r[1:] for r in rows]), flush=True)
        print("  interior positive samples:", result["interior_positive_samples"][name], flush=True)
    return result


def d4_layer_ablation(ctx):
    """按 G1..G4 分组关闭 texture residual，比较区域 ΔL 谁在帮忙、谁在制造副作用。"""
    import torch

    unet, scheduler = ctx["unet"], ctx["scheduler"]
    conditions = ["matched", "random", "zero_tokens"]
    configs = [("baseline", ())] + [(name, GROUP_RANGES[name]) for name in sorted(GROUP_RANGES)]
    steps = len(FULL_STEPS)
    totals = {config: {name: {region: np.zeros(steps) for region in REGIONS} for name in conditions}
              for config, _ in configs}
    count = 0
    with torch.inference_mode():
        for index in ctx["indices"]:
            tokens = condition_tokens(ctx, index)
            target, noise, text = sample_targets(ctx, index)
            masks = ctx["regions"][index]
            for config, disabled in configs:
                handle = LayerDisable(unet, disabled)
                try:
                    for t_position, timestep in enumerate(FULL_STEPS):
                        t = torch.tensor([timestep], device=ctx["device"], dtype=torch.long)
                        noisy = scheduler.add_noise(target, noise, t)
                        for name in conditions:
                            pred = unet(noisy, t,
                                        encoder_hidden_states=torch.cat([text, tokens[name]], 1)).sample
                            if not torch.isfinite(pred).all():
                                raise ValueError("预测非有限：%s/%s" % (config, name))
                            error = (pred.float() - noise.float()).square().mean(1)[0].cpu().numpy()
                            for region in REGIONS:
                                totals[config][name][region][t_position] += float(error[masks[region]].mean())
                finally:
                    handle.remove()
            count += 1
            print("d4 sample %d done" % index, flush=True)

    result = {"protocol": {"samples": count, "timesteps": FULL_STEPS, "conditions": conditions,
                           "groups": {name: list(GROUP_RANGES[name]) for name in sorted(GROUP_RANGES)},
                           "note": "ΔL = L_condition - L_matched；负值表示该条件下预测更好"},
              "base_steps": BASE_STEPS, "delta": {}}
    for config, _ in configs:
        entry = {}
        for region in REGIONS:
            per_step = (totals[config]["random"][region] - totals[config]["matched"][region]) / count
            zero_step = (totals[config]["zero_tokens"][region] - totals[config]["matched"][region]) / count
            entry[region] = {"random_minus_matched": per_step.tolist(),
                             "zero_minus_matched": zero_step.tolist()}
            selected = [position for position, timestep in enumerate(FULL_STEPS) if timestep in BASE_STEPS]
            entry[region]["base_steps_random_mean"] = float(per_step[selected].mean())
            entry[region]["base_steps_zero_mean"] = float(zero_step[selected].mean())
        result["delta"][config] = entry
    write_json(Path(ctx["output"]) / "d4_layer_ablation" / "report.json", result)
    print("d4 ΔL(random - matched) by config and region, base steps mean", flush=True)
    rows = []
    for config, _ in configs:
        entry = result["delta"][config]
        rows.append([config] + [entry[region]["base_steps_random_mean"] for region in REGIONS])
    print(compact_table(["config"] + REGIONS, rows), flush=True)
    print("d4 ΔL(zero_tokens - matched) by config and region, base steps mean", flush=True)
    rows = []
    for config, _ in configs:
        entry = result["delta"][config]
        rows.append([config] + [entry[region]["base_steps_zero_mean"] for region in REGIONS])
    print(compact_table(["config"] + REGIONS, rows), flush=True)
    return result
