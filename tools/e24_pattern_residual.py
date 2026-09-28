"""E24：E5 锚定的纹样 residual；A 安全 Gate 通过后才训练 B。"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import build_region_masks, mask_backend_info
from models.target_pattern_score import direction, interior_rectangle, patch
from tools import e23_mechanism as old
from tools.e15_common import write_json
from tools.e15_stages import texture_processors
from tools.e20_utilization import case_stat, context, load_pipeline
from tools.e21_target_supervision import clean_prediction, decode
from tools.e22_o4_metrics import axial_distance, orientation
from tools.e23_m2 import final_summary


TRAIN_STEPS_A = 600
TRAIN_STEPS_B = 600
SEEDS = (42, 43)
SAFETY_TOLERANCE = .02


class PatternResidual(nn.Module):
    """仅把参考的轴向纹样方向投影成附加 token，不重建 E5 condition。"""

    def __init__(self, token_shape):
        super().__init__()
        self.token_shape = tuple(token_shape)
        self.project = nn.Linear(2, math.prod(token_shape))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, feature):
        return self.project(feature.float()).reshape(-1, *self.token_shape)


def reference_feature(image, expected=None):
    """输入只读参考图；expected 仅用于检查，不喂入模型。"""
    readout = orientation(image, (0, 0, *image.size))
    if not readout["valid"]:
        raise ValueError("参考纹样方向读出无效")
    if expected is not None and axial_distance(readout["theta"], expected) > 20:
        raise ValueError(f"参考方向读出与预先定义的旋转不一致: {readout['theta']}, {expected}")
    radians = math.radians(2 * readout["theta"])
    return torch.tensor([[math.cos(radians), math.sin(radians)]], dtype=torch.float32)


def score_loss(new, baseline, mask):
    delta = new.float() - baseline.float().detach()
    return (delta.square() * mask).sum() / (mask.sum() * delta.shape[1]).clamp_min(1)


def score_masks(mask, latent_size):
    regions = build_region_masks(mask.float(), 17)
    return {name: F.interpolate(value, size=latent_size, mode="area") for name, value in
            zip(("interior", "boundary", "background"), regions)}


def safety_gate(contrast, interior_score_rms):
    return (contrast["leakage"]["ci95"][1] <= SAFETY_TOLERANCE and
            contrast["sketch_iou"]["ci95"][0] >= -SAFETY_TOLERANCE and
            contrast["edge_f1"]["ci95"][0] >= -SAFETY_TOLERANCE and
            interior_score_rms > 1e-4)


def find_source(root, relative):
    for base in (root / "output_eval", root):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


@torch.no_grad()
def e5_pair(pipe, image, width, height, encoded):
    pair = pipe.get_image_embeds(pil_image=image, width=width, height=height,
        texture_mode="patch_resampled", text_embeds=encoded[0], negative_text_embeds=encoded[1],
        text_mask=encoded[2], negative_text_mask=encoded[3],
        aa_tcr_captions="a cloth", aa_tcr_negative_captions=" worst quality, low quality")
    return tuple(x.detach().cpu() for x in pair)


@torch.no_grad()
def build_training_pairs(root, pipe, out, width, height):
    """E18.1 训练参考 + E20 训练 sketch；同一 sketch 合成横/竖目标。"""
    cached = out / "train_cache.pt"
    if cached.exists():
        data = torch.load(cached, map_location="cpu", weights_only=False)
        return data["pairs"], data["text"]
    clean = find_source(root, "e18_1/clean")
    saved = find_source(root, "e20/cache.pt")
    rows = json.loads((clean / "train.json").read_text())
    cache = torch.load(saved, map_location="cpu", weights_only=False)
    sketches = cache["train"]
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group"], {})[row["variant"]] = row
    assert all(set(pair) == {"original", "rot90"} for pair in groups.values())
    encoded = pipe.encode_prompt("a cloth", pipe.device, 1, True, " worst quality, low quality",
                                 return_text_masks=True)
    result = []
    for index, (key, pair_rows) in enumerate(sorted(groups.items())):
        sketch = sketches[index % min(64, len(sketches))]
        mask = sketch["mask"].float()
        mask_image = Image.fromarray((mask[0, 0].numpy() * 255).round().astype(np.uint8), "L")
        roi = interior_rectangle(build_region_masks(mask, 17)[0])
        assert roi is not None
        pair = {"id": index, "source_group": key, "mask": mask.half(),
                "sketch": sketch["sketch"], "roi": roi, "variants": {}}
        for variant, row in pair_rows.items():
            image = Image.open(clean / row["texture"]).convert("RGB")
            expected = 90 if row["axis"] == "vertical" else 0
            feature = reference_feature(image, expected)
            target_image = Image.composite(image.resize((width, height), Image.Resampling.BILINEAR),
                                           Image.new("RGB", (width, height), "white"), mask_image)
            pixels = to_tensor(target_image)[None]
            target = pipe.vae.encode((pixels.to(pipe.device) * 2 - 1)).latent_dist.mean
            target = (target * pipe.vae.config.scaling_factor).half().detach().cpu()
            pair["variants"][variant] = {"feature": feature, "tokens": e5_pair(pipe, image, width, height, encoded)[0],
                "target": target, "target_patch": patch(pixels, roi).cpu(), "axis": row["axis"]}
        assert axial_distance(
            math.degrees(math.atan2(pair["variants"]["original"]["feature"][0, 1],
                                    pair["variants"]["original"]["feature"][0, 0])) / 2,
            math.degrees(math.atan2(pair["variants"]["rot90"]["feature"][0, 1],
                                    pair["variants"]["rot90"]["feature"][0, 0])) / 2) >= 70
        result.append(pair)
        print("[e24-cache]", len(result), "/", len(groups), flush=True)
    validation_cases = json.loads((find_source(root, "e23_m") / "cases.json").read_text())
    train_frequencies = {r["frequency"] for r in rows}
    validation_frequencies = {r["frequency"] for r in validation_cases["references"]}
    assert train_frequencies.isdisjoint(validation_frequencies)
    write_json(out / "train_split.json", {"source": str(clean), "groups": list(groups),
        "sketch_source": str(saved), "validation_source": "output_eval/e23_m/cases.json",
        "train_frequencies": sorted(train_frequencies),
        "validation_frequencies": sorted(validation_frequencies),
        "frequency_disjoint_from_validation": True,
        "target": "same sketch/mask/color; exact reference rot90; composite onto white canvas"})
    data = {"pairs": result, "text": cache["text"].cpu()}
    torch.save(data, cached)
    return data["pairs"], data["text"]


def predict_conditional(pipe, noisy, timestep, text, sketch, mask, positive):
    return pipe.unet(noisy, timestep, encoder_hidden_states=torch.cat([text, positive], 1),
                     cross_attention_kwargs={"sa_hidden_states": sketch,
                                             "tcpm_garment_mask": mask,
                                             "balanced_gate_timestep": timestep.float() / 1000}).sample


def training_item(pipe, pair, variant, adapter, text, seed, timestep):
    item = pair["variants"][variant]
    target = item["target"].to(pipe.device)
    mask = pair["mask"].to(pipe.device)
    sketch = context(pipe, pair)
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed)).to(pipe.device, target.dtype)
    t = torch.tensor([timestep], device=pipe.device)
    noisy = pipe.scheduler.add_noise(target, noise, t)
    positive = item["tokens"].to(pipe.device)
    feature = item["feature"].to(pipe.device)
    with torch.no_grad():
        baseline = predict_conditional(pipe, noisy, t, text, sketch, mask, positive)
    delta = adapter(feature).to(positive.dtype)
    current = predict_conditional(pipe, noisy, t, text, sketch, mask, positive + delta)
    regions = score_masks(mask, current.shape[-2:])
    diffusion = (current.float() - noise.float()).square().mean()
    keep = score_loss(current, baseline, regions["background"])
    boundary = score_loss(current, baseline, regions["boundary"])
    return {"current": current, "baseline": baseline, "noisy": noisy, "target": target,
            "target_patch": item["target_patch"].to(pipe.device), "regions": regions,
            "timestep": t, "diffusion": diffusion, "keep": keep, "boundary": boundary,
            "token_delta_rms": delta.float().square().mean().sqrt()}


def train_stage(pipe, pairs, adapter, text, out, stage, steps):
    out.mkdir(exist_ok=True)
    checkpoint = out / "adapter.pt"
    if checkpoint.exists():
        adapter.load_state_dict(torch.load(checkpoint, map_location=pipe.device, weights_only=False)["adapter"])
        return
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(init_scale=16., growth_interval=100000)
    text = text.to(pipe.device)
    rng = random.Random(2401 if stage == "A" else 2402)
    logs = []
    for step in range(steps):
        pair = pairs[rng.randrange(len(pairs))]
        timestep = (181, 481, 781)[step % 3]
        seed = (240000 if stage == "A" else 250000) + step
        variants = ("original",) if stage == "A" and step % 2 == 0 else (
            ("rot90",) if stage == "A" else ("original", "rot90"))
        items = [training_item(pipe, pair, variant, adapter, text, seed, timestep) for variant in variants]
        loss = sum(item["diffusion"] + 5 * item["keep"] + 5 * item["boundary"] for item in items) / len(items)
        orientation_loss = loss.new_zeros(())
        pair_loss = loss.new_zeros(())
        if stage == "B":
            full_images = [decode(pipe, clean_prediction(pipe, item["noisy"], item["current"], timestep))
                           for item in items]
            images = [patch(image, pair["roi"]) for image in full_images]
            targets = [item["target_patch"] for item in items]
            orientation_loss = sum((direction(a) - direction(b)).square().mean()
                                   for a, b in zip(images, targets)) / 2
            # 配对输出在相同背景和轮廓区域保持一致；方向损失只在内部 ROI 上计算。
            outside = build_region_masks(pair["mask"].to(pipe.device), 17)
            pair_loss = sum(score_loss(full_images[0], full_images[1], mask) for mask in outside[1:])
            loss = loss + .1 * orientation_loss + .5 * pair_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % 25 == 0:
            row = {"step": step + 1, "stage": stage, "loss": float(loss.detach()),
                   "diffusion": float(sum(x["diffusion"].detach() for x in items) / len(items)),
                   "keep": float(sum(x["keep"].detach() for x in items) / len(items)),
                   "boundary": float(sum(x["boundary"].detach() for x in items) / len(items)),
                   "orientation": float(orientation_loss.detach()), "pair": float(pair_loss.detach()),
                   "token_delta_rms": float(items[0]["token_delta_rms"].detach()), "gradient": float(norm)}
            logs.append(row)
            write_json(out / "training_partial.json", logs)
            print("[e24-train]", row, flush=True)
    torch.save({"adapter": adapter.state_dict(), "token_shape": adapter.token_shape,
                "stage": stage, "steps": steps}, out / "adapter.pt")
    write_json(out / "training.json", {"steps": steps, "logs": logs,
        "trainable": sum(p.numel() for p in adapter.parameters())})


@torch.no_grad()
def validation_bank(source, cases, banks, out, adapter, count):
    bank = {}
    for ref, _ in old.selected(cases, 0, count):
        for variant in ref["variants"]:
            name = variant["variant"]
            image = Image.open(out / variant["path"]).convert("RGB")
            feature = reference_feature(image, variant["theta"]).to(next(adapter.parameters()).device)
            positive, negative = banks["E5"][ref["id"], name]
            delta = adapter(feature).to(positive.dtype).cpu()
            bank[ref["id"], name] = (positive + delta, negative)
    return bank


@torch.no_grad()
def interior_response(pipe, source, cases, banks, out, adapter):
    pipe.scheduler.set_timesteps(50, device=pipe.device)
    values = []
    for ref, sk in old.selected(cases, 0, 8):
        trace = torch.load(source / "E5" / f"c{ref['id']:02d}_s42_original.pt",
                           map_location="cpu", weights_only=False)
        ctx = old.tree(trace["context"], pipe.device)
        z = trace["latent"][22].to(pipe.device)
        t = pipe.scheduler.timesteps[22]
        masks = old.latent_regions(out, sk, z.shape[-2:], pipe.device)
        variant = ref["variants"][0]
        image = Image.open(out / variant["path"]).convert("RGB")
        feature = reference_feature(image, variant["theta"]).to(pipe.device)
        token = banks["E5"][ref["id"], "original"][0].to(pipe.device)
        negative = banks["E5"][ref["id"], "original"][1].to(pipe.device)
        baseline = old.forward(pipe, z, t, ctx, (token, negative))[1]
        current = old.forward(pipe, z, t, ctx,
                              (token + adapter(feature).to(token.dtype), negative))[1]
        values.append(old.rms(current - baseline, masks["interior"]))
    return float(np.mean(values))


def copy_baseline(source, out):
    target = out / "G0"
    target.mkdir(exist_ok=True)
    for path in (source / "G0").glob("c*"):
        if path.suffix in (".png", ".json") and not (target / path.name).exists():
            shutil.copy2(path, target / path.name)


def contrast(out, arm, count):
    rows = {}
    for group in ("G0", arm):
        rows[group] = {p.stem: json.loads(p.read_text()) for p in (out / group).glob("c*.json")
                       if int(p.name[1:3]) < count}
    assert len(rows["G0"]) == len(rows[arm]) == count * 4
    return {metric: case_stat([(int(name[1:3]), row[metric] - rows["G0"][name][metric])
                               for name, row in rows[arm].items()])
            for metric in ("leakage", "sketch_iou", "edge_f1")}


def generate_and_report(pipe, source, cases, banks, out, adapter, arm, count, interior_rms):
    bank = validation_bank(source, cases, banks, out, adapter, count)
    for ref, sk in old.selected(cases, 0, count):
        for seed in SEEDS:
            for variant in ref["variants"]:
                old.generate(pipe, arm, ref, sk, seed, variant, bank, out,
                             Image.open(out / sk["path"]).width,
                             Image.open(out / sk["path"]).height, trace=False)
    stats = final_summary(out, ("G0", arm), count)
    delta = contrast(out, arm, count)
    report = {"cases": count, "arm": arm, "generation": stats, "versus_G0": delta,
              "conditional_interior_score_delta_rms": interior_rms,
              "safety_pass": safety_gate(delta, interior_rms)}
    write_json(out / (arm + f"_{count}_report.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "output_eval/e24"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    source = find_source(root, "e23_m")
    previous = find_source(root, "e23_m2")
    cases = json.loads((source / "cases.json").read_text())
    banks = torch.load(source / "tokens.pt", map_location="cpu", weights_only=False)
    shutil.copytree(source / "inputs", out / "inputs", dirs_exist_ok=True)
    write_json(out / "cases.json", cases)
    copy_baseline(previous, out)
    pipe, _, _, modules, width, height = load_pipeline(root, "cuda:0")
    pipe.set_progress_bar_config(disable=True)
    for processor in texture_processors(pipe.unet):
        processor.num_tokens = 16
    from tools.e22_4_generation import module_hashes
    frozen = module_hashes(modules)
    shape = banks["E5"][0, "original"][0].shape[1:]
    adapter = PatternResidual(shape).to(pipe.device)
    check_ref = cases["references"][0]["variants"][0]
    check_image = Image.open(out / check_ref["path"]).convert("RGB")
    encoded = pipe.encode_prompt("a cloth", pipe.device, 1, True, " worst quality, low quality",
                                 return_text_masks=True)
    replay_pair = e5_pair(pipe, check_image, width, height, encoded)
    anchor_error = max(float((a.float() - b.float()).abs().max()) for a, b in
                       zip(replay_pair, banks["E5"][0, "original"]))
    assert anchor_error < 1e-4, f"E5 token anchor differs from E23-M: {anchor_error}"
    first_ref, first_sketch = next(old.selected(cases, 0, 1))
    old.generate(pipe, "E24_anchor", first_ref, first_sketch, 42, first_ref["variants"][0],
                 banks["E5"], out, width, height, trace=False)
    anchor_name = "c00_s42_original.png"
    pixel_exact = np.array_equal(
        np.array(Image.open(out / "E24_anchor" / anchor_name)),
        np.array(Image.open(out / "G0" / anchor_name)))
    assert pixel_exact, "零 residual 的 E5 生成未逐像素复现 G0"
    pipe.vae.float()
    pairs, train_text = build_training_pairs(root, pipe, out, width, height)
    write_json(out / "protocol.json", {"train_steps_A": TRAIN_STEPS_A, "train_steps_B": TRAIN_STEPS_B,
        "train_pairs": len(pairs), "validation_cases": 32, "seeds": SEEDS, "steps": 50,
        "native_cfg": 7, "score_keep": "conditional epsilon MSE vs stopped E5 in background and boundary",
        "gate_A": "case-bootstrap upper leakage delta <= .02; lower IoU/EdgeF1 delta >= -.02; interior score delta RMS > 1e-4",
        "gate_B": "flip accuracy positive with case-bootstrap lower paired delta > 0 plus A safety",
        "mask_backend": mask_backend_info(), "frozen_hashes": frozen,
        "E5_token_anchor_max_abs_error": anchor_error, "E5_zero_residual_pixel_exact": pixel_exact})
    train_stage(pipe, pairs, adapter, train_text, out / "A_train", "A", TRAIN_STEPS_A)
    pipe.vae.half()
    adapter.eval()
    a_rms = interior_response(pipe, source, cases, banks, out, adapter)
    pilot = generate_and_report(pipe, source, cases, banks, out, adapter, "A_residual", 8, a_rms)
    if not pilot["safety_pass"]:
        write_json(out / "report.json", {"stage": "A_pilot", "A": pilot,
            "B_executed": False, "C_executed": False, "decision": "stop residual route"})
        return
    a = generate_and_report(pipe, source, cases, banks, out, adapter, "A_residual", 32, a_rms)
    if not a["safety_pass"]:
        write_json(out / "report.json", {"stage": "A_confirmation", "A": a,
            "B_executed": False, "C_executed": False, "decision": "stop residual route"})
        return
    pipe.vae.float()
    train_stage(pipe, pairs, adapter, train_text, out / "B_train", "B", TRAIN_STEPS_B)
    pipe.vae.half()
    adapter.eval()
    b_rms = interior_response(pipe, source, cases, banks, out, adapter)
    b = generate_and_report(pipe, source, cases, banks, out, adapter, "B_counterfactual", 32, b_rms)
    flips = b["generation"]["B_counterfactual"]["flip_accuracy"]
    base_flips = b["generation"]["G0"]["flip_accuracy"]
    # E5 的 paired flip 为 0；这里仍按案例计算显著性，不能把 seed 当独立样本。
    from tools.e22_4_generation import paired
    current = paired([json.loads(p.read_text()) for p in (out / "B_counterfactual").glob("c*.json")])
    baseline = paired([json.loads(p.read_text()) for p in (out / "G0").glob("c*.json")])
    baseline_index = {(r["reference"], r["sketch"], r["seed"]): r for r in baseline}
    flip_delta = case_stat([(r["reference"], float(r["flip_correct"]) -
                            float(baseline_index[r["reference"], r["sketch"], r["seed"]]["flip_correct"]))
                            for r in current])
    b_pass = bool(b["safety_pass"] and flips["mean"] > base_flips["mean"] and flip_delta["ci95"][0] > 0)
    report = {"stage": "B_confirmation", "A": a, "B": b, "paired_flip_gain": flip_delta,
              "B_pass": b_pass, "C_executed": False,
              "decision": "proceed to period stage" if b_pass else "stop before period stage",
              "frozen_pass": module_hashes(modules) == frozen}
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
