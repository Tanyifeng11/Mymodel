"""E20 A0/A1，A 失败后 B0/B1；相同预算和 C-style 独立验证，不自动完整生成。"""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import build_region_masks, build_sketch_garment_mask
from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, rotation_only
from models.pattern_utilization import PatternInterface, PatternKVLoRA
from tools.e15_common import write_json
from tools.e15_d5_generation import build_inference_args, load_inference_module
from tools.e15_stages import texture_processors
from tools.e18_gold_probe import conditioner
from tools.e19_2_identity import PATTERNS
from tools.e19_handcrafted import bf_tokens, bootstrap, digest, rgb, text_tokens


WRONG = ("wrong_identity", "wrong_orientation", "wrong_period")
REGIONS = ("interior", "boundary", "background")


def masked_mse(pred, noise, mask):
    return ((pred.float() - noise.float()).square() * mask).sum() / (mask.sum() * pred.shape[1]).clamp_min(1e-8)


def frozen_digest(module):
    trainable = {k for k, v in module.named_parameters() if v.requires_grad}
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        if key not in trainable:
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_pipeline(root, device):
    args = argparse.Namespace(checkpoint=str(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt"),
                              texture_ckpt=None, base_model_path=str(root / "models/stable-diffusion-v1-5"),
                              vae_model_path=str(root / "models/stable-diffusion-v1-5/vae"),
                              clip_model=str(root / "models/clip"), device=device, seed=42, steps=50)
    ns = build_inference_args(args)
    ns.texture_num_tokens, ns.force_texture_num_tokens_override = 16, False
    pipe, _ = load_inference_module().prepare(ns)
    e18_path = root / "e18_2/b2/joint_model.pt"
    if not e18_path.exists():
        e18_path = root / "output_eval/e18_2/b2/joint_model.pt"
    state = torch.load(e18_path, map_location="cpu", weights_only=False)
    bf = conditioner(state["bf_texture_conditioner"]).to(device, torch.float16)
    pipe.tcpm_lite.load_state_dict(state["tcpm_lite"], strict=True)
    del state
    pattern = IdentityGeometryPattern().to(device)
    pattern_path = root / "e19_2_a3/fft_rotation_42.pt"
    if not pattern_path.exists():
        pattern_path = root / "output_eval/e19_2_a3/fft_rotation_42.pt"
    pattern.load_state_dict(torch.load(pattern_path, map_location=device, weights_only=False)["model"])
    modules = {"unet": pipe.unet, "sketch": pipe.reference_unet, "bf": bf, "pattern": pattern,
               "clip": pipe.image_encoder, "text": pipe.text_encoder, "vae": pipe.vae, "tcpm": pipe.tcpm_lite,
               "gam_bf": pipe.bf_texture_conditioner}
    for module in modules.values():
        module.eval().requires_grad_(False)
    pipe.set_scale(.6)
    pipe.set_ipa_scale(1.)
    for proc in texture_processors(pipe.unet):
        proc.num_tokens = 24
    assert pipe.scheduler.config.prediction_type == "epsilon"
    return pipe, bf, pattern, modules, ns.width, ns.height


def prepare_cache(root, data_root, pipe, bf, pattern, width, height, hashes):
    path = root / "e20/cache.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["hashes"] == hashes
        return saved
    previous = json.loads((root / "e19_2_c/protocol.json").read_text())
    manifest = json.loads((root / "e19_2_b/data/manifest.json").read_text())["splits"]
    assert set(r["frequency"] for r in manifest["train"]).isdisjoint(r["frequency"] for r in previous["cases"])
    assert set(r["phase"] for r in manifest["train"]).isdisjoint(r["phase"] for r in previous["cases"])
    eval_sketch_hashes = {hashlib.sha256(Image.open(root / ("e19_2_c/cases/%02d_sketch.png" % i)).tobytes()).hexdigest() for i in range(32)}
    train_manifest = json.loads((root / "data/processed/bf_full_audit_v1/training_clean.json").read_text())
    order = list(range(len(train_manifest)))
    random.Random(2042).shuffle(order)
    sketches, sketch_meta = [], []
    for index in order:
        image = Image.open(data_root / train_manifest[index]["sketch"]).convert("RGB").resize((width, height), Image.BILINEAR)
        sha = hashlib.sha256(image.tobytes()).hexdigest()
        if sha in eval_sketch_hashes or sha in {r["hash"] for r in sketch_meta}:
            continue
        mask, info = build_sketch_garment_mask(image, width, height)
        interior = build_region_masks(to_tensor(mask)[None], 17)[0]
        if F.interpolate(interior, size=(height // 8, width // 8), mode="area").sum() <= 4:
            continue
        sketches.append((image, mask))
        sketch_meta.append({"train_index": index, "hash": sha, "mask_source": info["mask_source"]})
        if len(sketches) == 64:
            break
    assert len(sketches) == 64
    saved = {"hashes": hashes, "train_sketches": sketch_meta, "evaluation_protocol": previous, "train": [], "eval": []}
    with torch.no_grad():
        text = text_tokens(pipe, ["a cloth"])
        saved["text"] = text.cpu()
        for split, source_rows, cases in (("train", manifest["train"], manifest["train"]),
                                         ("eval", manifest["primary"], previous["cases"])):
            lookup = {(r["pattern"], r["palette"], r["frequency"], r["angle"], r["phase"]): r for r in source_rows}
            frequencies = sorted({r["frequency"] for r in source_rows})
            banks = {}
            for row in source_rows:
                image = Image.open(root / "e19_2_b/data" / row["texture"]).convert("RGB")
                source = rgb(image, pipe.device)
                banks[row["texture"]] = (bf_tokens(pipe, bf, image, text, width, height),
                                         pattern.identity_tokens(rotation_only(source, fft_orientation(source)[0])).half(),
                                         pattern.geometry_tokens(source).half())
            for index, row in enumerate(cases):
                k, c, f, a, p = (row[q] for q in ("pattern", "palette", "frequency", "angle", "phase"))
                donor_rows = {"matched": row,
                              "wrong_identity": lookup[PATTERNS[(PATTERNS.index(k) + 1) % 4], c, f, a, p],
                              "wrong_orientation": lookup[k, c, f, 90 - a, p],
                              "wrong_period": lookup[k, c, frequencies[(frequencies.index(f) + 1) % len(frequencies)], a, p],
                              "wrong_appearance": lookup[k, (c + 1) % 4, f, a, p]}
                base = banks[row["texture"]]
                tokens = {}
                for name, donor in donor_rows.items():
                    bank = banks[donor["texture"]]
                    combined = torch.cat([bank[0] if name == "wrong_appearance" else base[0],
                                          bank[1] if name == "wrong_identity" else base[1],
                                          bank[2] if name in WRONG[1:] else base[2]], 1)
                    tokens[name] = pipe.tcpm_lite(combined, text).cpu()
                if split == "eval":
                    sketch = Image.open(root / ("e19_2_c/cases/%02d_sketch.png" % index)).convert("RGB")
                    mask_image = Image.open(root / ("e19_2_c/cases/%02d_mask.png" % index)).convert("L")
                    canvas = Image.open(root / ("e19_2_c/cases/%02d_matched.png" % index)).convert("RGB")
                else:
                    sketch, mask_image = sketches[index % len(sketches)]
                    image = Image.open(root / "e19_2_b/data" / row["texture"]).convert("RGB")
                    canvas = Image.composite(image.resize((width, height), Image.BILINEAR), Image.new("RGB", (width, height), "white"), mask_image)
                def latent(image):
                    return (pipe.vae.encode(to_tensor(image)[None].to(pipe.device, torch.float16) * 2 - 1).latent_dist.mean * pipe.vae.config.scaling_factor).cpu()
                target, sketch_latent = latent(canvas), latent(sketch)
                mask = to_tensor(mask_image)[None].half()
                regions = {name: F.interpolate(value, size=target.shape[-2:], mode="area") for name, value in zip(REGIONS, build_region_masks(mask, 17))}
                saved[split].append({"row": row, "tokens": tokens, "target": target, "sketch": sketch_latent,
                                     "mask": mask, "regions": regions, "case": index})
            print("[e20-cache]", split, len(saved[split]), flush=True)
    torch.save(saved, path)
    write_json(root / "e20/cache_manifest.json", {"train_rows": [q["row"] for q in saved["train"]], "train_sketches": sketch_meta,
                                               "eval_protocol": previous, "source_hashes": hashes})
    return saved


def context(pipe, example):
    device = pipe.device
    with torch.no_grad():
        pipe.reference_unet(example["sketch"].to(device), torch.tensor(0, device=device), encoder_hidden_states=None, return_dict=False)
    return {name: proc.cache["hidden_states"].detach() for name, proc in pipe.reference_unet.attn_processors.items() if "attn1" in name}


def predict(pipe, interface, tokens, noisy, t, text, sketch, mask):
    tokens = tokens.to(pipe.device)
    if interface is not None:
        tokens = interface(tokens)
    return pipe.unet(noisy, t, encoder_hidden_states=torch.cat([text, tokens], 1),
                     cross_attention_kwargs={"sa_hidden_states": sketch, "tcpm_garment_mask": mask}).sample


def evaluate(pipe, interface, cache, out):
    records = []
    text = cache["text"].to(pipe.device)
    with torch.no_grad():
        for example in cache["eval"]:
            i = example["case"]
            sketch = context(pipe, example)
            target, mask = example["target"].to(pipe.device), example["mask"].to(pipe.device)
            regions = {k: v.to(pipe.device) for k, v in example["regions"].items()}
            for seed in (42, 43):
                noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed + i * 1000)).to(pipe.device, torch.float16)
                for step in (181, 481, 781):
                    t = torch.tensor([step], device=pipe.device, dtype=torch.long)
                    noisy = pipe.scheduler.add_noise(target, noise, t)
                    base = None
                    for name, tokens in example["tokens"].items():
                        pred = predict(pipe, interface, tokens, noisy, t, text, sketch, mask)
                        assert torch.isfinite(pred).all()
                        if name == "matched":
                            base = pred.float()
                        records.append({"case": i, "noise_seed": seed, "timestep": step, "intervention": name,
                                        "valid_attribute": name != "wrong_orientation" or example["row"]["pattern"] == "stripe",
                                        "loss": {r: float(masked_mse(pred, noise, m)) for r, m in regions.items()},
                                        "epsilon_interior_rms": float(masked_mse(pred, base, regions["interior"]).sqrt())})
            if (i + 1) % 8 == 0:
                print("[e20-audit]", out.name, i + 1, flush=True)
    write_json(out / "records.json", records)
    return records


def indexed(records):
    return {(r["case"], r["noise_seed"], r["timestep"], r["intervention"]): r for r in records}


def case_stat(pairs):
    ids = sorted({i for i, v in pairs})
    return bootstrap([np.mean([v for j, v in pairs if i == j]) for i in ids])


def summarize(records, baseline):
    current, original = indexed(records), indexed(baseline)
    report = {"advantage": {}, "matched_error_change": {}, "off_target_error_change": {}}
    for name in WRONG:
        region_stats = {}
        for region in REGIONS:
            pairs = [(r["case"], r["loss"][region] - current[(r["case"], r["noise_seed"], r["timestep"], "matched")]["loss"][region])
                     for r in records if r["intervention"] == name and r["valid_attribute"]]
            region_stats[region] = case_stat(pairs)
        report["advantage"][name] = region_stats
    for region in REGIONS:
        pairs = [(r["case"], (r["loss"][region] - original[key]["loss"][region]) / max(original[key]["loss"][region], 1e-8))
                 for key, r in current.items() if r["intervention"] == "matched"]
        report["matched_error_change"][region] = case_stat(pairs)
    for region in ("boundary", "background"):
        report["off_target_error_change"][region] = {}
        for name in ("matched",) + WRONG:
            pairs = [(r["case"], (r["loss"][region] - original[key]["loss"][region]) / max(original[key]["loss"][region], 1e-8))
                     for key, r in current.items() if r["intervention"] == name and r["valid_attribute"]]
            report["off_target_error_change"][region][name] = case_stat(pairs)
    report["denoising_pass"] = all(q["interior"]["mean"] > 0 and q["interior"]["ci95"][0] > 0 for q in report["advantage"].values())
    report["preservation_pass"] = all(q["ci95"][1] <= .02 for values in report["off_target_error_change"].values() for q in values.values())
    report["matched_not_degraded"] = report["matched_error_change"]["interior"]["ci95"][1] <= .02
    report["pass"] = report["denoising_pass"] and report["preservation_pass"] and report["matched_not_degraded"]
    return report


def train(pipe, interface, parameters, cache, steps, aligned, out):
    rng = random.Random(2020)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=64., growth_interval=100000)
    text = cache["text"].to(pipe.device)
    logs, schedule = [], []
    start = [p.detach().cpu().clone() for p in parameters]
    for step in range(steps):
        name = WRONG[step % 3]
        eligible = [q for q in cache["train"] if name != "wrong_orientation" or q["row"]["pattern"] == "stripe"]
        example = eligible[rng.randrange(len(eligible))]
        timestep, noise_seed = rng.randrange(50, 951), 300000 + step
        sketch = context(pipe, example)
        target, mask = example["target"].to(pipe.device), example["mask"].to(pipe.device)
        interior = example["regions"]["interior"].to(pipe.device)
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(noise_seed)).to(pipe.device, torch.float16)
        t = torch.tensor([timestep], device=pipe.device, dtype=torch.long)
        noisy = pipe.scheduler.add_noise(target, noise, t)
        matched = predict(pipe, interface, example["tokens"]["matched"], noisy, t, text, sketch, mask)
        wrong = predict(pipe, interface, example["tokens"][name], noisy, t, text, sketch, mask)
        normal = (matched.float() - noise.float()).square().mean()
        lm, lw = masked_mse(matched, noise, interior), masked_mse(wrong, noise, interior)
        ranking = F.relu(.01 * lm.detach() + lm - lw)
        # A0/B0 也经过相同 paired graph 的前后向，监督项系数为零。
        loss = normal + (ranking if aligned else 0. * ranking)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(grad), "nonfinite interface gradient"
        scaler.step(optimizer)
        scaler.update()
        schedule.append({"step": step + 1, "case": example["case"], "wrong": name, "timestep": timestep, "noise_seed": noise_seed})
        if step == 0 or (step + 1) % 25 == 0:
            logs.append({"step": step + 1, "diffusion": float(normal.detach()), "matched_interior": float(lm.detach()),
                         "wrong_interior": float(lw.detach()), "ranking": float(ranking.detach()), "gradient_norm": float(grad)})
            print("[e20-train]", out.name, logs[-1], flush=True)
            write_json(out / "training_partial.json", logs)
    delta = sum(float((p.detach().cpu() - old).square().sum()) for p, old in zip(parameters, start)) ** .5
    assert delta > 0, "interface did not update"
    write_json(out / "train_report.json", {"steps": steps, "logs": logs, "schedule": schedule, "parameter_delta_l2": delta,
                                           "trainable_parameters": sum(p.numel() for p in parameters), "aligned": aligned})
    return schedule


def alignment_effect(aligned, control):
    a, b = indexed(aligned), indexed(control)
    results = {}
    for name in WRONG:
        pairs = []
        for key, row in a.items():
            if row["intervention"] != name or not row["valid_attribute"]:
                continue
            matched_key = key[:3] + ("matched",)
            gain = row["loss"]["interior"] - a[matched_key]["loss"]["interior"]
            gain -= b[key]["loss"]["interior"] - b[matched_key]["loss"]["interior"]
            pairs.append((row["case"], gain))
        results[name] = case_stat(pairs)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.root) / "e20"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = {"representation": "E19 A3-1 identity seed42, original frozen geometry, E18.2 BF, original frozen TCPM; all fixed",
                "A": "post-TCPM identity/geometry residual maps, independent 768->32->768, zero output initialization; appearance unchanged",
                "B_if_A_fails": "fresh original E19 interface + rank4 LoRA on active texture attention K/V for last8 pattern tokens only; no A warm start; same training budget",
                "arms": "A0/A1 and conditional B0/B1 use identical initialization, case/noise/timestep sequence, optimizer steps and paired forward/backward count",
                "training": {"steps": args.steps, "lr": .0001, "optimizer": "AdamW", "weight_decay": .01,
                             "diffusion": "full image matched epsilon MSE", "alignment": "relu(.01*stopgrad(Lmatched_interior)+Lmatched_interior-Lwrong_interior)",
                             "alignment_weight": 1., "wrong_sampling": "balanced id/orientation/period; orientation on stripes only", "timestep_range": [50, 950]},
                "evaluation": "exact saved E19.2-C 32 cases, targets/masks/sketches/noise seeds42/43, timesteps181/481/781; 8 stripe cases for meaningful orientation",
                "holdout": "train frequency/phase disjoint; training_clean sketch pool excludes exact resized validation sketch hashes; synthetic controlled garments only",
                "gate": "all3 interior wrong-minus-matched mean and case-bootstrap lower95CI>0; matched interior and every matched/wrong boundary/background relative-error upper95CI<=2% versus frozen baseline",
                "interpretation": "report A1-A0 / B1-B0 paired advantage gain and matched absolute loss separately; positive ranking alone does not imply improved reconstruction",
                "selection": "fixed final checkpoint only, no validation weight/step sweep", "generation": "not run until utilization gate passes"}
    write_json(out / "protocol.json", protocol)
    pipe, bf, pattern, modules, width, height = load_pipeline(root, args.device)
    hashes = {k: digest(v) for k, v in modules.items()}
    cache = prepare_cache(root, Path(args.data_root), pipe, bf, pattern, width, height, hashes)
    baseline_dir = out / "frozen_baseline"
    baseline_dir.mkdir(exist_ok=True)
    if (baseline_dir / "records.json").exists():
        baseline = json.loads((baseline_dir / "records.json").read_text())
    else:
        baseline = evaluate(pipe, None, cache, baseline_dir)
    old = indexed(json.loads((root / "e19_2_c/records_partial.json").read_text()))
    reproduction = max(abs(r["loss"][region] - old[key]["loss"][region]) for key, r in indexed(baseline).items() for region in REGIONS)
    write_json(baseline_dir / "reproduction.json", {"max_region_loss_abs_diff_from_C": reproduction})
    assert reproduction < 2e-5, "C baseline reproduction failed"
    report = {"protocol": protocol, "baseline_reproduction_max_error": reproduction, "arms": {}, "alignment_effect": {}, "generation_executed": False}
    selected = None
    for stage in ("A", "B"):
        torch.manual_seed(42)
        interface, adapters = None, {}
        if stage == "A":
            interface = PatternInterface().to(args.device)
            parameters = list(interface.parameters())
            initial = copy.deepcopy(interface.state_dict())
            with torch.no_grad():
                x = cache["eval"][0]["tokens"]["matched"].to(args.device)
                assert torch.equal(interface(x), x)
        else:
            for i, proc in enumerate(texture_processors(pipe.unet)):
                if proc.layer_group == "semantic":
                    continue
                for key in ("to_k_ip", "to_v_ip"):
                    wrapper = PatternKVLoRA(getattr(proc, key)).to(args.device)
                    setattr(proc, key, wrapper)
                    adapters["%d_%s" % (i, key)] = wrapper
            parameters = [p for module in adapters.values() for p in module.parameters() if p.requires_grad]
            initial = {k: copy.deepcopy(v.state_dict()) for k, v in adapters.items()}
        frozen_before = {k: frozen_digest(v) for k, v in modules.items()}
        histories, evaluations = {}, {}
        for variant in (0, 1):
            name = stage + str(variant)
            folder = out / name
            folder.mkdir(exist_ok=True)
            if interface is not None:
                interface.load_state_dict(initial)
            else:
                for key, module in adapters.items():
                    module.load_state_dict(initial[key])
            checkpoint = folder / "adapter.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
                assert saved["protocol"] == protocol and saved["frozen_source_hashes"] == hashes
                if interface is not None:
                    interface.load_state_dict(saved["interface"])
                else:
                    for key, module in adapters.items():
                        missing, unexpected = module.load_state_dict(saved["kv_adapters"][key], strict=False)
                        assert missing == ["base.weight"] and not unexpected
                histories[name] = json.loads((folder / "train_report.json").read_text())["schedule"]
                print("[e20-resume]", name, "fixed final checkpoint", flush=True)
            else:
                histories[name] = train(pipe, interface, parameters, cache, args.steps, bool(variant), folder)
            if variant:
                assert histories[name] == histories[stage + "0"]
            torch.save({"stage": stage, "variant": variant, "interface": interface.state_dict() if interface is not None else None,
                        "kv_adapters": {k: {n: v for n, v in m.state_dict().items() if n.startswith("adapter_")} for k, m in adapters.items()},
                        "protocol": protocol, "frozen_source_hashes": hashes}, checkpoint)
            records = (json.loads((folder / "records.json").read_text()) if (folder / "records.json").exists()
                       else evaluate(pipe, interface, cache, folder))
            evaluations[name] = records
            result = summarize(records, baseline)
            result["freeze_audit"] = {k: frozen_before[k] == frozen_digest(v) for k, v in modules.items()}
            assert all(result["freeze_audit"].values())
            report["arms"][name] = result
            write_json(folder / "report.json", result)
            write_json(out / "report_partial.json", report)
            print("[e20-arm]", name, result["pass"], {k: v["interior"] for k, v in result["advantage"].items()}, flush=True)
        report["alignment_effect"][stage] = alignment_effect(evaluations[stage + "1"], evaluations[stage + "0"])
        selected = next((stage + str(i) for i in (1, 0) if report["arms"][stage + str(i)]["pass"]), None)
        if selected:
            break
    report.update(complete=True, selected=selected, next="generation verification" if selected else "stop; projection and texture KV adaptation did not establish stable utilization")
    write_json(out / "report.json", report)
    print("[e20-final]", selected, flush=True)


if __name__ == "__main__":
    main()
