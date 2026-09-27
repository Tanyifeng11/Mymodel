"""E22-A: explicit spatial geometry positive control; stop at the first failed gate."""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import build_region_masks
from models.spatial_geometry import SpatialAdapter, SpatialInjection, geometry_map, spatial_gate
from tools.e15_common import write_json
from tools.e20_utilization import case_stat, context, frozen_digest, load_pipeline, masked_mse
from tools.e21_target_supervision import clean_prediction, decode, structure
from tools.e19_handcrafted import rgb


TIMESTEPS = (181, 481, 781)
SEEDS = (42, 43)


def paired_maps(root, cache, output, device):
    path = output / "map_cache.pt"
    banks = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else {}
    initial_count = len(banks)
    manifest = json.loads((root / "e19_2_b/data/manifest.json").read_text())["splits"]
    for split in ("train", "eval"):
        source = manifest["train" if split == "train" else "primary"]
        lookup = {(r["pattern"], r["palette"], r["frequency"], r["angle"], r["phase"]): r for r in source}
        frequencies = sorted({r["frequency"] for r in source})
        for example in cache[split]:
            row = example["row"]
            pattern, color, frequency, angle, phase = (row[k] for k in ("pattern", "palette", "frequency", "angle", "phase"))
            donor = {
                "matched": row,
                "wrong_orientation": lookup[(pattern, color, frequency, 90-angle, phase)],
                "wrong_period": lookup[(pattern, color, frequencies[(frequencies.index(frequency)+1) % len(frequencies)], angle, phase)],
            }
            for item in donor.values():
                key = item["texture"]
                if key not in banks:
                    image = Image.open(root / "e19_2_b/data" / key).convert("RGB")
                    banks[key] = F.interpolate(geometry_map(rgb(image, device)), size=(32, 32), mode="area").half().cpu()
            example["geometry_keys"] = {name: item["texture"] for name, item in donor.items()}
    if len(banks) != initial_count:
        torch.save(banks, path)
    return banks


def select_map(example, banks, name, mode):
    correct = banks[example["geometry_keys"]["matched"]].float()
    if mode in ("constant", "period_constant", "constant_all"):
        result = torch.ones_like(correct)
    else:
        result = correct.clone()
        if name == "wrong_orientation":
            result[:, :2] = banks[example["geometry_keys"][name]][:, :2]
        if name == "wrong_period":
            result[:, 2:4] = banks[example["geometry_keys"][name]][:, 2:4]
            result[:, 5:6] = banks[example["geometry_keys"][name]][:, 5:6]
    if mode in ("orientation", "constant"):
        return result[:, [0, 1, 4]]
    if mode in ("period", "period_constant"):
        return result[:, [2, 3, 5]]
    return result


def predict(pipe, injection, example, geometry, noisy, t, text, sketch, mask):
    if injection is not None:
        if "spatial_gate" not in example:
            example["spatial_gate"] = spatial_gate(example["mask"], (32, 32)).half()
        gate = example["spatial_gate"].to(pipe.device)
        injection.set(geometry.to(pipe.device, torch.float16), gate)
    tokens = example["tokens"]["matched"].to(pipe.device)
    return pipe.unet(noisy, t, encoder_hidden_states=torch.cat([text, tokens], 1),
                     cross_attention_kwargs={"sa_hidden_states": sketch, "tcpm_garment_mask": mask}).sample


def audit_maps(cache, banks, output):
    stripes, periods, separation = [], [], []
    for example in cache["eval"]:
        row = example["row"]
        good = banks[example["geometry_keys"]["matched"]].float()
        wrong_ori = select_map(example, banks, "wrong_orientation", "orientation")
        wrong_period = select_map(example, banks, "wrong_period", "all")
        period_frequency = (good[:, 2:4].square().sum(1).sqrt().mean() * 16).item()
        periods.append(abs(period_frequency-row["frequency"]) < .5)
        separation.append(float((good[:, 2:4]-wrong_period[:, 2:4]).abs().mean()))
        if row["pattern"] == "stripe":
            score = float(good[:, 0].mean())
            stripes.append((row["angle"], score, float((good[:, :2]-wrong_ori[:, :2]).abs().mean())))
    signs = {a: sum(s for aa,s,_ in stripes if aa==a)/sum(aa==a for aa,_,_ in stripes) for a in set(a for a,_,_ in stripes)}
    result = {"cases": len(cache["eval"]), "stripe_cases": len(stripes), "mean_cos2_by_angle": signs,
              "orientation_intervention_l1": sum(v for _,_,v in stripes)/len(stripes),
              "period_intervention_l1": sum(separation)/len(separation),
              "period_accuracy": sum(periods)/len(periods),
              "pass": len(cache["eval"]) == 32 and len(stripes) == 8 and len(signs) == 2 and
                      max(signs.values()) > .5 and min(signs.values()) < -.5 and
                      all(v > .5 for _,_,v in stripes) and sum(periods)/len(periods) >= .9}
    write_json(output / "map_audit.json", result)
    return result


def evaluate(pipe, injection, adapter, cache, banks, output, mode, variants):
    records, text = [], cache["text"].to(pipe.device)
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        for example in cache["eval"]:
            case = example["case"]
            sketch = context(pipe, example)
            target, mask = example["target"].to(pipe.device), example["mask"].to(pipe.device)
            regions = {k: v.to(pipe.device) for k,v in example["regions"].items()}
            full = dict(zip(("interior", "boundary", "background"), build_region_masks(mask.float(), 17)))
            target_image = to_tensor(Image.open(Path(cache["root"]) / ("e19_2_c/cases/%02d_matched.png" % case)).convert("RGB"))[None].to(pipe.device)
            sketch_image = Image.open(Path(cache["root"]) / ("e19_2_c/cases/%02d_sketch.png" % case)).convert("RGB")
            for seed in SEEDS:
                noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed+case*1000)).to(pipe.device, torch.float16)
                for step in TIMESTEPS:
                    t = torch.tensor([step], device=pipe.device)
                    noisy = pipe.scheduler.add_noise(target, noise, t)
                    for variant in variants:
                        if variant == "wrong_orientation" and example["row"]["pattern"] != "stripe":
                            continue
                        geo = None if mode == "base" else select_map(example, banks, variant, mode)
                        eps = predict(pipe, injection, example, geo, noisy, t, text, sketch, mask)
                        image = decode(pipe, clean_prediction(pipe, noisy, eps, step))
                        struct = structure(image, mask, sketch_image)
                        fg = (image.detach().clamp(0, 1).mean(1, keepdim=True) < .95).float()
                        records.append({"case": case, "seed": seed, "t": step, "variant": variant,
                                        "pattern": example["row"]["pattern"],
                                        "epsilon": {r: float(masked_mse(eps, noise, m)) for r,m in regions.items()},
                                        "rgb": {r: float(masked_mse(image, target_image, m)) for r,m in full.items()},
                                        **struct,
                                        "leakage": float((fg*full["background"]).sum()/full["background"].sum().clamp_min(1))})
            write_json(output / "records_partial.json", records)
            print("[e22-eval]", output.name, case, flush=True)
    write_json(output / "records.json", records)
    return records


def train(pipe, injection, adapter, cache, banks, output, mode):
    adapter.train()
    rng = random.Random(2222 if mode in ("orientation", "constant") else
                        2224 if mode in ("period", "period_constant") else 2223)
    parameters = list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=16., growth_interval=100000)
    text = cache["text"].to(pipe.device)
    schedule, logs = [], []
    orientation = [e for e in cache["train"] if e["row"]["pattern"] == "stripe"]
    for step in range(600):
        wrong = ("wrong_orientation" if mode in ("orientation", "constant") else
                 "wrong_period" if mode in ("period", "period_constant") else
                 ("wrong_orientation" if step%2==0 else "wrong_period"))
        pool = orientation if wrong == "wrong_orientation" else cache["train"]
        example = pool[rng.randrange(len(pool))]
        timestep, seed = (181, 481)[step%2], 500000+step
        target, mask = example["target"].to(pipe.device), example["mask"].to(pipe.device)
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed)).to(pipe.device, torch.float16)
        t = torch.tensor([timestep], device=pipe.device)
        noisy = pipe.scheduler.add_noise(target, noise, t)
        sketch = context(pipe, example)
        region = example["regions"]["interior"].to(pipe.device)
        right = select_map(example, banks, "matched", mode)
        incorrect = select_map(example, banks, wrong, mode)
        eps = predict(pipe, injection, example, right, noisy, t, text, sketch, mask)
        bad = predict(pipe, injection, example, incorrect, noisy, t, text, sketch, mask)
        diffusion = (eps.float()-noise.float()).square().mean()
        lm, lw = masked_mse(eps, noise, region), masked_mse(bad, noise, region)
        ranking = F.relu(.01*lm.detach()+lm-lw)
        loss = diffusion + ranking
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(grad)
        scaler.step(optimizer)
        scaler.update()
        schedule.append([example["case"], timestep, seed, wrong])
        if step == 0 or (step+1)%25 == 0:
            logs.append({"step": step+1, "diffusion": float(diffusion.detach()), "matched": float(lm.detach()),
                         "wrong": float(lw.detach()), "rank": float(ranking.detach()), "grad": float(grad)})
            write_json(output / "training_partial.json", logs)
            print("[e22-train]", output.name, logs[-1], flush=True)
    write_json(output / "train_report.json", {"schedule": schedule, "logs": logs,
                                               "trainable": sum(p.numel() for p in parameters)})
    torch.save({"adapter": adapter.state_dict(), "mode": mode}, output / "adapter.pt")
    return schedule


def summarize(aligned, baseline, wrongs):
    key = lambda r: (r["case"], r["seed"], r["t"], r["variant"])
    aa, bb = {key(r):r for r in aligned}, {key(r):r for r in baseline}
    report = {"advantage": {}, "safety": {}}
    for wrong in wrongs:
        pairs = [(r["case"], r["epsilon"]["interior"]-aa[(r["case"],r["seed"],r["t"],"matched")]["epsilon"]["interior"])
                 for r in aligned if r["variant"] == wrong]
        report["advantage"][wrong] = case_stat(pairs)
        report["advantage"][wrong+"_by_timestep"] = {
            str(t): case_stat([(r["case"], r["epsilon"]["interior"]-aa[(r["case"],r["seed"],r["t"],"matched")]["epsilon"]["interior"])
                               for r in aligned if r["variant"]==wrong and r["t"]==t]) for t in TIMESTEPS}
    matched = [r for r in aligned if r["variant"] == "matched"]
    for region in ("boundary", "background"):
        report["safety"][region] = case_stat([(r["case"], (r["rgb"][region]-bb[key(r)]["rgb"][region])/
                                                max(bb[key(r)]["rgb"][region], 1e-8)) for r in matched])
    for metric in ("sketch_iou", "edge_f1", "leakage"):
        report["safety"][metric] = case_stat([(r["case"], r[metric]-bb[key(r)][metric]) for r in matched])
    report["safety_pass"] = (all(report["safety"][r]["ci95"][1] <= .03 for r in ("boundary", "background"))
                             and all(report["safety"][r]["ci95"][0] >= -.02 for r in ("sketch_iou", "edge_f1"))
                             and report["safety"]["leakage"]["ci95"][1] <= .02)
    report["geometry_pass"] = all(report["advantage"][w]["ci95"][0] > 0 for w in wrongs)
    report["pass"] = report["geometry_pass"] and report["safety_pass"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.root)/"e22"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    pipe, _, _, modules, _, _ = load_pipeline(root, "cuda:0")
    pipe.vae.float()
    cache = torch.load(root/"e20/cache.pt", map_location="cpu", weights_only=False)
    cache["root"] = str(root)
    banks = paired_maps(root, cache, output, pipe.device)
    map_report = audit_maps(cache, banks, output)
    if not map_report["pass"]:
        write_json(output/"report.json", {"stage":"A0", "map_audit":map_report, "next":"repair explicit map; no adapter training"})
        return
    frozen = {k: frozen_digest(m) for k,m in modules.items()}
    base_dir = output/"S0"
    base_dir.mkdir(exist_ok=True)
    base = json.loads((base_dir/"records.json").read_text()) if (base_dir/"records.json").exists() else evaluate(pipe, None, None, cache, banks, base_dir, "base", ("matched",))
    reports, schedules = {}, {}
    for name, mode in (("S1_constant", "constant"), ("S2_orientation", "orientation")):
        folder = output/name
        if (folder/"records.json").exists() and (folder/"train_report.json").exists():
            schedules[name] = json.loads((folder/"train_report.json").read_text())["schedule"]
            records = json.loads((folder/"records.json").read_text())
            reports[name] = summarize(records, base, ("wrong_orientation",) if mode=="orientation" else ())
            continue
        torch.manual_seed(42)
        adapter = SpatialAdapter(640, 3).to(pipe.device)
        injection = SpatialInjection(pipe.unet, adapter)
        folder.mkdir(exist_ok=True)
        schedules[name] = train(pipe, injection, adapter, cache, banks, folder, mode)
        variants = ("matched",) if mode=="constant" else ("matched", "wrong_orientation")
        records = evaluate(pipe, injection, adapter, cache, banks, folder, mode, variants)
        reports[name] = summarize(records, base, ("wrong_orientation",) if mode=="orientation" else ())
        injection.close()
        del adapter, injection
        assert all(frozen_digest(m)==frozen[k] for k,m in modules.items())
        write_json(output/"report_partial.json", {"map_audit":map_report, "groups":reports})
    # Constant and correct arms have identical train schedule and pair-forward budget.
    assert schedules["S1_constant"] == schedules["S2_orientation"]
    orientation_pass = reports["S2_orientation"]["geometry_pass"]
    result = {"stage":"A1_orientation", "map_audit":map_report, "groups":reports,
              "schedule_identical":True, "frozen_pass":True, "orientation_pass":orientation_pass,
              "orientation_safety_pass":reports["S2_orientation"]["safety_pass"],
              "generation_executed":False,
              "next":"A3 period positive control" if orientation_pass else "stop E22-A; inspect training target rather than train an encoder"}
    write_json(output/"report.json", result)
    print("[e22-A1]", orientation_pass, flush=True)
    if not orientation_pass:
        return
    # A3 keeps period isolated from orientation; A4 runs only if both geometry gates pass.
    for name, mode in (("S1_period_constant", "period_constant"), ("S2_period", "period")):
        folder = output/name
        folder.mkdir(exist_ok=True)
        if (folder/"records.json").exists() and (folder/"train_report.json").exists():
            schedules[name] = json.loads((folder/"train_report.json").read_text())["schedule"]
            records = json.loads((folder/"records.json").read_text())
        else:
            torch.manual_seed(42)
            adapter = SpatialAdapter(640, 3).to(pipe.device)
            injection = SpatialInjection(pipe.unet, adapter)
            schedules[name] = train(pipe, injection, adapter, cache, banks, folder, mode)
            records = evaluate(pipe, injection, adapter, cache, banks, folder, mode,
                               ("matched",) if mode=="period_constant" else ("matched", "wrong_period"))
            injection.close()
            del adapter, injection
            assert all(frozen_digest(m)==frozen[k] for k,m in modules.items())
        reports[name] = summarize(records, base, ("wrong_period",) if mode=="period" else ())
        write_json(output/"report_partial.json", {"map_audit":map_report, "groups":reports})
    assert schedules["S1_period_constant"] == schedules["S2_period"]
    period_pass = reports["S2_period"]["geometry_pass"]
    result.update(stage="A3_period", groups=reports, period_pass=period_pass,
                  period_safety_pass=reports["S2_period"]["safety_pass"],
                  next="A4 orientation+period" if period_pass else "stop E22-A3; no joint or learned geometry encoder")
    write_json(output/"report.json", result)
    if not period_pass:
        return
    # A4 combines the two explicit maps, with the final safety gate.
    for name, mode in (("S1_OP_constant", "constant_all"), ("S2_OP", "all")):
        torch.manual_seed(42)
        adapter = SpatialAdapter(640, 6).to(pipe.device)
        injection = SpatialInjection(pipe.unet, adapter)
        folder = output/name
        folder.mkdir(exist_ok=True)
        schedules[name] = train(pipe, injection, adapter, cache, banks, folder, mode)
        variants = ("matched",) if mode=="constant_all" else ("matched", "wrong_orientation", "wrong_period")
        records = evaluate(pipe, injection, adapter, cache, banks, folder, mode, variants)
        reports[name] = summarize(records, base, ("wrong_orientation", "wrong_period") if mode=="all" else ())
        injection.close()
        del adapter, injection
        assert all(frozen_digest(m)==frozen[k] for k,m in modules.items())
        write_json(output/"report_partial.json", {"map_audit":map_report, "groups":reports})
    assert schedules["S1_OP_constant"] == schedules["S2_OP"]
    result.update(stage="A4_orientation_period", groups=reports, joint_pass=reports["S2_OP"]["pass"],
                  next="E22-B spatial utilization / learned-map distillation" if reports["S2_OP"]["pass"] else "stop at spatial positive control; no learned encoder")
    write_json(output/"report.json", result)
    print("[e22-A4]", result["joint_pass"], flush=True)


if __name__ == "__main__":
    main()
