"""E21：先校准目标侧评分，再以相同预算比较 A0/A1；结果驱动后续阶段。"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import build_region_masks
from eval.eval_utils import estimate_foreground_mask
from eval.metrics import _binary_edges, _dilate_binary
from models.harmonic_period import harmonic_period
from models.pattern_utilization import PatternKVLoRA
from models.target_pattern_score import TargetScorer, direction, interior_rectangle, patch
from tools.e15_common import write_json
from tools.e15_stages import texture_processors
from tools.e19_2_identity import PATTERNS
from tools.e19_handcrafted import digest
from tools.e20_utilization import WRONG, REGIONS, case_stat, context, frozen_digest, load_pipeline, masked_mse, predict


ATTRS = ("identity", "orientation", "period")


def structure(image, mask, sketch):
    """复用 E15 的边缘阈值、容差与前景估计，输入为单步 x0 而非完整生成。"""
    array = (image[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    pil = Image.fromarray(array)
    edge = _binary_edges(np.asarray(pil.convert("L"), dtype=np.float32), .08, .16)
    reference = _binary_edges(np.asarray(sketch.convert("L"), dtype=np.float32), .04, .12)
    precision = float((edge & _dilate_binary(reference, 5)).sum() / max(edge.sum(), 1))
    recall = float((reference & _dilate_binary(edge, 5)).sum() / max(reference.sum(), 1))
    fg = estimate_foreground_mask(pil, pil.size)
    truth = mask.squeeze().cpu().numpy() > .5
    return {"sketch_iou": float((fg & truth).sum()/max((fg | truth).sum(), 1)),
            "edge_f1": 2*precision*recall/max(precision+recall, 1e-8)}


def decode(pipe, latent):
    # VAE 权重冻结，但 A1 的梯度必须经过解码器回到 epsilon/attention。
    return (pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample + 1) / 2


def clean_prediction(pipe, noisy, epsilon, t):
    alpha = pipe.scheduler.alphas_cumprod[t].to(noisy.device).float()
    return (noisy.float() - (1-alpha).sqrt() * epsilon.float()) / alpha.sqrt()


def canvas(root, row, roi, mask):
    """在已知衣身矩形内放完整参考，向外周期延拓；不把残缺 crop 当完整纹样。"""
    x, y, w, h = roi
    source = Image.open(root / "e19_2_b/data" / row["texture"]).convert("RGB").resize((w, h), Image.BILINEAR)
    tile = to_tensor(source)
    xx = (torch.arange(mask.shape[-1]) - x) % w
    yy = (torch.arange(mask.shape[-2]) - y) % h
    image = tile[:, yy[:, None], xx[None, :]][None]
    return image * mask + 1 - mask


@torch.no_grad()
def prepare(root, pipe, old, out):
    path = out / "cache.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    saved = {"text": old["text"], "train": [], "eval": [], "excluded": []}
    for split in ("train", "eval"):
        for ex in old[split]:
            ex = copy.deepcopy(ex)
            roi = interior_rectangle(build_region_masks(ex["mask"].float(), 17)[0])
            if roi is None:
                saved["excluded"].append({"split": split, "case": ex["case"], "reason": "no complete 48x64 interior rectangle"})
                continue
            image = canvas(root, ex["row"], roi, ex["mask"].float())
            latent = pipe.vae.encode((image.to(pipe.device) * 2 - 1)).latent_dist.mean * pipe.vae.config.scaling_factor
            ex.update(target=latent.half().cpu(), roi=roi, rgb=image.half(), target_patch=patch(image, roi))
            saved[split].append(ex)
        print("[e21-cache]", split, len(saved[split]), flush=True)
    torch.save(saved, path)
    write_json(out / "subset.json", {s: [{"case": e["case"], "row": e["row"], "roi": e["roi"]} for e in saved[s]] for s in ("train", "eval")})
    return saved


@torch.no_grad()
def calibrate(pipe, pattern, cache, out):
    scorer = TargetScorer(pattern.identity).to(pipe.device).eval()
    features, labels, images = [], [], {"train": [], "eval": []}
    for split in ("train", "eval"):
        for ex in cache[split]:
            target = ex["target_patch"].to(pipe.device)
            reconstructed = patch(decode(pipe, ex["target"].to(pipe.device)), ex["roi"])
            label = PATTERNS.index(ex["row"]["pattern"])
            images[split].append((ex, target, reconstructed, label))
            if split == "train":
                for image in (target, reconstructed):
                    features.append(scorer.features(image).cpu())
                    labels.append(label)
    scorer.fit(torch.cat(features).numpy(), labels)
    result = {}
    for split in ("train", "eval"):
        rows = []
        for ex, target, reconstructed, label in images[split]:
            for name, image in (("clean", target), ("vae", reconstructed)):
                pred = int(scorer(image).argmax(-1))
                freq = float(harmonic_period(image)["scalar"])
                row = {"case": ex["case"], "source": name, "class": PATTERNS[label], "id_correct": pred == label,
                       "frequency": ex["row"]["frequency"], "predicted_frequency": freq,
                       "period_correct": abs(freq-ex["row"]["frequency"]) < .5,
                       "period_mae": abs(128/max(freq, 1)-128/ex["row"]["frequency"]),
                       "orientation_correct": bool((direction(image)[:, 0] * direction(target)[:, 0] > 0).item()) if label == 0 else None}
                rows.append(row)
        result[split] = {}
        for name in ("clean", "vae"):
            selected = [r for r in rows if r["source"] == name]
            result[split][name] = {"identity_accuracy": float(np.mean([r["id_correct"] for r in selected])),
                                  "identity_by_class": {k: float(np.mean([r["id_correct"] for r in selected if r["class"] == k])) for k in PATTERNS},
                                  "direction_accuracy": float(np.mean([r["orientation_correct"] for r in selected if r["class"] == "stripe"])),
                                  "period_accuracy": float(np.mean([r["period_correct"] for r in selected])),
                                  "period_mae": float(np.mean([r["period_mae"] for r in selected]))}
        write_json(out / (split + "_scorer_records.json"), rows)
    result["pass"] = all(v["identity_accuracy"] >= .90 and min(v["identity_by_class"].values()) >= .75
                         and v["direction_accuracy"] == 1. and v["period_accuracy"] >= .90 for v in result["eval"].values())
    # 验证 loss 确实可微；冻结评分器不等于切断到预测图的梯度。
    with torch.enable_grad():
        x = images["eval"][0][2].detach().requires_grad_(True)
        loss = sum(scorer.losses(x, images["eval"][0][1], images["eval"][0][3]).values())
        grad = torch.autograd.grad(loss, x)[0]
        result["image_gradient_finite_nonzero"] = bool(torch.isfinite(grad).all() and grad.abs().sum() > 0)
    result["pass"] &= result["image_gradient_finite_nonzero"]
    torch.save(scorer.state_dict(), out / "scorer.pt")
    write_json(out / "scorer_audit.json", result)
    print("[e21-scorer]", result, flush=True)
    return scorer, result


def train(pipe, scorer, parameters, cache, aligned, steps, out):
    rng = random.Random(2121)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=16., growth_interval=100000)
    text = cache["text"].to(pipe.device)
    schedule, logs = [], []
    for step in range(steps):
        ex = cache["train"][rng.randrange(len(cache["train"]))]
        timestep, seed = (181, 481, 781)[step % 3], 400000 + step
        t = torch.tensor([timestep], device=pipe.device)
        target, mask = ex["target"].to(pipe.device), ex["mask"].to(pipe.device)
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed)).to(pipe.device, torch.float16)
        noisy = pipe.scheduler.add_noise(target, noise, t)
        eps = predict(pipe, None, ex["tokens"]["matched"], noisy, t, text, context(pipe, ex), mask)
        predicted = patch(decode(pipe, clean_prediction(pipe, noisy, eps, timestep)), ex["roi"])
        scores = scorer.losses(predicted, ex["target_patch"].to(pipe.device), PATTERNS.index(ex["row"]["pattern"]))
        # 对等轴格纹/点阵不伪造横竖方向监督。
        pattern_loss = scores["identity"] + scores["period"] + (scores["orientation"] if ex["row"]["pattern"] == "stripe" else 0. * scores["orientation"])
        diffusion = (eps.float() - noise.float()).square().mean()
        loss = diffusion + (.1 if aligned else 0.) * pattern_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(grad)
        scaler.step(optimizer)
        scaler.update()
        schedule.append([ex["case"], timestep, seed])
        if step == 0 or (step+1) % 25 == 0:
            logs.append({"step": step+1, "diffusion": float(diffusion.detach()), **{k: float(v.detach()) for k, v in scores.items()}, "grad": float(grad)})
            write_json(out / "training_partial.json", logs)
            print("[e21-train]", out.name, logs[-1], flush=True)
    write_json(out / "train_report.json", {"schedule": schedule, "logs": logs, "parameters": sum(p.numel() for p in parameters)})
    return schedule


@torch.no_grad()
def evaluate(pipe, scorer, cache, out):
    records = []
    text = cache["text"].to(pipe.device)
    for ex in cache["eval"]:
        sketch = context(pipe, ex)
        sketch_image = Image.open(out.parent.parent / ("e19_2_c/cases/%02d_sketch.png" % ex["case"])).convert("RGB")
        target, mask = ex["target"].to(pipe.device), ex["mask"].to(pipe.device)
        regions = {k: v.to(pipe.device) for k, v in ex["regions"].items()}
        fullregions = dict(zip(REGIONS, build_region_masks(mask.float(), 17)))
        for seed in (42, 43):
            noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed + ex["case"]*1000)).to(pipe.device, torch.float16)
            for step in (181, 481, 781):
                t = torch.tensor([step], device=pipe.device)
                noisy = pipe.scheduler.add_noise(target, noise, t)
                for name in ("matched",) + WRONG:
                    if name == "wrong_orientation" and ex["row"]["pattern"] != "stripe":
                        continue
                    eps = predict(pipe, None, ex["tokens"][name], noisy, t, text, sketch, mask)
                    image = decode(pipe, clean_prediction(pipe, noisy, eps, step))
                    predicted = patch(image, ex["roi"])
                    losses = scorer.losses(predicted, ex["target_patch"].to(pipe.device), PATTERNS.index(ex["row"]["pattern"]))
                    # 白色背景受控目标上的软前景；阈值仅用于报告结构，训练不依赖它。
                    fg = (image.detach().clamp(0, 1).mean(1, keepdim=True) < .95).float()
                    struct = structure(image, mask, sketch_image)
                    hard_period = float(harmonic_period(predicted)["scalar"])
                    records.append({"case": ex["case"], "seed": seed, "t": step, "intervention": name,
                                    "pattern": ex["row"]["pattern"], "scores": {k: float(v) for k, v in losses.items()},
                                    "identity_correct": int(scorer(predicted).argmax(-1)) == PATTERNS.index(ex["row"]["pattern"]),
                                    "direction_correct": bool((direction(predicted)[:, 0]*direction(ex["target_patch"].to(pipe.device))[:, 0]>0).item()) if ex["row"]["pattern"] == "stripe" else None,
                                    "period_correct": abs(hard_period-ex["row"]["frequency"]) < .5,
                                    "period_pixel_mae": abs(128/max(hard_period, 1)-128/ex["row"]["frequency"]),
                                    "epsilon": {k: float(masked_mse(eps, noise, v)) for k, v in regions.items()},
                                    "rgb": {k: float(masked_mse(image, ex["rgb"].to(pipe.device), v)) for k, v in fullregions.items()},
                                    **struct, "leakage": float((fg*fullregions["background"]).sum()/fullregions["background"].sum().clamp_min(1))})
        write_json(out / "records_partial.json", records)
        print("[e21-eval]", out.name, ex["case"], flush=True)
    write_json(out / "records.json", records)
    return records


def compare(a, b):
    index = lambda rows: {(r["case"], r["seed"], r["t"], r["intervention"]): r for r in rows}
    aa, bb = index(a), index(b)
    report = {"matched_improvement": {}, "target_advantage": {}, "epsilon_advantage": {}, "preservation": {}, "hard_metrics": {}}
    for attr, wrong in zip(ATTRS, WRONG):
        valid = lambda r: attr != "orientation" or r["pattern"] == "stripe"
        report["matched_improvement"][attr] = case_stat([(r["case"], bb[k]["scores"][attr]-r["scores"][attr]) for k, r in aa.items() if r["intervention"] == "matched" and valid(r)])
        for metric, dest in (("scores", "target_advantage"), ("epsilon", "epsilon_advantage")):
            field = attr if metric == "scores" else "interior"
            report[dest][attr] = case_stat([(r["case"], r[metric][field]-aa[k[:3]+("matched",)][metric][field]) for k, r in aa.items() if r["intervention"] == wrong and valid(r)])
    for region in ("boundary", "background"):
        report["preservation"][region] = case_stat([(r["case"], (r["rgb"][region]-bb[k]["rgb"][region])/max(bb[k]["rgb"][region], 1e-8)) for k, r in aa.items()])
    for field in ("sketch_iou", "edge_f1", "leakage"):
        report["preservation"][field] = case_stat([(r["case"], r[field]-bb[k][field]) for k, r in aa.items()])
    for field in ("identity_correct", "direction_correct", "period_correct", "period_pixel_mae"):
        report["hard_metrics"][field] = {name: case_stat([(r["case"], float(r[field])) for r in rows if r["intervention"] == "matched" and r[field] is not None]) for name, rows in (("aligned", a), ("control", b))}
    report["preservation_pass"] = all(report["preservation"][r]["ci95"][1] <= .02 for r in ("boundary", "background", "leakage")) and all(report["preservation"][r]["ci95"][0] >= -.02 for r in ("sketch_iou", "edge_f1"))
    report["improved_attributes"] = [k for k, v in report["matched_improvement"].items() if v["ci95"][0] > 0]
    report["A_pass"] = bool(report["improved_attributes"]) and report["preservation_pass"]
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    root, out = Path(args.root), Path(args.root)/"e21"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = {"version": 1, "training_steps": args.steps, "lr": 1e-4, "pattern_weight": .1,
                "t": [181, 481, 781], "capacity": "E20 B rank4 texture K/V, last8 pattern tokens, original E19 start",
                "target": "controlled garment: complete reference in maximal eroded-interior 3:4 rectangle, periodic extension outside; same saved train/eval sketches and condition tokens as E20",
                "freeze": "all representations, TCPM, sketch/text, base U-Net and VAE; VAE float32 for differentiable x0 decode",
                "scorer": "frozen A3-1 identity + train-only logistic head; gradient direction stripes only; separate radial FFT cosine; hard harmonic period calibration",
                "oracle_gate": "clean and VAE eval: id>=90%, eachclass>=75%, stripe direction100%, period>=90%; finite nonzero image gradients",
                "A_gate": "paired case-bootstrap target-score improvement A1 over A0 for at least one attribute; RGB boundary/background upperCI<=2%, leakage<=2pp, IoU drop<=2pp",
                "selection": "fixed600steps final checkpoint, no validation sweep; A0 computes same pattern graph with coefficient0",
                "generation": "not before B stable all3 attributes; no automatic new architecture"}
    write_json(out/"protocol.json", protocol)
    pipe, bf, pattern, modules, _, _ = load_pipeline(root, args.device)
    original = {k: digest(m) for k, m in modules.items()}
    pipe.vae.float()
    cache = prepare(root, pipe, torch.load(root/"e20/cache.pt", map_location="cpu", weights_only=False), out)
    scorer, audit = calibrate(pipe, pattern, cache, out)
    if not audit["pass"] or args.audit_only:
        write_json(out/"status.json", {"stage": "scorer_audit", "pass": audit["pass"], "training_executed": False,
                                      "next": "A0/A1" if audit["pass"] else "repair target measurement; not a GAM training failure"})
        return
    adapters = {}
    for i, proc in enumerate(texture_processors(pipe.unet)):
        if proc.layer_group == "semantic":
            continue
        for key in ("to_k_ip", "to_v_ip"):
            wrapper = PatternKVLoRA(getattr(proc, key)).to(args.device)
            setattr(proc, key, wrapper)
            adapters[str(i)+key] = wrapper
    parameters = [p for m in adapters.values() for p in m.parameters() if p.requires_grad]
    initial = {k: copy.deepcopy(m.state_dict()) for k, m in adapters.items()}
    frozen = {k: frozen_digest(m) for k, m in modules.items()}
    baseline_dir = out/"frozen_baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline = (json.loads((baseline_dir/"records.json").read_text()) if (baseline_dir/"records.json").exists()
                else evaluate(pipe, scorer, cache, baseline_dir))
    records, schedules = {}, {}
    for variant in (0, 1):
        name = "A"+str(variant)
        folder = out/name
        folder.mkdir(exist_ok=True)
        for k, m in adapters.items():
            m.load_state_dict(initial[k])
        if (folder/"adapter.pt").exists():
            checkpoint = torch.load(folder/"adapter.pt", map_location=args.device, weights_only=False)
            assert checkpoint["protocol"] == protocol and checkpoint["source_hashes"] == original
            for k, m in adapters.items():
                missing, unexpected = m.load_state_dict(checkpoint["adapters"][k], strict=False)
                assert missing == ["base.weight"] and not unexpected
            schedules[name] = json.loads((folder/"train_report.json").read_text())["schedule"]
        else:
            schedules[name] = train(pipe, scorer, parameters, cache, variant == 1, args.steps, folder)
        torch.save({"adapters": {k: {n: v for n, v in m.state_dict().items() if n.startswith("adapter_")} for k, m in adapters.items()}, "protocol": protocol, "source_hashes": original}, folder/"adapter.pt")
        records[name] = (json.loads((folder/"records.json").read_text()) if (folder/"records.json").exists()
                         else evaluate(pipe, scorer, cache, folder))
        assert all(frozen_digest(m) == frozen[k] for k, m in modules.items())
    assert schedules["A0"] == schedules["A1"]
    result = compare(records["A1"], records["A0"])
    result["versus_frozen"] = {k: compare(v, baseline) for k, v in records.items()}
    result["A_pass"] &= result["versus_frozen"]["A1"]["preservation_pass"]
    result.update(freeze_pass=True, schedule_identical=True, generation_executed=False,
                  next="B isolated attribute hard negatives" if result["A_pass"] else "stop A: no reliable target-side improvement with preservation")
    write_json(out/"report.json", result)
    print("[e21-final]", result, flush=True)


if __name__ == "__main__":
    main()
