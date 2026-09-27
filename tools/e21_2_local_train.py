"""E21.2：固定梯度标定权重，局部 geometry-only 与同口径 G0 比较。"""

import argparse
import copy
import json
import random
from pathlib import Path

import torch

from models.local_pattern_geometry import losses as local_losses
from models.pattern_utilization import PatternKVLoRA
from models.target_pattern_score import TargetScorer, patch
from tools.e15_common import write_json
from tools.e15_stages import texture_processors
from tools.e20_utilization import frozen_digest, load_pipeline
from tools.e21_target_supervision import clean_prediction, compare, context, decode, evaluate, predict


def load_adapter(adapters, checkpoint):
    for key, module in adapters.items():
        missing, extra = module.load_state_dict(checkpoint["adapters"][key], strict=False)
        assert missing == ["base.weight"] and not extra


def train(pipe, cache, parameters, out):
    rng = random.Random(2121)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=16., growth_interval=100000)
    text = cache["text"].to(pipe.device)
    schedule, logs = [], []
    for step in range(600):
        ex = cache["train"][rng.randrange(len(cache["train"]))]
        timestep, seed = (181, 481, 781)[step % 3], 400000 + step
        target, mask = ex["target"].to(pipe.device), ex["mask"].to(pipe.device)
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed)).to(pipe.device, torch.float16)
        t = torch.tensor([timestep], device=pipe.device)
        noisy = pipe.scheduler.add_noise(target, noise, t)
        eps = predict(pipe, None, ex["tokens"]["matched"], noisy, t, text, context(pipe, ex), mask)
        predicted = patch(decode(pipe, clean_prediction(pipe, noisy, eps, timestep)), ex["roi"])
        geometry = local_losses(predicted, ex["target_patch"].to(pipe.device), ex["row"]["frequency"])
        diffusion = (eps.float()-noise.float()).square().mean()
        ori = geometry["orientation"] if ex["row"]["pattern"] == "stripe" else 0. * geometry["orientation"]
        supervised = .1*ori + .1*geometry["frequency"] + .025*geometry["peak"]
        loss = diffusion + (supervised if timestep <= 481 else 0.*supervised)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        schedule.append([ex["case"], timestep, seed])
        if step == 0 or (step+1)%25 == 0:
            row = {"step": step+1, "diffusion": float(diffusion.detach()), "active": timestep<=481,
                   **{k: float(v.detach()) for k, v in geometry.items()}, "gradient": float(norm)}
            logs.append(row)
            write_json(out/"training_partial.json", logs)
            print("[e21.2-train]", row, flush=True)
    write_json(out/"train_report.json", {"schedule": schedule, "logs": logs, "parameters": sum(p.numel() for p in parameters)})
    return schedule


def local_advantage(rows):
    from tools.e20_utilization import case_stat
    index = {(r["case"], r["seed"], r["t"], r["intervention"]): r for r in rows}
    report = {}
    for name, attr in (("wrong_orientation", "orientation"), ("wrong_period", "frequency"), ("wrong_period", "peak")):
        pairs = [(r["case"], r["local"][attr]-index[(r["case"],r["seed"],r["t"],"matched")]["local"][attr])
                 for r in rows if r["intervention"] == name and (name!="wrong_orientation" or r["pattern"]=="stripe")]
        report[name+"_"+attr] = case_stat(pairs)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root, out = Path(args.root), Path(args.root)/"e21_2/local_geometry"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"source": "E21.2 12-case train-only gradient audit", "geometry": "4x64 Sobel direction histogram + local radial/x/y FFT cosine + target fundamental peak",
                "ROI": "maximal 3:4 rectangle strictly inside 17-pixel-eroded garment mask", "timestep": "local geometry only at 181/481, 781 diffusion only",
                "weights": {"orientation_stripe": .1, "frequency_cosine": .1, "fundamental_peak": .025},
                "comparison": "same original E19 start/rank4 K/V/600steps/schedule as G0/G2; G0 reevaluated with same local measurements",
                "gate": "orientation and period matched advantage lower95CI>0; period accuracy>=G0; boundary/background upper95CI<=3%; no generation"}
    write_json(out/"protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    pipe, _, pattern, modules, _, _ = load_pipeline(root, args.device)
    pipe.vae.float()
    cache = torch.load(root/"e21/qualified_three_class/cache.pt", map_location="cpu", weights_only=False)
    scorer = TargetScorer(pattern).to(args.device).eval()
    scorer.load_state_dict(torch.load(root/"e21/qualified_three_class/scorer.pt", map_location=args.device, weights_only=False))
    adapters = {}
    for i, proc in enumerate(texture_processors(pipe.unet)):
        if proc.layer_group == "semantic":
            continue
        for key in ("to_k_ip", "to_v_ip"):
            wrapper = PatternKVLoRA(getattr(proc, key)).to(args.device)
            setattr(proc, key, wrapper)
            adapters[str(i)+key] = wrapper
    parameters = [p for m in adapters.values() for p in m.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 99840
    initial = {k: copy.deepcopy(m.state_dict()) for k, m in adapters.items()}
    frozen = {k: frozen_digest(v) for k,v in modules.items()}
    baseline = out/"G0_replay"
    baseline.mkdir(exist_ok=True)
    load_adapter(adapters, torch.load(root/"e21/qualified_three_class/A0/adapter.pt", map_location=args.device, weights_only=False))
    g0 = evaluate(pipe, scorer, cache, baseline, include_local=True)
    old = json.loads((root/"e21/qualified_three_class/A0/records.json").read_text())
    maxdiff = max(abs(r["scores"][k]-o["scores"][k]) for r,o in zip(g0,old) for k in r["scores"])
    assert maxdiff < 1e-4, maxdiff
    for key, module in adapters.items():
        module.load_state_dict(initial[key])
    folder = out/"G2_local"
    folder.mkdir(exist_ok=True)
    schedule = train(pipe, cache, parameters, folder)
    assert schedule == json.loads((root/"e21/qualified_three_class/A0/train_report.json").read_text())["schedule"]
    torch.save({"adapters": {k: {n: v for n,v in m.state_dict().items() if n.startswith("adapter_")} for k,m in adapters.items()}, "protocol": protocol}, folder/"adapter.pt")
    adapted = evaluate(pipe, scorer, cache, folder, include_local=True)
    assert all(frozen_digest(v)==frozen[k] for k,v in modules.items())
    report = {"protocol": protocol, "G0_reproduction_max_score_absdiff": maxdiff, "versus_G0": compare(adapted,g0),
              "local_advantage": local_advantage(adapted), "G0_local_advantage": local_advantage(g0),
              "freeze_pass": True, "schedule_identical": True, "generation_executed": False}
    v = report["versus_G0"]
    report["geometry_pass"] = (all(report["local_advantage"][k]["ci95"][0]>0 for k in ("wrong_orientation_orientation","wrong_period_frequency"))
                               and v["hard_metrics"]["period_correct"]["aligned"]["mean"] >= v["hard_metrics"]["period_correct"]["control"]["mean"]
                               and v["preservation"]["boundary"]["ci95"][1] <= .03 and v["preservation"]["background"]["ci95"][1] <= .03)
    report["next"] = "consider E21.3 only after independent identity-safe stage1" if report["geometry_pass"] else "stop local geometry token adaptation; no E21.3 or generation"
    write_json(out/"report.json", report)
    print("[e21.2-final]", report["geometry_pass"], flush=True)


if __name__ == "__main__":
    main()
