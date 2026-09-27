"""E21.2：仅诊断腐蚀衣身局部 geometry loss 的区分能力与梯度，不训练。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from models.local_pattern_geometry import losses as local_losses
from models.pattern_utilization import PatternKVLoRA
from models.target_pattern_score import TargetScorer, patch
from tools.e15_common import write_json
from tools.e15_stages import texture_processors
from tools.e21_target_supervision import clean_prediction, context, decode, load_pipeline, predict


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.float().square().sum()) for g in grads if g is not None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = Path(args.root)
    out = root/"e21_2"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    pipe, _, pattern, _, _, _ = load_pipeline(root, args.device)
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
    checkpoint = torch.load(root/"e21_1/G1/adapter.pt", map_location=args.device, weights_only=False)
    for key, wrapper in adapters.items():
        missing, extra = wrapper.load_state_dict(checkpoint["adapters"][key], strict=False)
        assert missing == ["base.weight"] and not extra
    parameters = [v for m in adapters.values() for v in m.parameters() if v.requires_grad]
    text = cache["text"].to(args.device)
    # 训练数据上预定义样本：每类、每频率至少一个；不使用验证标签来选择梯度权重。
    examples = []
    seen = set()
    for ex in cache["train"]:
        key = (ex["row"]["pattern"], ex["row"]["frequency"])
        if key not in seen:
            examples.append(ex)
            seen.add(key)
    rows = []
    for ex in examples:
        target = ex["target"].to(args.device)
        truth = ex["target_patch"].to(args.device)
        vae = patch(decode(pipe, target), ex["roi"])
        with torch.no_grad():
            clean_scores = local_losses(truth, truth, ex["row"]["frequency"])
            vae_scores = local_losses(vae, truth, ex["row"]["frequency"])
        sketch = context(pipe, ex)
        mask = ex["mask"].to(args.device)
        for tval in (181, 481, 781):
            seed = 510000 + ex["case"]*1000 + tval
            noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed)).to(args.device, torch.float16)
            t = torch.tensor([tval], device=args.device)
            noisy = pipe.scheduler.add_noise(target, noise, t)
            eps = predict(pipe, None, ex["tokens"]["matched"], noisy, t, text, sketch, mask)
            x0 = patch(decode(pipe, clean_prediction(pipe, noisy, eps, tval)), ex["roi"])
            score = local_losses(x0, truth, ex["row"]["frequency"])
            label = ("stripe", "plaid", "dots", "repeated_print").index(ex["row"]["pattern"])
            identity = scorer.losses(x0, truth, label)["identity"]
            diffusion = (eps.float()-noise.float()).square().mean()
            norms = {"diffusion": grad_norm(diffusion, parameters), "identity": grad_norm(identity, parameters),
                     **{k: grad_norm(v, parameters) for k, v in score.items()}}
            rows.append({"case": ex["case"], "pattern": ex["row"]["pattern"], "frequency": ex["row"]["frequency"], "t": tval,
                         "clean": {k: float(v) for k, v in clean_scores.items()}, "vae": {k: float(v) for k, v in vae_scores.items()},
                         "predicted": {k: float(v.detach()) for k, v in score.items()}, "gradients": norms})
            print("[e21.2-audit]", ex["case"], tval, norms, flush=True)
            write_json(out/"records_partial.json", rows)
    write_json(out/"records.json", rows)
    report = {"n_train_cases": len(examples), "timestep": {}, "gradient_median": {},
              "note": "interior ROI fully inside 17px erosion; local 4x64 Sobel orientation hist, FFT radial/axis cosine and fundamental peak; no training or generation"}
    for t in (181, 481, 781):
        selected = [r for r in rows if r["t"] == t]
        report["timestep"][str(t)] = {metric: {k: float(np.median([r[metric][k] for r in selected])) for k in selected[0][metric]} for metric in ("clean", "vae", "predicted", "gradients")}
    selected = [r for r in rows if r["t"] in (181, 481)]
    report["gradient_median"] = {k: float(np.median([r["gradients"][k] for r in selected])) for k in selected[0]["gradients"]}
    reference = min(report["gradient_median"]["diffusion"], report["gradient_median"]["identity"])
    report["suggested_max_weight"] = {k: min(.1, .25 * reference/max(report["gradient_median"][k], 1e-12)) for k in ("orientation", "frequency", "peak")}
    write_json(out/"report.json", report)
    print("[e21.2-final]", report["suggested_max_weight"], flush=True)


if __name__ == "__main__":
    main()
