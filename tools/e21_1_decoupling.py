"""E21.1：与既有 A0/A1 同起点同预算，新增 identity-only 和 geometry-only 两臂。"""

import argparse
import copy
import json
from pathlib import Path

import torch

from models.pattern_utilization import PatternKVLoRA
from models.target_pattern_score import TargetScorer
from tools.e15_common import write_json
from tools.e15_stages import texture_processors
from tools.e20_utilization import frozen_digest, load_pipeline
from tools.e21_target_supervision import compare, evaluate, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = Path(args.root)
    source = root/"e21/qualified_three_class"
    out = root/"e21_1"
    out.mkdir(exist_ok=True)
    protocol = {"arms": {"G0": "existing E21 A0 diffusion only", "G1": "diffusion + identity",
                         "G2": "diffusion + orientation + period", "G3": "existing E21 A1 diffusion + all three"},
                "budget": "600 steps, same 264 train cases, same random/noise/timestep schedule, same rank4 texture K/V LoRA and original E19 start",
                "scorer": "reuse E21 qualified-three-class scorer and exact 22 held-out cases; no retraining head",
                "holdout": "frequency and phase held out; 8 stripe direction cases, 22 identity/period cases",
                "safety": "boundary/background RGB relative upper95CI <=3%; IoU/EdgeF1 lower95CI >=-0.02; leakage upper95CI<=0.02",
                "selection": "fixed 600-step checkpoint; no weight sweep, no generation"}
    write_json(out/"protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    pipe, bf, pattern, modules, width, height = load_pipeline(root, args.device)
    pipe.vae.float()
    cache = torch.load(source/"cache.pt", map_location="cpu", weights_only=False)
    scorer = TargetScorer(pattern).to(args.device).eval()
    scorer.load_state_dict(torch.load(source/"scorer.pt", map_location=args.device, weights_only=False))
    assert json.loads((source/"scorer_audit.json").read_text())["pass"]
    adapters = {}
    for i, proc in enumerate(texture_processors(pipe.unet)):
        if proc.layer_group == "semantic":
            continue
        for key in ("to_k_ip", "to_v_ip"):
            wrapper = PatternKVLoRA(getattr(proc, key)).to(args.device)
            setattr(proc, key, wrapper)
            adapters[str(i)+key] = wrapper
    params = [p for m in adapters.values() for p in m.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == 99840
    initial = {k: copy.deepcopy(m.state_dict()) for k, m in adapters.items()}
    frozen = {k: frozen_digest(v) for k, v in modules.items()}
    records = {"G0": json.loads((source/"A0/records.json").read_text()),
               "G3": json.loads((source/"A1/records.json").read_text())}
    common = json.loads((source/"A0/train_report.json").read_text())["schedule"]
    assert common == json.loads((source/"A1/train_report.json").read_text())["schedule"]
    for name, components in (("G1", ("identity",)), ("G2", ("orientation", "period"))):
        folder = out/name
        folder.mkdir(exist_ok=True)
        for key, module in adapters.items():
            module.load_state_dict(initial[key])
        torch.manual_seed(42)
        schedule = train(pipe, scorer, params, cache, True, 600, folder, components)
        assert schedule == common
        torch.save({"adapters": {k: {n: v for n, v in m.state_dict().items() if n.startswith("adapter_")} for k, m in adapters.items()},
                    "protocol": protocol, "components": components}, folder/"adapter.pt")
        records[name] = evaluate(pipe, scorer, cache, folder)
        assert all(frozen_digest(v) == frozen[k] for k, v in modules.items())
    result = {"protocol": protocol, "arms": {}, "freeze_pass": True, "schedule_identical": True, "generation_executed": False}
    for name in ("G1", "G2", "G3"):
        result["arms"][name] = compare(records[name], records["G0"])
    result["G1_identity_advantage"] = result["arms"]["G1"]["target_advantage"]["identity"]
    result["G1_identity_pass"] = result["G1_identity_advantage"]["ci95"][0] > 0
    result["G1_period_not_worse"] = result["arms"]["G1"]["hard_metrics"]["period_correct"]["aligned"]["mean"] >= result["arms"]["G1"]["hard_metrics"]["period_correct"]["control"]["mean"]
    result["G1_boundary_safe"] = result["arms"]["G1"]["preservation"]["boundary"]["ci95"][1] <= .03
    result["identity_clean"] = all(result[k] for k in ("G1_identity_pass", "G1_period_not_worse", "G1_boundary_safe"))
    result["G2_geometry_harm"] = {"period_accuracy_drop": result["arms"]["G2"]["hard_metrics"]["period_correct"]["aligned"]["mean"] < result["arms"]["G2"]["hard_metrics"]["period_correct"]["control"]["mean"],
                                  "boundary_worse": result["arms"]["G2"]["preservation"]["boundary"]["ci95"][0] > .03}
    result["next"] = "E21.2 geometry redesign diagnostic" if result["identity_clean"] else "report objective attribution; do not assume identity-only is safe"
    write_json(out/"report.json", result)
    print("[e21.1-final]", result["identity_clean"], result["G2_geometry_harm"], flush=True)


if __name__ == "__main__":
    main()
