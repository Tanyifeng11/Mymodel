"""E22.1: frozen orientation positive control, mask then single-layer location audit."""

import argparse
import json
from pathlib import Path

import torch

from models.localized_spatial import LocalizedAdapter, LocalizedInjection
from tools.e15_common import write_json
from tools.e20_utilization import frozen_digest, load_pipeline
from tools.e22_spatial import evaluate, paired_maps, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = root/"e22_1"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    pipe, _, _, modules, _, _ = load_pipeline(root, "cuda:0")
    pipe.vae.float()
    frozen = {k: frozen_digest(m) for k,m in modules.items()}
    cache = torch.load(root/"e20/cache.pt", map_location="cpu", weights_only=False)
    cache["root"] = str(root)
    banks = paired_maps(root, cache, root/"e22", pipe.device)
    # Existing evaluator passes this field to injection; use the full mask here.
    for split in ("train", "eval"):
        for example in cache[split]:
            example["spatial_gate"] = example["mask"].float()
    baseline = json.loads((root/"e22/S0/records.json").read_text())
    original = json.loads((root/"e22/S2_orientation/records.json").read_text())
    state = torch.load(root/"e22/S2_orientation/adapter.pt", map_location="cpu", weights_only=False)["adapter"]
    reports = {"O0_original": summarize(original, baseline, ("wrong_orientation",))}
    candidates = []
    configs = {}

    def run(name, policy, site="up_blocks.2.resnets.0", control=None):
        folder = output/name
        folder.mkdir(exist_ok=True)
        adapter = LocalizedAdapter(channels=640, input_channels=3, policy=policy).to(pipe.device)
        adapter.load_state_dict(state, strict=True)
        adapter.eval().requires_grad_(False)
        before = frozen_digest(adapter)
        injection = LocalizedInjection(pipe.unet, adapter, site, control)
        if (folder/"records.json").exists():
            records = json.loads((folder/"records.json").read_text())
        else:
            records = evaluate(pipe, injection, adapter, cache, banks, folder,
                               "orientation", ("matched", "wrong_orientation"))
        injection.close()
        assert frozen_digest(adapter) == before
        reports[name] = summarize(records, baseline, ("wrong_orientation",))
        configs[name] = {"policy":policy, "site":site, "control":control}
        write_json(output/"report_partial.json", {"groups":reports, "configs":configs})
        print("[e22.1]", name, json.dumps(reports[name]), flush=True)
        return reports[name]["pass"]

    # Fixed checkpoint and gate thresholds; no retraining or loss/strength search.
    for name, policy in (("O1_input", "input"), ("O2_post", "post"),
                         ("O3_eroded", "eroded"), ("O4_feather", "feather")):
        if run(name, policy):
            candidates.append(name)
    if not candidates:
        # These sites all have 640 channels. No learned channel remapping and no
        # parameter-count confound when moving the exact same frozen adapter.
        for name, site in (("L_mid", "down_blocks.1.resnets.1"),
                           ("L_late_up", "up_blocks.2.upsamplers.0")):
            if run(name, "eroded", site):
                candidates.append(name)
    selected = candidates[0] if candidates else None
    if selected:
        cfg = configs[selected]
        for control in ("constant", "zero"):
            name = selected+"_"+control
            run(name, cfg["policy"], cfg["site"], control)
            advantage = reports[name]["advantage"]["wrong_orientation"]
            assert abs(advantage["mean"]) < 1e-8, "Control must have identical correct/wrong inputs"
        torch.save({"adapter":state, **cfg, "source":"e22/S2_orientation/adapter.pt"}, output/"safe_orientation.pt")
    assert all(frozen_digest(m)==frozen[k] for k,m in modules.items())
    result = {"experiment":"E22.1", "groups":reports, "configs":configs,
              "selected":selected, "pass":selected is not None, "frozen_pass":True,
              "training_steps":0, "cases":32, "orientation_cases":8,
              "seeds":[42,43], "timesteps":[181,481,781],
              "selection":"first passing predeclared group; exploratory fixed-set comparison",
              "safety_gate":"95% CI upper relative boundary/background <=3%; IoU/EdgeF1 lower >=-0.02; leakage upper<=0.02",
              "next":"E22.2 independent period carrier" if selected else "stop; no safe orientation injection established",
              "period_executed":False, "joint_executed":False, "generation_executed":False}
    write_json(output/"report.json", result)
    print("[e22.1-final]", selected, flush=True)


if __name__ == "__main__":
    main()
