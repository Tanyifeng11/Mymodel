"""E22.2：安全单层注入下，独立比较周期 scalar / 坐标 / sin-cos carrier。"""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from models.localized_spatial import LocalizedAdapter, LocalizedInjection
from models.periodic_carrier import estimate, field, interior_spectrum
from tools.e15_common import write_json
from tools.e19_handcrafted import rgb
from tools.e20_utilization import case_stat, frozen_digest, load_pipeline
from tools.e21_target_supervision import decode
from tools.e22_spatial import evaluate, paired_maps, summarize, train


class PeriodInputs:
    def __init__(self, estimates, mode):
        self.estimates, self.mode, self.memo = estimates, mode, {}

    def __call__(self, example, banks, variant, unused_mode):
        good = example["geometry_keys"]["matched"]
        donor = example["geometry_keys"]["wrong_period"] if variant == "wrong_period" else good
        phase = {"phase90":math.pi/2, "phase180":math.pi, "phase270":3*math.pi/2}.get(variant, 0.)
        key = good, donor, phase
        if key not in self.memo:
            self.memo[key] = field(self.estimates[good], self.estimates[donor], self.mode, phase)
        return self.memo[key]


class PeriodMeasure:
    """固定两周期候选的目标图谱读出；必须和同口径 S0 比，非 E21 ROI 准确率。"""
    def __init__(self, pipe, root, cache):
        self.pipe, self.root, self.templates = pipe, root, {}
        self.calibration = []
        for ex in cache["eval"]:
            mask = ex["mask"].to(pipe.device)
            rows = []
            for name in ("matched", "wrong_period"):
                image = rgb(Image.open(root/"e19_2_b/data"/ex["geometry_keys"][name]), pipe.device)
                rows.append(interior_spectrum(image, mask))
            self.templates[ex["case"]] = torch.cat(rows)
            clean = to_tensor(Image.open(root/("e19_2_c/cases/%02d_matched.png" % ex["case"])).convert("RGB"))[None].to(pipe.device)
            reconstructed = decode(pipe, ex["target"].to(pipe.device))
            self.calibration.append({"case":ex["case"], "pattern":ex["row"]["pattern"],
                                     "clean":self(clean, ex), "vae":self(reconstructed, ex)})

    def __call__(self, image, ex):
        vector = interior_spectrum(image, ex["mask"].to(image.device))
        similarity = vector @ self.templates[ex["case"]].T
        return {"period_correct":int(similarity.argmax(-1)) == 0,
                "target_period_loss":float(1-similarity[0,0]),
                "period_template_margin":float(similarity[0,0]-similarity[0,1])}


def accuracy(rows):
    return case_stat([(r["case"], float(r["period_correct"])) for r in rows if r["variant"]=="matched"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.root)/"e22_2"
    out.mkdir(exist_ok=True)
    safe_report = json.loads((root/"e22_1/report.json").read_text())
    assert safe_report["pass"], "E22.1 safety must pass before period training"
    safe = torch.load(root/"e22_1/safe_orientation.pt", map_location="cpu", weights_only=False)
    assert safe["policy"] in ("post", "eroded", "feather")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    pipe, _, _, modules, _, _ = load_pipeline(root, "cuda:0")
    pipe.vae.float()
    frozen = {k:frozen_digest(m) for k,m in modules.items()}
    cache = torch.load(root/"e20/cache.pt", map_location="cpu", weights_only=False)
    cache["root"] = str(root)
    banks = paired_maps(root, cache, root/"e22", pipe.device)
    for split in ("train", "eval"):
        for ex in cache[split]:
            ex["spatial_gate"] = ex["mask"].float()
    estimates = {}
    for key in banks:
        estimates[key] = estimate(rgb(Image.open(root/"e19_2_b/data"/key), pipe.device))
    input_rows = []
    for ex in cache["eval"]:
        q = estimates[ex["geometry_keys"]["matched"]]
        active = q["active"].float()
        frequency = float((q["frequency"]*active).sum()/active.sum().clamp_min(1))
        input_rows.append({"case":ex["case"], "predicted":frequency, "expected":ex["row"]["frequency"],
                           "correct":abs(frequency-ex["row"]["frequency"])<.5, "active":active.tolist()})
    write_json(out/"input_audit.json", input_rows)
    assert all(r["correct"] for r in input_rows), "Reference period positive control must be correct"
    protocol = {"steps":600, "seed":42, "training_schedule_seed":2224, "channels":5,
                "groups":["P0_scalar", "P1_coordinates", "P2_carrier"],
                "safe_policy":safe["policy"], "site":safe["site"], "cases":32,
                "seeds":[42,43], "timesteps":[181,481,781],
                "coordinates":"full-canvas fractions, matching original E20 target resize; carrier sampled at 32x32 injection resolution",
                "condition":"all appearance/identity/geometry tokens fixed matched; only spatial period changed; correct orientation and active axes held fixed",
                "period_measure":"closed-set matched/wrong frequency identification by interior power-spectrum templates, all 32 cases; calibrated clean/VAE before training; distinct from E21 ROI absolute period accuracy",
                "selection":"P0/P1/P2 all fixed budget; P2 is the predeclared carrier hypothesis, always evaluated with all three phase controls",
                "phase_gate":"every pi/2,pi,3pi/2 phase penalty smaller than wrong-frequency penalty with case-bootstrap CI>0; phase mean <= half wrong-frequency mean",
                "generation":False, "joint":False}
    write_json(out/"protocol.json", protocol)
    with torch.no_grad():
        measure = PeriodMeasure(pipe, root, cache)
    write_json(out/"period_measure_calibration.json", measure.calibration)
    calibrated = all(sum(r[s]["period_correct"] for r in measure.calibration)/32 >= .9 for s in ("clean", "vae"))
    if not calibrated:
        write_json(out/"report.json", {"stage":"measurement", "pass":False, "training_executed":False,
                                      "next":"repair target period measurement before judging utilization"})
        return
    baseline_dir = out/"S0"
    baseline_dir.mkdir(exist_ok=True)
    baseline = (json.loads((baseline_dir/"records.json").read_text()) if (baseline_dir/"records.json").exists()
                else evaluate(pipe, None, None, cache, banks, baseline_dir, "base", ("matched",), measure=measure))
    reports, schedules = {}, {}
    for name, mode in (("P0_scalar", "scalar"), ("P1_coordinates", "coordinates"), ("P2_carrier", "carrier")):
        folder = out/name
        folder.mkdir(exist_ok=True)
        torch.manual_seed(42)
        adapter = LocalizedAdapter(channels=640, input_channels=5, policy=safe["policy"]).to(pipe.device)
        injection = LocalizedInjection(pipe.unet, adapter, safe["site"])
        selector = PeriodInputs(estimates, mode)
        if (folder/"adapter.pt").exists():
            adapter.load_state_dict(torch.load(folder/"adapter.pt", map_location=pipe.device, weights_only=False)["adapter"])
            schedules[name] = json.loads((folder/"train_report.json").read_text())["schedule"]
        else:
            schedules[name] = train(pipe, injection, adapter, cache, banks, folder, "period", selector=selector)
        records = (json.loads((folder/"records.json").read_text()) if (folder/"records.json").exists()
                   else evaluate(pipe, injection, adapter, cache, banks, folder, "period", ("matched", "wrong_period"), selector, measure))
        report = summarize(records, baseline, ("wrong_period",))
        report["period_accuracy"] = accuracy(records)
        report["baseline_period_accuracy"] = accuracy(baseline)
        base_lookup = {(r["case"],r["seed"],r["t"]):r for r in baseline}
        report["period_accuracy_change"] = case_stat([(r["case"],float(r["period_correct"])-float(base_lookup[(r["case"],r["seed"],r["t"])]["period_correct"])) for r in records if r["variant"]=="matched"])
        report["period_nondegrading"] = report["period_accuracy"]["mean"] >= report["baseline_period_accuracy"]["mean"]
        lookup = {(r["case"],r["seed"],r["t"]):r for r in records if r["variant"]=="matched"}
        report["target_period_advantage"] = case_stat([(r["case"],r["target_period_loss"]-lookup[(r["case"],r["seed"],r["t"])]["target_period_loss"]) for r in records if r["variant"]=="wrong_period"])
        report["pass"] &= report["period_nondegrading"]
        reports[name] = report
        injection.close()
        assert all(frozen_digest(m)==frozen[k] for k,m in modules.items())
        write_json(out/"report_partial.json", {"groups":reports})
        print("[e22.2-group]", name, json.dumps(report), flush=True)
    assert schedules["P0_scalar"] == schedules["P1_coordinates"] == schedules["P2_carrier"]
    # P4 is required for the explicit carrier even if its primary gate already fails.
    name = "P2_carrier"
    folder = out/"P2_phase_control"
    folder.mkdir(exist_ok=True)
    adapter = LocalizedAdapter(channels=640, input_channels=5, policy=safe["policy"]).to(pipe.device)
    adapter.load_state_dict(torch.load(out/name/"adapter.pt", map_location=pipe.device, weights_only=False)["adapter"])
    adapter.eval().requires_grad_(False)
    injection = LocalizedInjection(pipe.unet, adapter, safe["site"])
    phases = evaluate(pipe, injection, adapter, cache, banks, folder, "period", ("phase90","phase180","phase270"), PeriodInputs(estimates,"carrier"), measure)
    injection.close()
    records = json.loads((out/name/"records.json").read_text())
    bykey = {(r["case"],r["seed"],r["t"],r["variant"]):r for r in records}
    phase_reports = {}
    wrong_mean = reports[name]["advantage"]["wrong_period"]["mean"]
    for variant in ("phase90","phase180","phase270"):
        selected = [r for r in phases if r["variant"]==variant]
        penalty = case_stat([(r["case"],r["epsilon"]["interior"]-bykey[(r["case"],r["seed"],r["t"],"matched")]["epsilon"]["interior"]) for r in selected])
        separation = case_stat([(r["case"],bykey[(r["case"],r["seed"],r["t"],"wrong_period")]["epsilon"]["interior"]-r["epsilon"]["interior"]) for r in selected])
        phase_reports[variant] = {"penalty":penalty, "wrong_minus_phase":separation,
                                  "pass":separation["ci95"][0]>0 and penalty["mean"]<=.5*wrong_mean}
    phase_pass = all(r["pass"] for r in phase_reports.values())
    assert all(frozen_digest(m)==frozen[k] for k,m in modules.items())
    result = {"groups":reports, "phase_control":phase_reports, "phase_pass":phase_pass,
              "pass":reports[name]["pass"] and phase_pass, "schedule_identical":True,
              "frozen_pass":True, "orientation_checkpoint_unchanged":True,
              "generation_executed":False, "joint_executed":False}
    result["next"] = "E22.3 joint only after independent carrier and phase gates" if result["pass"] else "stop additive period encoding; inspect frequency-modulated synthesis rather than more encoding variants"
    write_json(out/"report.json", result)
    print("[e22.2-final]", result["pass"], result["next"], flush=True)


if __name__ == "__main__":
    main()
