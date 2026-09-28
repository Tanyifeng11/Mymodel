"""E23-M：共同状态、单步注入、自由轨迹和条件统计；无训练。"""
import argparse
import gc
import json
import math
import shutil
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import build_region_masks
from tools import e22_o4_generation as gen
from tools.e15_common import write_json
from tools.e15_stages import texture_processors
from tools.e18_1_clean_patterns import make_image, PALETTES
from tools.e19_handcrafted import text_tokens
from tools.e20_utilization import case_stat, load_pipeline
from tools.e22_4_generation import native_pipeline, module_hashes, paired
from tools.e22_4_genealogy import file_hash
from tools.e22_o4_metrics import measure, orientation, axial_distance

ARMS = ("E5", "E17_direct", "S0")
POSITIONS = (0, 5, 10, 16, 22, 28, 33, 39, 44, 49)
DECODE_POSITIONS = (0, 10, 22, 33, 49)
REGIONS = ("interior", "boundary", "background")
WINDOWS = {"high": (0, 17), "middle": (17, 34), "low": (34, 50)}


def tree(value, device="cpu"):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree(v, device) for v in value)
    return value


def rms(value, mask=None):
    x = value.float()
    if mask is None:
        return float(x.square().mean().sqrt())
    return float(((x.square()*mask).sum()/(mask.sum()*x.shape[1]).clamp_min(1e-12)).sqrt())


def regional(value, masks):
    return {r: rms(value, m) for r, m in masks.items()}


def prepare_cases(root, out):
    path = out/"cases.json"
    if path.exists():
        return json.loads(path.read_text())
    source = root/"e22_o4_generation"
    old = json.loads((source/"cases.json").read_text())
    shutil.copytree(source/"inputs", out/"inputs", dirs_exist_ok=True)
    refs = list(old["references"])
    for f in (6, 7, 9, 11, 13, 15):
        for phase in (.23, .47, .71, .89):
            i = len(refs)
            image = make_image(f, phase, PALETTES[i % 4])
            if i % 2:
                image = image.transpose(Image.Transpose.ROTATE_90)
            variants = []
            for name, im in (("original", image), ("rot90", image.transpose(Image.Transpose.ROTATE_90))):
                relative = "inputs/ref%02d_%s.png" % (i, name)
                im.save(out/relative)
                variants.append({"variant": name, "path": relative, "sha256": gen.sha(im.tobytes()),
                                 "theta": float((90 if i % 2 == 0 else 0)+(90 if name == "rot90" else 0)) % 180})
            refs.append({"id": i, "frequency": f, "phase": phase, "palette": i % 4, "variants": variants})
    result = {"references": refs, "sketches": old["sketches"],
              "case_definition": "one reference pair and fixed sketch id=reference id modulo 2; bootstrap reference case",
              "pilot": 8, "confirmation": 32, "synthetic": True, "selected_before_outputs": True}
    write_json(path, result)
    return result


def selected(cases, start, stop):
    for ref in cases["references"][start:stop]:
        yield ref, cases["sketches"][ref["id"] % 2]


def processor_config(pipe):
    return [{"class": type(p).__name__, "group": p.layer_group,
             **{k: getattr(p, k) for k in ("use_texture_gate", "gate_min", "gate_max",
                                          "use_balanced_fusion_gate", "use_conflict_aware_gate")}}
            for p in texture_processors(pipe.unet)]


def normalize_scheduler(pipe):
    config = json.loads(json.dumps(dict(pipe.scheduler.config)))
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    return config


@torch.inference_mode()
def token_bank(pipe, cases, out, width, height, arm, bf=None, pattern=None):
    result = {}
    if arm == "S0":
        negative = gen.current_null(pipe, bf, width, height)
    else:
        encoded = pipe.encode_prompt("a cloth", pipe.device, 1, True, " worst quality, low quality",
                                     return_text_masks=True)
    for ref in cases["references"]:
        for v in ref["variants"]:
            image = Image.open(out/v["path"]).convert("RGB")
            if arm == "S0":
                pair = gen.current_tokens(pipe, bf, pattern, image, width, height), negative
            else:
                pair = pipe.get_image_embeds(pil_image=image, width=width, height=height,
                    texture_mode="patch_resampled", text_embeds=encoded[0], negative_text_embeds=encoded[1],
                    text_mask=encoded[2], negative_text_mask=encoded[3],
                    aa_tcr_captions="a cloth", aa_tcr_negative_captions=" worst quality, low quality")
            result[ref["id"], v["variant"]] = tree(pair)
    return result


class Trace:
    """只观察现有 pipeline，不重写采样过程；DDIM eta=0 无历史状态。"""
    def __init__(self, pipe):
        self.pipe = pipe
        self.latents, self.epsilon, self.conditional, self.unconditional = [], [], [], []
        self.context = {}
        self.step = pipe.scheduler.step
        self.pre = pipe.unet.register_forward_pre_hook(self.before, with_kwargs=True)
        self.post = pipe.unet.register_forward_hook(self.after, with_kwargs=True)
        pipe.scheduler.step = self.update

    def before(self, module, args, kwargs):
        self.branch = "conditional" if "sa_hidden_states" in kwargs["cross_attention_kwargs"] else "unconditional"
        if self.branch not in self.context:
            ctx = dict(kwargs)
            # 保留真正的 pipeline 文本前缀；换 condition 只拼接 texture tokens。
            ctx["encoder_hidden_states"] = ctx["encoder_hidden_states"][:, :77]
            self.context[self.branch] = tree(ctx)

    def after(self, module, args, kwargs, output):
        getattr(self, self.branch).append(output[0].detach().cpu())

    def update(self, model_output, timestep, sample, **kwargs):
        self.latents.append(sample.detach().cpu())
        self.epsilon.append(model_output.detach().cpu())
        result = self.step(model_output, timestep, sample, **kwargs)
        self.final = result[0].detach().cpu()
        return result

    def close(self):
        self.pipe.scheduler.step = self.step
        self.pre.remove()
        self.post.remove()

    def result(self):
        assert len(self.latents) == len(self.conditional) == len(self.unconditional) == 50
        return {"latent": torch.stack(self.latents+[self.final]), "epsilon": torch.stack(self.epsilon),
                "conditional": torch.stack(self.conditional), "unconditional": torch.stack(self.unconditional),
                "context": self.context, "timesteps": self.pipe.scheduler.timesteps.cpu()}


@torch.inference_mode()
def generate(pipe, arm, ref, sk, seed, v, bank, out, width, height, trace=True, window=None):
    folder = out/arm
    folder.mkdir(exist_ok=True)
    name = "c%02d_s%d_%s" % (ref["id"], seed, v["variant"])
    metadata, trace_path = folder/(name+".json"), folder/(name+".pt")
    if metadata.exists() and (not trace or trace_path.exists()):
        return json.loads(metadata.read_text())
    mask_image = Image.open(out/sk["mask"]).convert("L")
    sketch = Image.open(out/sk["path"]).convert("RGB")
    image = Image.open(out/v["path"]).convert("RGB")
    mask = to_tensor(mask_image)[None].to(pipe.device, torch.float16)
    n = 24 if arm == "S0" else 16
    pair = tree(bank[ref["id"], v["variant"]], pipe.device)
    old_get = pipe.get_image_embeds
    pipe.get_image_embeds = MethodType(lambda self, **kwargs: pair, pipe)
    recording = Trace(pipe) if trace else None
    intervention = None
    if window:
        # 原 E5 sampler、sketch 和 CFG 不变；只在指定窗口替换正/负 texture 条件。
        other = tree(window["bank"][ref["id"], v["variant"]], pipe.device)
        calls = [0]
        def switch(module, args, kwargs):
            step = calls[0] // 2
            branch = 0 if "sa_hidden_states" in kwargs["cross_attention_kwargs"] else 1
            calls[0] += 1
            if window["range"][0] <= step < window["range"][1]:
                kwargs = dict(kwargs)
                kwargs["encoder_hidden_states"] = torch.cat([kwargs["encoder_hidden_states"][:, :77], other[branch]], 1)
            return args, kwargs
        intervention = pipe.unet.register_forward_pre_hook(switch, with_kwargs=True)
    try:
        generated = pipe(prompt="a cloth", null_prompt="", negative_prompt=" worst quality, low quality",
            ref_image=to_tensor(sketch)[None]*2-1, texture_clip_image=image,
            width=width, height=height, num_inference_steps=50, guidance_scale=7., sketch_scale=.6,
            ipa_scale=1., texture_mode="patch_resampled", texture_condition_mode="token",
            texture_preprocess_mode="plain_resize", texture_num_tokens=n, force_texture_num_tokens_override=n != 16,
            texture_scale=1., spatial_mask=mask,
            generator=torch.Generator(device=pipe.device).manual_seed(seed))[0]
        generated.save(folder/(name+".png"))
        row = {"arm": arm, "case": ref["id"], "reference": ref["id"], "sketch": sk["id"],
               "seed": seed, "variant": v["variant"], "path": str((folder/(name+".png")).relative_to(out)),
               **measure(generated, mask_image, sketch, image, v["theta"], sk["roi"])}
        if recording:
            record = recording.result()
            torch.save(record, trace_path)
            row["initial_latent_sha256"] = gen.sha(record["latent"][0].numpy().tobytes())
            # Pilot 必须复现 E22.4 同 case、seed、variant 的实际图。
            if ref["id"] < 8:
                previous = out.parent/"e22_o4_generation"/arm if arm != "E17_direct" else out.parent/"e22_4"/arm
                previous /= "r%02d_k%d_s%d_%s.png" % (ref["id"], sk["id"], seed, v["variant"])
                row["previous_pixel_exact"] = bool(np.array_equal(np.array(generated), np.array(Image.open(previous))))
                assert row["previous_pixel_exact"], "Token bank / pipeline changed; repair before interpreting mechanism"
        write_json(metadata, row)
        print("[e23-generation]", arm, name, flush=True)
        return row
    finally:
        pipe.get_image_embeds = old_get
        if recording:
            recording.close()
        if intervention:
            intervention.remove()


def latent_regions(out, sk, shape, device):
    mask = to_tensor(Image.open(out/sk["mask"]).convert("L"))[None]
    return {r: F.interpolate(m.to(device), shape, mode="area") for r, m in zip(REGIONS, build_region_masks(mask, 17))}


class AttentionStats:
    def __init__(self, pipe):
        self.data, self.handles = {}, []
        self.processors = texture_processors(pipe.unet)
        for i, p in enumerate(self.processors):
            self.data[i] = {}
            for name in ("k", "v"):
                def hook(module, args, output, i=i, name=name):
                    self.data[i][name+"_rms"] = rms(output)
                    self.data[i][name+"_norm"] = float(output.float().norm(dim=-1).mean())
                self.handles.append(getattr(p, "to_"+name+"_ip").register_forward_hook(hook))
            p.texture_attention_observer = lambda x, i=i: self.data[i].update(attention_output_rms=rms(x))
            p.texture_probe_observer = lambda x, i=i: self.data[i].update(texture_residual_rms=rms(x))

    def reset(self):
        for d in self.data.values():
            d.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        for p in self.processors:
            del p.texture_attention_observer
            del p.texture_probe_observer


@torch.inference_mode()
def forward(pipe, z, t, context, pair, disabled=False, probe=None):
    n = pair[0].shape[1]
    for p in texture_processors(pipe.unet):
        p.num_tokens, p.use_ip_adapter, p.scale = n, True, 0. if disabled else 1.
    outputs, stats = [], {}
    for b, branch in enumerate(("conditional", "unconditional")):
        kwargs = dict(context[branch])
        kwargs["encoder_hidden_states"] = torch.cat([kwargs["encoder_hidden_states"], pair[b]], 1)
        if probe:
            probe.reset()
        outputs.append(pipe.unet(z, t, **kwargs)[0])
        if probe:
            stats[branch] = tree(probe.data)
    return outputs[1]+7*(outputs[0]-outputs[1]), outputs[0], outputs[1], stats


@torch.inference_mode()
def decoded_metrics(pipe, z, eps, t, ref, sk, out):
    alpha = pipe.scheduler.alphas_cumprod[int(t)].to(z.device)
    clean = (z.float()-(1-alpha).sqrt()*eps.float())/alpha.sqrt()
    pixels = pipe.vae.decode((clean/pipe.vae.config.scaling_factor).to(pipe.vae.dtype), return_dict=False)[0]
    array = (pixels/2+.5).clamp(0, 1).float().cpu()[0].permute(1, 2, 0).numpy()
    image = Image.fromarray((array*255).round().astype(np.uint8))
    mask = Image.open(out/sk["mask"]).convert("L")
    sketch = Image.open(out/sk["path"]).convert("RGB")
    reference = Image.open(out/ref["variants"][0]["path"]).convert("RGB")
    result = measure(image, mask, sketch, reference, ref["variants"][0]["theta"], sk["roi"])
    regions = dict(zip(REGIONS, build_region_masks(to_tensor(mask)[None], 17)))
    bg = regions["background"]
    boundary = regions["boundary"]
    result["boundary_error_vs_E5_final"] = None
    result = {k: result[k] for k in ("sketch_iou", "edge_f1", "leakage", "background_white_mae", "direction")}
    # x0 无真实 target，boundary 使用相对健康 E5 final 的像素误差，明确记录代理定义。
    healthy = Image.open(out/"E5"/("c%02d_s%d_original.png" % (ref["id"], pipe._e23_seed))).convert("RGB")
    difference = (to_tensor(image)[None]-to_tensor(healthy)[None]).square()
    result["boundary_mse_vs_E5_final"] = float((difference*boundary).sum()/(boundary.sum()*3))
    return result


@torch.inference_mode()
def common_state(pipe, banks, cases, out, start, stop):
    folder = out/"common_state"
    folder.mkdir(exist_ok=True)
    pipe.scheduler.set_timesteps(50, device=pipe.device)
    probe = AttentionStats(pipe)
    try:
        for ref, sk in selected(cases, start, stop):
            case = ref["id"]
            for seed in gen.SEEDS:
                destination = folder/("c%02d_s%d.json" % (case, seed))
                if destination.exists():
                    continue
                base = torch.load(out/"E5"/("c%02d_s%d_original.pt" % (case, seed)), map_location="cpu", weights_only=False)
                trajectories = {arm: torch.load(out/arm/("c%02d_s%d_original.pt" % (case, seed)), map_location="cpu", weights_only=False) for arm in ARMS}
                context = tree(base["context"], pipe.device)
                masks = latent_regions(out, sk, base["latent"].shape[-2:], pipe.device)
                pipe._e23_seed = seed
                rows, free = [], []
                for i in range(51):
                    f = {a: regional(trajectories[a]["latent"][i].to(pipe.device)-base["latent"][i].to(pipe.device), masks) for a in ARMS}
                    free.append({"step_index": i, "F": f})
                for i in POSITIONS:
                    z = base["latent"][i].to(pipe.device)
                    t = pipe.scheduler.timesteps[i]
                    alpha = float(pipe.scheduler.alphas_cumprod[int(t)])
                    predictions = {}
                    for arm in ARMS:
                        pair = tree(banks[arm][case, "original"], pipe.device)
                        rot = tree(banks[arm][case, "rot90"], pipe.device)
                        full = forward(pipe, z, t, context, pair, probe=probe)
                        off = forward(pipe, z, t, context, pair, disabled=True)
                        rotation = forward(pipe, z, t, context, rot)
                        predictions[arm] = full, off, rotation
                    assert rms(predictions["E5"][0][0]-base["epsilon"][i].to(pipe.device)) < 1e-5, "Replay is not baseline-exact"
                    off_error = {a: float((predictions[a][1][0]-predictions["E5"][1][0]).abs().max()) for a in ARMS}
                    assert max(off_error.values()) < 1e-4, "no_tex differs: text/sketch contaminates comparison"
                    row = {"case": case, "seed": seed, "step_index": i, "timestep": int(t),
                           "sigma_definition": "sqrt((1-alpha_cumprod)/alpha_cumprod), not DDIM stochastic eta",
                           "sigma": math.sqrt((1-alpha)/alpha), "logSNR": math.log(alpha/(1-alpha)),
                           "no_tex_max_abs_difference": off_error, "models": {}}
                    e5_tex = predictions["E5"][0][0]-predictions["E5"][1][0]
                    e5_tex_cond = predictions["E5"][0][1]-predictions["E5"][1][1]
                    e5_score = regional(predictions["E5"][0][0], masks)
                    e5_tex_rms = regional(e5_tex, masks)
                    for arm, (full, off, rotation) in predictions.items():
                        delta, delta_cond = full[0]-off[0], full[1]-off[1]
                        D = regional(delta-e5_tex, masks)
                        strength = regional(delta, masks)
                        update = pipe.scheduler.step(full[0], t, z, eta=0., return_dict=False)[0]
                        injection = regional(update-base["latent"][i+1].to(pipe.device), masks)
                        accumulated = free[i]["F"][arm]
                        stats = {"texture_rms": strength, "D": D,
                                 "D_over_E5_score_rms": {r: D[r]/max(e5_score[r], 1e-8) for r in REGIONS},
                                 "D_over_E5_texture_rms": {r: D[r]/max(e5_tex_rms[r], 1e-8) for r in REGIONS},
                                 "LR": strength["background"]/max(strength["interior"], 1e-8),
                                 "rotation_rms": regional(rotation[0]-full[0], masks),
                                 "conditional_texture_rms": regional(delta_cond, masks),
                                 "conditional_D": regional(delta_cond-e5_tex_cond, masks),
                                 "conditional_rotation_rms": regional(rotation[1]-full[1], masks),
                                 "I": injection, "F": accumulated,
                                 "A": {r: accumulated[r]/max(injection[r], 1e-8) for r in REGIONS},
                                 "attention": full[3]}
                        if i in DECODE_POSITIONS:
                            stats["common_x0"] = decoded_metrics(pipe, z, full[0], t, ref, sk, out)
                            own_z, own_eps = trajectories[arm]["latent"][i].to(pipe.device), trajectories[arm]["epsilon"][i].to(pipe.device)
                            stats["free_x0"] = decoded_metrics(pipe, own_z, own_eps, t, ref, sk, out)
                        row["models"][arm] = stats
                    rows.append(row)
                write_json(destination, {"common_state": rows, "free_trajectory": free})
                print("[e23-common]", case, seed, flush=True)
    finally:
        probe.close()


def condition_statistics(banks, count):
    # Token/channel 子空间避免把 S0 的24tokens硬对齐成E5的16tokens。
    # 每次排除同一reference的original/rot90，避免训练子空间自重建偏差。
    records = []
    for case in range(count):
        fit = torch.cat([banks["E5"][c, v][0][0].float() for c in range(count) if c != case for v in ("original", "rot90")])
        mean = fit.mean(0)
        _, singular, vh = torch.linalg.svd(fit-mean, full_matrices=False)
        n = int(torch.searchsorted(singular.square().cumsum(0)/singular.square().sum(), .99))+1
        basis = vh[:n]
        for arm in ARMS:
            for variant in ("original", "rot90"):
                x = banks[arm][case, variant][0][0].float()
                normalized = F.normalize(x, dim=-1)
                cos = normalized@normalized.T
                s = torch.linalg.svdvals(x-x.mean(0))
                p = s.square()/s.square().sum().clamp_min(1e-12)
                center = x-mean
                projected = center@basis.T@basis+mean
                records.append({"case": case, "arm": arm, "variant": variant, "token_count": len(x),
                    "rms": rms(x), "channel_mean": x.mean(0).tolist(), "channel_std": x.std(0).tolist(),
                    "pairwise_cosine": float(cos[torch.triu_indices(len(x), len(x), 1).unbind()].mean()),
                    "effective_rank": float(torch.exp(-(p[p>0]*p[p>0].log()).sum())),
                    "empirical_E5_affine_OOR": float((x-projected).norm()/x.norm().clamp_min(1e-8)),
                    "subspace_rank": n})
    return {"definition": "leave-one-reference-out E5 affine token/channel PCA, 99% centered variance; not full UNet training distribution", "records": records}


def summarize(cases, out, count):
    rows, free = [], []
    for case in range(count):
        for seed in gen.SEEDS:
            data = json.loads((out/"common_state"/("c%02d_s%d.json" % (case, seed))).read_text())
            rows.extend(data["common_state"])
            free.extend({"case": case, "seed": seed, **r} for r in data["free_trajectory"])
    stat = lambda values: case_stat(values)
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for metric in ("D", "texture_rms", "rotation_rms", "I", "F", "A", "D_over_E5_score_rms", "D_over_E5_texture_rms"):
            groups[arm][metric] = {r: stat([(v["case"], v["models"][arm][metric][r]) for v in rows]) for r in REGIONS}
        groups[arm]["LR"] = stat([(v["case"], v["models"][arm]["LR"]) for v in rows])
        groups[arm]["by_step"] = {str(i): {m: {r: stat([(v["case"], v["models"][arm][m][r]) for v in rows if v["step_index"] == i]) for r in REGIONS} for m in ("D", "I", "rotation_rms")} for i in POSITIONS}
        groups[arm]["F_by_step"] = {str(i): {r: stat([(v["case"], v["F"][arm][r]) for v in free if v["step_index"] == i]) for r in REGIONS} for i in range(51)}
        final_rows = [json.loads(p.read_text()) for p in (out/arm).glob("c*.json") if int(p.name[1:3]) < count]
        pairs = paired(final_rows)
        groups[arm]["final"] = {k: stat([(v["case"], v[k]) for v in final_rows]) for k in ("sketch_iou", "edge_f1", "leakage", "background_white_mae")}
        groups[arm]["final"]["flip_correct"] = stat([(v["reference"], v["flip_correct"]) for v in pairs])
        groups[arm]["final"]["direction_accuracy"] = stat([(v["case"], float(v["direction"]["correct"])) for v in final_rows])
    contrast = stat([(v["case"], v["models"]["E17_direct"]["D"]["background"]-v["models"]["S0"]["D"]["background"]) for v in rows])
    phase_injection = {name: stat([(v["case"], v["models"]["E17_direct"]["I"]["background"]) for v in rows if a <= v["step_index"] < b]) for name, (a, b) in WINDOWS.items()}
    top = max(phase_injection, key=lambda name: phase_injection[name]["mean"])
    bottom = min(phase_injection, key=lambda name: phase_injection[name]["mean"])
    temporal = phase_injection[top]["ci95"][0] > 2*phase_injection[bottom]["ci95"][1]
    # 预设效应量门槛只用于是否扩展确认，连续数值和CI始终报告。
    bg = groups["E17_direct"]["D_over_E5_score_rms"]["background"]
    rel = groups["E17_direct"]["D_over_E5_texture_rms"]["background"]
    h1 = bg["ci95"][0] > .02 and rel["ci95"][0] > .25
    h2 = bg["ci95"][1] < .02 and groups["E17_direct"]["A"]["background"]["ci95"][0] > 5
    result = {"cases": count, "seeds": list(gen.SEEDS), "training_steps": 0, "groups": groups,
              "E17_minus_S0_D_background": contrast, "E17_I_background_by_window": phase_injection,
              "time_interval_difference": temporal, "H1_supported": h1, "H2_candidate": h2,
              "confirmation_trigger": h1 or h2 or temporal,
              "first_failure_candidate": "common-state score field" if h1 else "trajectory amplification" if h2 else "not yet localized",
              "bootstrap_unit": "case; aggregate both seeds and all relevant timesteps before resampling",
              "A_definition": "descriptive accumulated/local-error ratio only; not Jacobian or stability gain",
              "rotation_scope": "epsilon rotation sensitivity is not semantic correctness; paired generation flip retained",
              "scope": "controlled stripes; empirical E5 subspace is not the UNet training distribution"}
    write_json(out/("pilot_report.json" if count == 8 else "confirmation_report.json"), result)
    write_json(out/"report.json", result)
    return result


def run_windows(pipe, banks, cases, out, count, width, height):
    result = {}
    for name, interval in WINDOWS.items():
        rows = []
        for ref, sk in selected(cases, 0, count):
            for seed in gen.SEEDS:
                for v in ref["variants"]:
                    rows.append(generate(pipe, "window_"+name, ref, sk, seed, v, banks["E5"], out, width, height,
                                         trace=False, window={"bank": banks["E17_direct"], "range": interval}))
        pairs = paired(rows)
        result[name] = {k: case_stat([(r["case"], r[k]) for r in rows]) for k in ("sketch_iou", "edge_f1", "leakage")}
        result[name]["flip_correct"] = case_stat([(r["reference"], r["flip_correct"]) for r in pairs])
        write_json(out/"window_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.root)/"e23_m"
    out.mkdir(exist_ok=True)
    cases = prepare_cases(root, out)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = {"arms": ARMS, "pilot_cases": 8, "confirmation_cases": 32, "seeds": list(gen.SEEDS),
                "positions": POSITIONS, "decode_positions": DECODE_POSITIONS, "inference_steps": 50,
                "CFG": 7, "sketch_scale": .6, "texture_scale": 1, "mask_kernel": 17,
                "no_tex": "texture processor scale=0 for positive AND negative CFG branches; preserve token split/text/sketch",
                "common_state": "E5 original-reference actual latent; original/rot90/off all evaluated there",
                "H1_confirmation_effect_gate": "case-aggregated lowerCI D_bg/E5 full score RMS >.02 AND D_bg/E5 texture-response RMS >.25; diagnostic thresholds, not universal stability limits",
                "window_gate": "top phase I_bg lowerCI >2x bottom phase upperCI; fixed [0,17),[17,34),[34,50)",
                "H2_candidate_gate": "upperCI common score change ratio <.02 AND lowerCI descriptive F/I >5",
                "genealogy_source_sha256": file_hash(root/"e22_4/genealogy.json"),
                "old_protocol_sha256": file_hash(root/"e22_o4_generation/protocol.json"),
                "training_steps": 0, "O4_used": False, "period_adapter_used": False,
                "dose_response": "optional, omitted: primary localization takes priority"}
    write_json(out/"protocol.json", protocol)
    audits, banks = {}, {}

    def load_arm(arm):
        if arm == "S0":
            pipe, bf, pattern, modules, w, h = load_pipeline(root, "cuda:0")
        else:
            pipe, modules, w, h = native_pipeline(root, arm)
            bf, pattern = None, None
        pipe.set_progress_bar_config(disable=True)
        assert pipe.scheduler.__class__.__name__ == "DDIMScheduler" and pipe.scheduler.config.prediction_type == "epsilon"
        if audits:
            previous = audits["E5"]
            assert (w, h) == tuple(previous["resolution"])
            assert normalize_scheduler(pipe) == previous["scheduler"]
        hashes = module_hashes(modules)
        runtime = processor_config(pipe)
        if arm != "E5":
            aliases = {"unet": "unet", "reference_unet" if arm != "S0" else "sketch": "reference_unet",
                       "text_encoder" if arm != "S0" else "text": "text_encoder", "vae": "vae",
                       "image_encoder" if arm != "S0" else "clip": "image_encoder",
                       "tcpm_lite" if arm != "S0" else "tcpm": "tcpm_lite"}
            assert all(hashes[a] == audits["E5"]["loaded_module_hashes"][b] for a, b in aliases.items())
            assert runtime == audits["E5"]["processor_runtime_config"]
        actual_bf = bf if arm == "S0" else pipe.bf_texture_conditioner
        parts = {name: module_hashes({name: getattr(actual_bf, name)})[name]
                 for name in ("stage1", "stage2", "stage3", "stage4", "token_source_proj", "resampler", "token_mlp", "direct_readout")
                 if getattr(actual_bf, name, None) is not None}
        audits[arm] = {**audits.get(arm, {}), "loaded_module_hashes": hashes, "processor_runtime_config": runtime,
                       "bf_readout_components": parts,
                       "resolution": [w, h], "scheduler": normalize_scheduler(pipe)}
        if arm not in banks:
            banks[arm] = token_bank(pipe, cases, out, w, h, arm, bf, pattern)
        return pipe, modules, w, h

    for start, stop in ((0, 8), (8, 32)):
        if start and not report["confirmation_trigger"]:
            break
        for arm in ARMS:
            pipe, modules, w, h = load_arm(arm)
            before = module_hashes(modules)
            for ref, sk in selected(cases, start, stop):
                for v in ref["variants"]:
                    calibration = orientation(Image.open(out/v["path"]).resize((w, h)), sk["roi"])
                    assert calibration["valid"] and axial_distance(calibration["theta"], v["theta"]) <= 20
                for seed in gen.SEEDS:
                    for v in ref["variants"]:
                        generate(pipe, arm, ref, sk, seed, v, banks[arm], out, w, h)
            assert module_hashes(modules) == before
            audits[arm]["frozen_pass"] = True
            write_json(out/"stage0_audit.json", audits)
            if arm != "S0":
                del pipe, modules
                gc.collect()
                torch.cuda.empty_cache()
        torch.save(banks, out/"tokens.pt")
        common_state(pipe, banks, cases, out, start, stop)
        assert module_hashes(modules) == before
        report = summarize(cases, out, stop)
        print("[e23-gate]", stop, report["first_failure_candidate"], report["time_interval_difference"], flush=True)
        del pipe, modules
        gc.collect()
        torch.cuda.empty_cache()
    count = report["cases"]
    write_json(out/"condition_audit.json", condition_statistics(banks, count))
    if report["time_interval_difference"]:
        pipe, modules, w, h = load_arm("E5")
        before = module_hashes(modules)
        report["windows"] = run_windows(pipe, banks, cases, out, count, w, h)
        assert module_hashes(modules) == before
    report["freeze_pass"] = all(v["frozen_pass"] for v in audits.values())
    report["stage4_attention_statistics"] = "per-layer actual K/V/attention output/texture residual in common_state/*.json"
    report["stopped_after_localization"] = True
    write_json(out/"report.json", report)
    print("[e23-complete]", report["first_failure_candidate"], count, flush=True)


if __name__ == "__main__":
    main()
