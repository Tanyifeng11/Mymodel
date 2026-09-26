"""E19-A：真实 GAM 条件接口的手工纹样分支正向对照，不运行完整生成。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.explicit_pattern import ExplicitPatternTokens, pattern_features
from tools.e15_common import sample_indices, write_json
from tools.e18_joint_alignment import token_vector


def rgb(image, device):
    a = np.array(image.convert("RGB").resize((128, 128)), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)


def text_tokens(pipe, captions):
    ids = pipe.tokenizer(captions, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(pipe.device)
    return pipe.text_encoder(ids)[0]


def bf_tokens(pipe, bf, image, text, width, height):
    # 与真实生成接口一致，CLIP 处理原 PIL；CNN 走服装画幅 plain resize。
    clip = pipe.clip_image_processor(images=[image], return_tensors="pt").pixel_values
    visual = pipe.image_encoder(clip.to(pipe.device, torch.float16), output_hidden_states=True)
    cnn = pipe.cond_image_processor.preprocess([image], height=height, width=width)
    cnn = cnn.to(pipe.device, torch.float16) * 2 - 1
    return bf(clip_image_embeds=visual.image_embeds, clip_vision_tokens=visual.hidden_states[-1][:, 1:],
              texture_images=cnn, text_embeds=text, texture_mode="patch_resampled")[0]


def summary(vectors, rows):
    labels = np.array([r["axis"] == "horizontal" for r in rows])
    x = F.normalize(torch.as_tensor(vectors).float(), dim=-1).numpy()
    cosine = x @ x.T
    same = labels[:, None] == labels[None, :]
    eye = np.eye(len(rows), dtype=bool)
    def gap(indices):
        c, s, d = cosine[np.ix_(indices, indices)], same[np.ix_(indices, indices)], eye[np.ix_(indices, indices)]
        return float(c[s & ~d].mean() - c[~s].mean())
    groups = sorted({(r["frequency"], r["phase"]) for r in rows})
    per_group = [{"frequency": f, "phase": p, "margin": gap(
        [i for i, r in enumerate(rows) if (r["frequency"], r["phase"]) == (f, p)])} for f, p in groups]
    return {"margin": gap(list(range(len(rows)))), "by_frequency_phase": per_group,
            "minimum_group_margin": min(r["margin"] for r in per_group)}


def digest(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(42)
    means = values[rng.randint(len(values), size=(2000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.percentile(means, [2.5, 97.5]).tolist()}


def residual_response(pipe, dataset, indices, args, original, current, mapper, constant, clean_rows=None):
    """固定 target/noise/text/sketch；采样前向而非完整生成，记录实际注入残差。"""
    from garment_mask_utils import build_sketch_garment_mask
    from tools.e15_stages import ResidualProbe, texture_processors
    from torchvision.transforms.functional import to_tensor

    records = []
    probe = ResidualProbe(pipe.unet)
    active = [i for i, proc in enumerate(probe.processors) if proc.layer_group != "semantic"]
    pipe.set_scale(.6)
    pipe.set_ipa_scale(1.0)
    try:
        for position, index in enumerate(indices):
            row, batch = dataset.data[index], dataset[index]
            text = pipe.text_encoder(batch["text_input_ids"].to(pipe.device))[0]
            reference = (Path(args.clean) / clean_rows[position]["texture"] if clean_rows is not None
                         else Path(args.data_root) / row.get("texture", row.get("color")))
            with Image.open(reference) as im:
                image = im.convert("RGB")
            rotated = image.transpose(Image.Transpose.ROTATE_90)
            raw = [pattern_features(rgb(im, pipe.device)) for im in (image, rotated)]
            patterns = [mapper(x).half() for x in raw]
            gam = [bf_tokens(pipe, original, im, text, args.width, args.height) for im in (image, rotated)]
            bf = [bf_tokens(pipe, current, im, text, args.width, args.height) for im in (image, rotated)]
            constant_tokens = mapper(constant).half()
            banks = {"gam": gam, "bf_only": bf,
                     "handcrafted": [torch.cat([b, p], 1) for b, p in zip(bf, patterns)],
                     "constant_pattern": [torch.cat([b, constant_tokens], 1) for b in bf]}
            tokens = {}
            for arm, pair in banks.items():
                tokens[arm] = [pipe.tcpm_lite(t, text) for t in pair]
                tokens[arm].append(torch.zeros_like(tokens[arm][0]))
            # 只旋转 pattern，appearance 保持 matched，排除 BF 同时变化的混杂。
            pattern_only = pipe.tcpm_lite(torch.cat([bf[0], patterns[1]], 1), text)
            with Image.open(Path(args.data_root) / row["sketch"]) as im:
                sketch = im.convert("RGB").resize((args.width, args.height), Image.BILINEAR)
            sketch_latent = pipe.vae.encode((to_tensor(sketch)[None].to(pipe.device, torch.float16) * 2 - 1)).latent_dist.mean * .18215
            pipe.reference_unet(sketch_latent, torch.tensor(0, device=pipe.device), encoder_hidden_states=None, return_dict=False)
            cached = {name: proc.cache["hidden_states"] for name, proc in pipe.reference_unet.attn_processors.items() if "attn1" in name}
            mask_image, mask_info = build_sketch_garment_mask(sketch, args.width, args.height)
            mask = to_tensor(mask_image)[None].to(pipe.device, torch.float16)
            posterior = pipe.vae.encode(batch["image"][None].to(pipe.device, torch.float16)).latent_dist
            eta = torch.randn(posterior.mean.shape, generator=torch.Generator().manual_seed(100000 + index)).to(pipe.device, torch.float16)
            target = (posterior.mean + posterior.std * eta) * pipe.vae.config.scaling_factor
            noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(42 + index * 1000)).to(pipe.device, torch.float16)
            for step in args.timesteps:
                t = torch.tensor([step], device=pipe.device, dtype=torch.long)
                noisy = pipe.scheduler.add_noise(target, noise, t)
                for arm, triplet in tokens.items():
                    for proc in texture_processors(pipe.unet):
                        proc.num_tokens = triplet[0].shape[1]
                    captured, preds = [], []
                    for value in triplet + ([pattern_only] if arm == "handcrafted" else []):
                        probe.reset()
                        pred = pipe.unet(noisy, t, encoder_hidden_states=torch.cat([text, value], 1),
                                         cross_attention_kwargs={"sa_hidden_states": cached, "tcpm_garment_mask": mask}).sample
                        if not torch.isfinite(pred).all() or sorted(probe.store) != active:
                            raise ValueError("nonfinite prediction or missing active residual")
                        captured.append(dict(probe.store))
                        preds.append(pred.float())
                    layer_values = []
                    for layer in active:
                        numerator = (captured[0][layer].float() - captured[1][layer].float()).square().mean().sqrt()
                        denom = (captured[0][layer].float() - captured[2][layer].float()).square().mean().sqrt()
                        layer_values.append(float(numerator / denom.clamp_min(1e-8)))
                    item = {"sample": index, "timestep": step, "arm": arm, "r_rot": float(np.mean(layer_values)),
                            "r_rot_layers": layer_values, "active_layer_indices": active,
                            "epsilon_rot_rms": float((preds[0] - preds[1]).square().mean().sqrt()),
                            "token_d_rot": float((triplet[0] - triplet[1]).float().norm() / triplet[0].float().norm().clamp_min(1e-8)),
                            "token_rms": float(triplet[0].float().square().mean().sqrt()),
                            "mask_source": mask_info["mask_source"]}
                    if arm == "handcrafted":
                        item["pattern_only_r_rot"] = float(np.mean([float(
                            (captured[0][l].float() - captured[3][l].float()).norm() /
                            (captured[0][l].float() - captured[2][l].float()).norm().clamp_min(1e-8)) for l in active]))
                    records.append(item)
            domain = "clean" if clean_rows is not None else "real"
            print("[e19-a-response]", domain, "sample", index, flush=True)
            write_json(Path(args.output) / ("response_%s_partial.json" % domain), records)
    finally:
        probe.remove()
    aggregates = {}
    for arm in ("gam", "bf_only", "handcrafted", "constant_pattern"):
        per_sample = [np.mean([r["r_rot"] for r in records if r["sample"] == i and r["arm"] == arm]) for i in indices]
        aggregates[arm] = {"r_rot": bootstrap(per_sample), "per_sample": per_sample,
                           "d_rot": float(np.mean([r["token_d_rot"] for r in records if r["arm"] == arm]))}
    for arm in ("bf_only", "constant_pattern"):
        difference = np.array(aggregates["handcrafted"]["per_sample"]) - aggregates[arm]["per_sample"]
        aggregates["handcrafted_vs_" + arm] = bootstrap(difference)
    return {"records": records, "aggregate": aggregates, "indices": indices, "timesteps": args.timesteps,
            "reference_domain": "held-out clean stripes" if clean_rows is not None else "fixed real references",
            "note": "E5/GAM sketch, TCPM and learned injection gates active; R averaged only over active texture layers; inactive semantic layers excluded"}


def main():
    from tools.e15_d5_generation import load_inference_module, build_inference_args
    from tools.e18_gold_probe import conditioner
    from train_texture_adapter import MyDataset

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("clean", "checkpoint", "appearance-checkpoint", "base-model", "clip-model", "manifest", "data-root", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timesteps", nargs="+", type=int, default=[181, 481, 781])
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    out, clean = Path(args.output), Path(args.clean)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = json.loads((clean / "train.json").read_text())
    rows = json.loads((clean / "validation.json").read_text())
    assert not ({r["frequency"] for r in train_rows} & {r["frequency"] for r in rows})
    assert not ({r["phase"] for r in train_rows} & {r["phase"] for r in rows})
    args.texture_ckpt = args.checkpoint
    args.base_model_path, args.vae_model_path = args.base_model, args.base_model + "/vae"
    args.seed, args.steps = 42, 50
    infer = load_inference_module()
    namespace = build_inference_args(args)
    namespace.texture_num_tokens = 16
    namespace.force_texture_num_tokens_override = False
    pipe, _ = infer.prepare(namespace)
    args.width, args.height = namespace.width, namespace.height
    original = pipe.bf_texture_conditioner.eval().requires_grad_(False)
    state = torch.load(args.appearance_checkpoint, map_location="cpu", weights_only=False)
    current = conditioner(state["bf_texture_conditioner"]).to(args.device, torch.float16).eval().requires_grad_(False)
    del state
    modules = {"gam_bf": original, "appearance_bf": current, "tcpm": pipe.tcpm_lite,
               "unet": pipe.unet, "reference_unet": pipe.reference_unet, "clip": pipe.image_encoder}
    for module in modules.values():
        module.eval().requires_grad_(False)
    before = {name: digest(module) for name, module in modules.items()}
    protocol = {"feature": "four local 64x64 grayscale gradient structure tensors and FFT x/y marginal spectra",
                "mapper": "fixed orthogonal 36->768, seeds 42/43/44; no orientation labels or training",
                "pattern_tokens": 4, "appearance_tokens": current.num_tokens,
                "amplitude": "each pattern token RMS equals train-only mean BF token RMS; no scale sweep",
                "interface": "concat appearance+pattern BEFORE frozen TCPM; dynamic processor token count",
                "training": "none", "primary_mapping_seed": 42,
                "primary_response": "32 held-out clean stripe references paired with fixed 32 target/sketch/noise cases; real references reported separately",
                "checkpoints": {"gam": args.checkpoint, "appearance": args.appearance_checkpoint},
                "preprocess": "actual generation path; CLIP original PIL, BF CNN plain resize to checkpoint dimensions",
                "gates": {"geometry": "all mapping seeds and all unseen frequency/phase groups final margin > .01; final global margin exceeds BF-only by .01",
                          "response": "paired sample bootstrap lower CI > 0 against BF-only AND equal-length constant-pattern control; mean Rrot at least 20% above BF-only"},
                "limits": "Rrot is numerical responsiveness, not denoising advantage or generated orientation; synthetic pass alone does not establish real pattern geometry"}
    write_json(out / "protocol.json", protocol)
    with torch.inference_mode():
        text = text_tokens(pipe, ["a cloth"])
        calibration, calibration_features = [], []
        # 每个训练频率/相位/配色均参与幅度标定，验证集不参与。
        for row in train_rows:
            image = Image.open(clean / row["texture"]).convert("RGB")
            value = bf_tokens(pipe, current, image, text, args.width, args.height)
            calibration.append(float(value.float().square().mean().sqrt()))
            calibration_features.append(pattern_features(rgb(image, args.device)))
        rms = float(np.mean(calibration))
        constant = torch.cat(calibration_features).mean(0, keepdim=True)
        constant = F.normalize(constant, dim=-1)
        mappers = {seed: ExplicitPatternTokens(seed=seed, rms=rms).to(args.device) for seed in (42, 43, 44)}
        vectors = {key: [] for key in ("feature", "gam", "bf_only", "constant_pattern")}
        for seed in mappers:
            vectors.update({"pattern_%d" % seed: [], "final_%d" % seed: []})
        for row in rows:
            image = Image.open(clean / row["texture"]).convert("RGB")
            features = pattern_features(rgb(image, args.device))
            base = bf_tokens(pipe, current, image, text, args.width, args.height)
            gam = bf_tokens(pipe, original, image, text, args.width, args.height)
            vectors["feature"].append(token_vector(features).cpu()[0])
            vectors["gam"].append(token_vector(pipe.tcpm_lite(gam, text)).cpu()[0])
            vectors["bf_only"].append(token_vector(pipe.tcpm_lite(base, text)).cpu()[0])
            control = torch.cat([base, mappers[42](constant).half()], 1)
            vectors["constant_pattern"].append(token_vector(pipe.tcpm_lite(control, text)).cpu()[0])
            for seed, mapper in mappers.items():
                pattern = mapper(features).half()
                final = pipe.tcpm_lite(torch.cat([base, pattern], 1), text)
                vectors["pattern_%d" % seed].append(token_vector(pattern).cpu()[0])
                vectors["final_%d" % seed].append(token_vector(final).cpu()[0])
        geometry = {key: summary(torch.stack(value), rows) for key, value in vectors.items()}
        write_json(out / "geometry.json", geometry)
        torch.save({"mapper": mappers[42].state_dict(), "constant_feature": constant.cpu(),
                    "protocol": protocol, "calibration_rms": rms}, out / "pattern_branch.pt")
        geometry_pass = all(geometry["final_%d" % seed]["minimum_group_margin"] > .01 and
                            geometry["final_%d" % seed]["margin"] > geometry["bf_only"]["margin"] + .01
                            for seed in mappers)
        print("[e19-a-geometry]", {k: round(v["margin"], 6) for k, v in geometry.items()}, flush=True)
        response = None
        if not args.geometry_only:
            dataset = MyDataset(args.manifest, pipe.tokenizer, height=args.height, width=args.width,
                                image_root_path=args.data_root, texture_preprocess_mode="plain_resize",
                                t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)
            response = residual_response(pipe, dataset, sample_indices(dataset, 32), args,
                                         original, current, mappers[42], constant, rows)
            write_json(out / "response.json", response)
            real_response = residual_response(pipe, dataset, sample_indices(dataset, 32), args,
                                              original, current, mappers[42], constant)
            write_json(out / "response_real.json", real_response)
        response_pass = False
        if response:
            a = response["aggregate"]
            response_pass = (a["handcrafted_vs_bf_only"]["ci95"][0] > 0 and
                             a["handcrafted_vs_constant_pattern"]["ci95"][0] > 0 and
                             a["handcrafted"]["r_rot"]["mean"] > 1.2 * a["bf_only"]["r_rot"]["mean"])
    after = {name: digest(module) for name, module in modules.items()}
    freeze = {name: before[name] == after[name] for name in before}
    assert all(freeze.values())
    report = {"geometry": geometry, "response": response["aggregate"] if response else None,
              "freeze_audit": freeze, "calibration_rms": rms, "geometry_pass": geometry_pass,
              "response_pass": response_pass, "a_pass": geometry_pass and response_pass,
              "real_response": real_response["aggregate"] if response else None,
              "next": "evaluate learned pattern encoder" if geometry_pass and response_pass else "stop before E19-B; inspect conditioning interface",
              "complete": not args.geometry_only}
    write_json(out / "report.json", report)
    print("[e19-a-final]", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
