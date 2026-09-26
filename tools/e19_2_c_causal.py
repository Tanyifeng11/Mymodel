"""C：已知纹样真值的服装轮廓内单步去噪因果验证；不训练、不完整采样。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from garment_mask_utils import build_sketch_garment_mask, build_region_masks, mask_backend_info
from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, rotation_only
from tools.e15_common import sample_indices, write_json
from tools.e15_d5_generation import build_inference_args, load_inference_module
from tools.e15_stages import ResidualProbe, texture_processors
from tools.e18_gold_probe import conditioner
from tools.e19_2_identity import PATTERNS
from tools.e19_handcrafted import bf_tokens, bootstrap, digest, rgb, text_tokens


def region_mean(value, mask):
    return float((value.float() * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1e-8))


def regional_cosine(a, b, mask):
    a, b = (a.float() * mask).flatten(), (b.float() * mask).flatten()
    return float(F.cosine_similarity(a[None], b[None]))


def aggregate(records):
    results = {}
    for name in sorted({r["intervention"] for r in records} - {"matched"}):
        subset = [r for r in records if r["intervention"] == name and r["valid_attribute"]]
        samples = sorted({r["case"] for r in subset})
        def stat(key, rows=subset):
            # noise/timestep 不是独立样本：先在服装案例内平均，再 bootstrap 案例。
            groups = sorted({r["case"] for r in rows})
            return bootstrap([np.mean([r[key] for r in rows if r["case"] == i]) for i in groups])
        results[name] = {"cases": len(samples), "interior_advantage": stat("interior_advantage"),
                         "epsilon_interior_rms": stat("epsilon_interior_rms"), "residual_ratio": stat("residual_ratio"),
                         "by_timestep": {str(t): stat("interior_advantage", [r for r in subset if r["timestep"] == t])
                                         for t in sorted({r["timestep"] for r in subset})}}
        if name.startswith("wrong_"):
            results[name]["target_attribute_projection_cosine"] = stat("target_attribute_projection_cosine")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_c"
    out.mkdir(exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    assert json.loads((root / "e19_2_b/period_audit_report.json").read_text())["structured_readout_pass"]
    torch.manual_seed(42)
    torch.set_num_threads(4)
    args.checkpoint = str(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt")
    args.texture_ckpt = args.checkpoint
    args.base_model_path = str(root / "models/stable-diffusion-v1-5")
    args.vae_model_path = args.base_model_path + "/vae"
    args.clip_model = str(root / "models/clip")
    args.seed, args.steps = 42, 50
    namespace = build_inference_args(args)
    namespace.texture_num_tokens = 16
    namespace.force_texture_num_tokens_override = False
    pipe, _ = load_inference_module().prepare(namespace)
    width, height = namespace.width, namespace.height
    state = torch.load(root / "e18_2/b2/joint_model.pt", map_location="cpu", weights_only=False)
    current = conditioner(state["bf_texture_conditioner"]).to(device, torch.float16)
    # 与 B 的最终 conditioning 完全相同，而非误用另一个 checkpoint 的 TCPM。
    pipe.tcpm_lite.load_state_dict(state["tcpm_lite"], strict=True)
    del state
    model = IdentityGeometryPattern().to(device)
    model.load_state_dict(torch.load(root / "e19_2_a3/fft_rotation_42.pt", map_location=device, weights_only=False)["model"])
    modules = {"identity_geometry": model, "appearance": current, "gam_bf": pipe.bf_texture_conditioner,
               "tcpm": pipe.tcpm_lite, "unet": pipe.unet, "sketch": pipe.reference_unet,
               "clip": pipe.image_encoder, "vae": pipe.vae, "text": pipe.text_encoder}
    for module in modules.values():
        module.eval().requires_grad_(False)
    before = {k: digest(v) for k, v in modules.items()}
    rows = json.loads((root / "e19_2_b/data/manifest.json").read_text())["splits"]["primary"]
    lookup = {(r["pattern"], r["palette"], r["frequency"], r["angle"], r["phase"]): r for r in rows}
    cases = []
    for kind in PATTERNS:
        for color in (0, 2):
            for freq in (4, 10):
                for angle in (0, 90):
                    phase = (.11, .36)[(PATTERNS.index(kind) + color // 2 + angle // 90) % 2]
                    cases.append(lookup[kind, color, freq, angle, phase])
    from train_texture_adapter import MyDataset
    dataset = MyDataset(str(root / "data/processed/bf_full_audit_v1/validation_clean.json"), pipe.tokenizer,
                        height=height, width=width, image_root_path=args.data_root,
                        texture_preprocess_mode="plain_resize", t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)
    indices = sample_indices(dataset, len(cases))
    protocol = {"stage": "C controlled causal sanity check; no training or full sampling",
                "target": "held-out pattern image plain-resized onto fixed real sketch garment mask; white background; target and reference share known identity/color/orientation/period",
                "scope": "synthetic flat garment targets, not photorealistic garment success; repeated_print is chevron prototype; orientation gate only on stripes",
                "identity_seed": 42, "seed_selection": "preassigned primary seed, no performance selection",
                "noise_seeds": [42, 43], "timesteps": [181, 481, 781], "text": "a cloth",
                "fixed": "target latent posterior mean, sketch, text, noise, timestep; one branch replaced at a time",
                "readouts": "regional noise MSE; wrong-minus-matched; residual response; x0 latent delta projection onto known counterfactual target attribute delta",
                "projection_limit": "latent counterfactual signature is descriptive, not an independently validated semantic classifier; all cross-attribute projections reported",
                "gate": "identity, stripe orientation and period all have case-bootstrap lower95CI interior advantage>0, positive mean at every timestep, and positive mean target-attribute projection; otherwise stop before D",
                "mask_backend": mask_backend_info(), "sketch_indices": indices, "cases": cases}
    write_json(out / "protocol.json", protocol)
    pipe.set_scale(.6)
    pipe.set_ipa_scale(1.)
    probe = ResidualProbe(pipe.unet)
    active = [i for i, p in enumerate(probe.processors) if p.layer_group != "semantic"]
    records = []
    try:
        with torch.inference_mode():
            text = text_tokens(pipe, ["a cloth"])
            for case, (row, index) in enumerate(zip(cases, indices)):
                k, c, f, a, p = (row[q] for q in ("pattern", "palette", "frequency", "angle", "phase"))
                donors = {"matched": row,
                          "wrong_identity": lookup[PATTERNS[(PATTERNS.index(k) + 1) % 4], c, f, a, p],
                          "wrong_orientation": lookup[k, c, f, 90 - a, p],
                          "wrong_period": lookup[k, c, 14 - f, a, p],
                          "wrong_appearance": lookup[k, (c + 1) % 4, f, a, p]}
                images = {name: Image.open(root / "e19_2_b/data" / r["texture"]).convert("RGB") for name, r in donors.items()}
                app, identity, geo = {}, {}, {}
                for name, image in images.items():
                    app[name] = bf_tokens(pipe, current, image, text, width, height)
                    source = rgb(image, device)
                    identity[name] = model.identity_tokens(rotation_only(source, fft_orientation(source)[0])).half()
                    geo[name] = model.geometry_tokens(source).half()
                tokens = {}
                for name in donors:
                    value = torch.cat([app[name if name == "wrong_appearance" else "matched"],
                                       identity[name if name == "wrong_identity" else "matched"],
                                       geo[name if name in ("wrong_orientation", "wrong_period") else "matched"]], 1)
                    tokens[name] = pipe.tcpm_lite(value, text)
                tokens["zero"] = torch.zeros_like(tokens["matched"])
                tokens["bf_only"] = pipe.tcpm_lite(app["matched"], text)
                tokens["gam"] = pipe.tcpm_lite(bf_tokens(pipe, pipe.bf_texture_conditioner, images["matched"], text, width, height), text)
                sketch = Image.open(Path(args.data_root) / dataset.data[index]["sketch"]).convert("RGB").resize((width, height), Image.BILINEAR)
                mask_image, info = build_sketch_garment_mask(sketch, width, height)
                from torchvision.transforms.functional import to_tensor
                mask = to_tensor(mask_image)[None].to(device, torch.float16)
                full_regions = build_region_masks(mask, 17)
                targets = {}
                for name, image in images.items():
                    canvas = Image.composite(image.resize((width, height), Image.BILINEAR), Image.new("RGB", (width, height), "white"), mask_image)
                    canvas.save(out / "cases" / ("%02d_%s.png" % (case, name)))
                    x = to_tensor(canvas)[None].to(device, torch.float16) * 2 - 1
                    targets[name] = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
                sketch.save(out / "cases" / ("%02d_sketch.png" % case))
                mask_image.save(out / "cases" / ("%02d_mask.png" % case))
                regions = {name: F.interpolate(value, size=targets["matched"].shape[-2:], mode="area").float()
                           for name, value in zip(("interior", "boundary", "background"), full_regions)}
                assert regions["interior"].sum() > 4, "invalid garment interior"
                sketch_latent = pipe.vae.encode(to_tensor(sketch)[None].to(device, torch.float16) * 2 - 1).latent_dist.mean * pipe.vae.config.scaling_factor
                pipe.reference_unet(sketch_latent, torch.tensor(0, device=device), encoder_hidden_states=None, return_dict=False)
                cached = {name: proc.cache["hidden_states"] for name, proc in pipe.reference_unet.attn_processors.items() if "attn1" in name}
                signatures = {name: value - targets["matched"] for name, value in targets.items() if name != "matched"}
                assert pipe.scheduler.config.prediction_type == "epsilon"
                for noise_seed in (42, 43):
                    noise = torch.randn(targets["matched"].shape, generator=torch.Generator().manual_seed(noise_seed + case * 1000)).to(device, torch.float16)
                    for timestep in (181, 481, 781):
                        t = torch.tensor([timestep], device=device, dtype=torch.long)
                        noisy = pipe.scheduler.add_noise(targets["matched"], noise, t)
                        predictions, captures = {}, {}
                        for name, value in tokens.items():
                            for proc in texture_processors(pipe.unet):
                                proc.num_tokens = value.shape[1]
                            probe.reset()
                            pred = pipe.unet(noisy, t, encoder_hidden_states=torch.cat([text, value], 1),
                                             cross_attention_kwargs={"sa_hidden_states": cached, "tcpm_garment_mask": mask}).sample
                            assert torch.isfinite(pred).all() and sorted(probe.store) == active
                            predictions[name], captures[name] = pred.float(), dict(probe.store)
                        base = predictions["matched"]
                        alpha = float(pipe.scheduler.alphas_cumprod[timestep])
                        for name, pred in predictions.items():
                            error = (pred - noise.float()).square()
                            delta = pred - base
                            x0_delta = -((1 - alpha) / alpha) ** .5 * delta
                            scores = {key: region_mean(error, m) for key, m in regions.items()}
                            response = {key: region_mean(delta.square(), m) ** .5 for key, m in regions.items()}
                            ratios = [float((captures[name][l].float() - captures["matched"][l].float()).norm() /
                                            (captures["matched"][l].float() - captures["zero"][l].float()).norm().clamp_min(1e-8)) for l in active]
                            projection = {key: regional_cosine(x0_delta, sig, regions["interior"]) for key, sig in signatures.items()}
                            records.append({"case": case, "pattern": k, "sketch_index": index, "noise_seed": noise_seed,
                                            "timestep": timestep, "intervention": name, "mask_source": info["mask_source"],
                                            "valid_attribute": name != "wrong_orientation" or k == "stripe",
                                            "loss": scores, "epsilon_rms": response, "epsilon_interior_rms": response["interior"],
                                            "interior_advantage": scores["interior"] - region_mean((base - noise.float()).square(), regions["interior"]),
                                            "residual_ratio": float(np.mean(ratios)), "projection_to_each_attribute": projection,
                                            "target_attribute_projection_cosine": projection.get(name, 0.)})
                write_json(out / "records_partial.json", records)
                print("[c-case]", case + 1, len(cases), k, flush=True)
    finally:
        probe.remove()
    frozen = {k: before[k] == digest(v) for k, v in modules.items()}
    assert all(frozen.values())
    results = aggregate(records)
    passed = all(results[name]["interior_advantage"]["ci95"][0] > 0 and
                 all(q["mean"] > 0 for q in results[name]["by_timestep"].values()) and
                 results[name]["target_attribute_projection_cosine"]["mean"] > 0
                 for name in ("wrong_identity", "wrong_orientation", "wrong_period"))
    write_json(out / "report.json", {"protocol": protocol, "aggregate": results, "freeze_audit": frozen,
                                     "c_pass": passed, "complete": True, "generation_executed": False,
                                     "next": "real-domain confirmation before full generation" if passed else "stop before D; correct pattern condition lacks stable causal advantage"})
    write_json(out / "stage_status.json", {"complete": True, "c_pass": passed, "generation_executed": False})
    b_status = json.loads((root / "e19_2_b/stage_status.json").read_text())
    b_status.update(c_executed=True, c_pass=passed, c_report="../e19_2_c/report.json",
                    next="real-domain confirmation" if passed else "stop before D")
    write_json(root / "e19_2_b/stage_status.json", b_status)
    print("[c-final]", passed, {k: v["interior_advantage"] for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    main()
