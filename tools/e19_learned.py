"""A 通过后训练小型 pattern encoder；单一手工特征蒸馏目标，固定 BF/接口。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.explicit_pattern import ExplicitPatternTokens, pattern_features
from models.learned_pattern import LearnedPatternEncoder
from tools.e15_common import sample_indices, write_json
from tools.e18_joint_alignment import token_vector
from tools.e19_handcrafted import bf_tokens, digest, residual_response, rgb, summary, text_tokens


def feature_metrics(pred, teacher):
    return {"cosine": float(F.cosine_similarity(pred, teacher, dim=-1).mean()),
            "spectrum_cosine": float(F.cosine_similarity(pred[..., 4:], teacher[..., 4:], dim=-1).mean()),
            "mse": float(F.mse_loss(pred, teacher)),
            "direction_accuracy": float(((pred[..., 0] > pred[..., 1]) ==
                                         (teacher[..., 0] > teacher[..., 1])).float().mean())}


def main():
    from tools.e15_d5_generation import build_inference_args, load_inference_module
    from tools.e18_gold_probe import conditioner
    from train_texture_adapter import MyDataset

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("a-output", "clean", "checkpoint", "appearance-checkpoint", "base-model", "clip-model", "manifest", "data-root", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-steps", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out, clean = Path(args.output), Path(args.clean)
    out.mkdir(parents=True, exist_ok=True)
    a_report = json.loads((Path(args.a_output) / "report.json").read_text())
    if not a_report["complete"] or not a_report["a_pass"]:
        raise ValueError("E19-A has not passed")
    train_rows = json.loads((clean / "train.json").read_text())
    rows = json.loads((clean / "validation.json").read_text())
    assert not ({r["frequency"] for r in train_rows} & {r["frequency"] for r in rows})
    assert not ({r["phase"] for r in train_rows} & {r["phase"] for r in rows})
    train = torch.cat([rgb(Image.open(clean / row["texture"]), args.device) for row in train_rows])
    validation = torch.cat([rgb(Image.open(clean / row["texture"]), args.device) for row in rows])
    with torch.no_grad():
        teacher, held_teacher = pattern_features(train), pattern_features(validation)
    encoders, training, feature_results = {}, {}, {}
    protocol = {"objective": "MSE to fixed E19-A gradient/FFT descriptors; no other losses",
                "trainable": "small grayscale pixel CNN only; A orthogonal map stays frozen",
                "seeds": [42, 43, 44], "primary_seed": 42, "steps": args.train_steps, "lr": .001,
                "data": args.clean, "limits": "first learned candidate trained on clean stripes, not real fabrics",
                "gate": "all seeds held-out feature cosine >= .90, spectrum cosine >= .80, all final group margins > .01, final margin retains >=50% of A; primary U-Net response significantly exceeds BF/constant controls"}
    write_json(out / "protocol.json", protocol)
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        model = LearnedPatternEncoder().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        with torch.no_grad():
            before = feature_metrics(model(validation), held_teacher)
        logs = []
        for step in range(1, args.train_steps + 1):
            indices = torch.randperm(len(train), device=args.device)[:16]
            pred = model(train[indices])
            loss = F.mse_loss(pred, teacher[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                logs.append({"step": step, "loss": float(loss.detach())})
                print("[e19-b-train]", seed, logs[-1], flush=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            feature_results[str(seed)] = {"before": before,
                                         "train": feature_metrics(model(train), teacher),
                                         "validation": feature_metrics(model(validation), held_teacher)}
        training[str(seed)] = logs
        encoders[seed] = model
        torch.save({"encoder": model.state_dict(), "seed": seed, "protocol": protocol}, out / ("encoder_%d.pt" % seed))
    write_json(out / "training.json", {"losses": training, "features": feature_results})
    del optimizer, train, teacher
    args.texture_ckpt, args.base_model_path, args.vae_model_path = args.checkpoint, args.base_model, args.base_model + "/vae"
    args.seed, args.steps, args.timesteps = 42, 50, [181, 481, 781]
    namespace = build_inference_args(args)
    namespace.texture_num_tokens, namespace.force_texture_num_tokens_override = 16, False
    pipe, _ = load_inference_module().prepare(namespace)
    args.width, args.height = namespace.width, namespace.height
    original = pipe.bf_texture_conditioner.eval().requires_grad_(False)
    state = torch.load(args.appearance_checkpoint, map_location="cpu", weights_only=False)
    current = conditioner(state["bf_texture_conditioner"]).to(args.device, torch.float16).eval().requires_grad_(False)
    del state
    mapper = ExplicitPatternTokens().to(args.device)
    saved = torch.load(Path(args.a_output) / "pattern_branch.pt", map_location=args.device, weights_only=False)
    mapper.load_state_dict(saved["mapper"], strict=True)
    constant = saved["constant_feature"].to(args.device)
    modules = {"gam_bf": original, "appearance_bf": current, "tcpm": pipe.tcpm_lite,
               "unet": pipe.unet, "reference_unet": pipe.reference_unet, "clip": pipe.image_encoder, "mapper": mapper}
    for module in modules.values():
        module.eval().requires_grad_(False)
    before = {name: digest(module) for name, module in modules.items()}
    geometry = {}
    with torch.inference_mode():
        text = text_tokens(pipe, ["a cloth"])
        bases = [bf_tokens(pipe, current, Image.open(clean / row["texture"]).convert("RGB"), text,
                           args.width, args.height) for row in rows]
        for seed, model in encoders.items():
            patterns = mapper(model(validation)).half()
            finals = torch.cat([token_vector(pipe.tcpm_lite(torch.cat([base, pattern[None]], 1), text))
                                for base, pattern in zip(bases, patterns)]).cpu()
            geometry[str(seed)] = {"pattern": summary(token_vector(patterns).cpu(), rows),
                                   "final": summary(finals, rows)}
        write_json(out / "geometry.json", geometry)
        direction_geometry_pass = all(geometry[str(seed)]["final"]["minimum_group_margin"] > .01 and
                                      geometry[str(seed)]["final"]["margin"] >= .5 * a_report["geometry"]["final_42"]["margin"]
                                      for seed in encoders)
        feature_fidelity_pass = all(feature_results[str(seed)]["validation"]["cosine"] >= .9 and
                                    feature_results[str(seed)]["validation"]["spectrum_cosine"] >= .8
                                    for seed in encoders)
        geometry_pass = direction_geometry_pass and feature_fidelity_pass
        response = None
        if geometry_pass:
            dataset = MyDataset(args.manifest, pipe.tokenizer, height=args.height, width=args.width,
                                image_root_path=args.data_root, texture_preprocess_mode="plain_resize",
                                t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)
            response = residual_response(pipe, dataset, sample_indices(dataset, 32), args, original, current,
                                         mapper, constant, rows, encoders[42], "learned")
            write_json(out / "response.json", response)
        response_pass = False
        if response:
            a = response["aggregate"]
            response_pass = (a["learned_vs_bf_only"]["ci95"][0] > 0 and
                             a["learned_vs_constant_pattern"]["ci95"][0] > 0 and
                             a["learned"]["r_rot"]["mean"] > 1.2 * a["bf_only"]["r_rot"]["mean"])
    freeze = {name: before[name] == digest(module) for name, module in modules.items()}
    assert all(freeze.values())
    report = {"features": feature_results, "geometry": geometry, "freeze_audit": freeze,
              "parameters": sum(p.numel() for p in encoders[42].parameters()),
              "direction_geometry_pass": direction_geometry_pass, "feature_fidelity_pass": feature_fidelity_pass,
              "geometry_pass": geometry_pass, "response_pass": response_pass,
              "b_pass": geometry_pass and response_pass, "complete": True,
              "response": response["aggregate"] if response else None,
              "next": "attribute combination validation" if geometry_pass and response_pass else "stop; improve learned pattern representation, keep generator fixed"}
    write_json(out / "report.json", report)
    print("[e19-b-final]", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
