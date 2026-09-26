"""三组相同预算的频率泛化对照；只训练 pattern encoder，不加载 U-Net。"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.explicit_pattern import ExplicitPatternTokens, pattern_features
from models.frequency_pattern import FrequencyBiasedEncoder, frequency_loss
from models.learned_pattern import LearnedPatternEncoder
from models.tcpm_lite import TCPMLite
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES, make_image
from tools.e18_joint_alignment import token_vector
from tools.e19_frequency_audit import evaluate, load_images
from tools.e19_handcrafted import bf_tokens, digest, summary
from tools.e19_learned import feature_metrics


def make_extra(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for frequency in (6, 14):
        for phase in (.07, .19, .41):
            for color, palette in enumerate(PALETTES):
                image = make_image(frequency, phase, palette)
                for axis in ("vertical", "horizontal"):
                    variant = image if axis == "vertical" else image.transpose(Image.Transpose.ROTATE_90)
                    name = "f%d_p%.2f_c%d_%s.png" % (frequency, phase, color, axis)
                    variant.save(output / name)
                    rows.append({"texture": name, "frequency": frequency, "phase": phase,
                                 "axis": axis, "palette": color,
                                 "sha256": hashlib.sha256(variant.tobytes()).hexdigest()})
    write_json(output / "manifest.json", rows)
    return rows


def conditioning(root, device):
    """复用 E19 的实际 CLIP/CNN 预处理和 TCPM，省去无关 U-Net 的加载。"""
    from diffusers.image_processor import VaeImageProcessor
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
    from train_texture_adapter import load_image_encoder_flexible
    from tools.e18_gold_probe import conditioner

    state = torch.load(root / "e18_2/b2/joint_model.pt", map_location="cpu", weights_only=False)
    meta = state.get("meta", {})
    bf = conditioner(state["bf_texture_conditioner"]).to(device, torch.float16)
    tcpm = TCPMLite(768).to(device, torch.float16)
    tcpm.load_state_dict(state["tcpm_lite"], strict=True)
    width, height = int(meta.get("width", 384)), int(meta.get("height", 512))
    del state
    base = str(root / "models/stable-diffusion-v1-5")
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", local_files_only=True).to(device, torch.float16).eval()
    ids = tokenizer(["a cloth"], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        text = text_encoder(ids)[0]
    del text_encoder
    pipe = SimpleNamespace(device=device, clip_image_processor=CLIPImageProcessor(),
                           cond_image_processor=VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False),
                           image_encoder=load_image_encoder_flexible(str(root / "models/clip"), device, torch.float16),
                           tcpm_lite=tcpm)
    mapper = ExplicitPatternTokens().to(device)
    mapper.load_state_dict(torch.load(root / "e19/a/pattern_branch.pt", map_location=device, weights_only=False)["mapper"])
    for module in (bf, tcpm, pipe.image_encoder, mapper):
        module.eval().requires_grad_(False)
    return pipe, bf, mapper, text, width, height


def metrics(pred, target, rows):
    result = evaluate(pred, target, rows)
    result["feature"] = feature_metrics(pred, target)
    # 增加使用预测方向选轴的周期读出，避免方向选轴的 oracle 掩盖方向退化。
    guessed = [dict(row, axis="vertical" if bool(p[:, 0].mean() > p[:, 1].mean()) else "horizontal")
               for p, row in zip(pred, rows)]
    result["predicted_axis_frequency"] = evaluate(pred, target, guessed)["aggregate"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out, clean = root / "e19_1", root / "e18_1/clean"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    train_rows = json.loads((clean / "train.json").read_text())
    primary = json.loads((clean / "validation.json").read_text())
    extra = make_extra(out / "extra_validation")
    for rows in (primary, extra):
        assert not ({r["frequency"] for r in rows} & {r["frequency"] for r in train_rows})
        assert not ({r["phase"] for r in rows} & {r["phase"] for r in train_rows})
    protocol = {"arms": {"cnn_mse": "current CNN, complete 36D MSE",
                         "cnn_frequency": "same CNN, MSE + (1-spectrum cosine) + .1*peak CE, temperature .1",
                         "fft_bias": "same CNN + explicit FFT and shared frequency residual filter; same frequency objective"},
                "seeds": [42, 43, 44], "steps": 500, "batch": 16, "lr": .001, "weight_decay": .0001,
                "paired_design": "same CNN initialization and independent seeded batch-index generator across arms",
                "primary": "E18.1 validation f=4/10 phase=.11/.36",
                "extra": "predeclared f=6/14 phase=.07/.19/.41; no training or selection of steps on validation",
                "gate": "all three seeds on BOTH sets: frequency cosine >=.90, peak accuracy >=.90, MAE <=.5, predicted-axis peak accuracy >=.90; final margin >=80% of corresponding E19-B and every group >.01",
                "frozen": ["BF", "CLIP", "TCPM", "E19-A mapper"], "unet": "not loaded or changed",
                "limits": "fft_bias is a hybrid, not a purely learned FFT replacement; report before-training metrics to expose built-in frequency ability",
                "selection": "prefer cnn_frequency if it passes, otherwise fft_bias; seed42 fixed for subsequent C"}
    write_json(out / "protocol.json", protocol)
    train = load_images(clean, train_rows).to(device)
    datasets = {"primary": (primary, load_images(clean, primary).to(device)),
                "extra": (extra, load_images(out / "extra_validation", extra).to(device))}
    with torch.no_grad():
        teacher = pattern_features(train)
        targets = {name: pattern_features(value[1]) for name, value in datasets.items()}
    old = json.loads((root / "e19/b/report.json").read_text())
    saved_predictions, reports = {}, {}
    for arm in protocol["arms"]:
        reports[arm] = {}
        for seed in protocol["seeds"]:
            torch.manual_seed(seed)
            model = (FrequencyBiasedEncoder() if arm == "fft_bias" else LearnedPatternEncoder()).to(device)
            batch_generator = torch.Generator(device=device).manual_seed(seed + 1000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            with torch.no_grad():
                initial = {name: metrics(model(images), targets[name], rows) for name, (rows, images) in datasets.items()}
            logs = []
            for step in range(1, 501):
                indices = torch.randperm(len(train), generator=batch_generator, device=device)[:16]
                prediction = model(train[indices])
                if arm == "cnn_mse":
                    loss = F.mse_loss(prediction, teacher[indices])
                    parts = {"mse": loss}
                else:
                    loss, parts = frequency_loss(prediction, teacher[indices])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    logs.append(dict(step=step, **{k: float(v.detach()) for k, v in parts.items()}))
                    print("[e19.1-train]", arm, seed, logs[-1], flush=True)
            model.eval().requires_grad_(False)
            report = {"initial": initial, "losses": logs, "parameters": sum(p.numel() for p in model.parameters())}
            with torch.inference_mode():
                report["train"] = feature_metrics(model(train), teacher)
                for name, (rows, images) in datasets.items():
                    prediction = model(images)
                    report[name] = metrics(prediction, targets[name], rows)
                    saved_predictions[arm, seed, name] = prediction.cpu()
            folder = out / arm
            folder.mkdir(exist_ok=True)
            torch.save({"encoder": model.state_dict(), "arm": arm, "seed": seed, "protocol": protocol}, folder / ("encoder_%d.pt" % seed))
            reports[arm][str(seed)] = report
            write_json(out / "training_partial.json", reports)
            del model, optimizer
    del train, teacher
    pipe, bf, mapper, text, width, height = conditioning(root, device)
    modules = {"bf": bf, "tcpm": pipe.tcpm_lite, "mapper": mapper, "clip": pipe.image_encoder}
    before = {k: digest(v) for k, v in modules.items()}
    bases = {}
    reference_geometry = {}
    with torch.inference_mode():
        for name, (rows, _) in datasets.items():
            image_root = clean if name == "primary" else out / "extra_validation"
            bases[name] = [bf_tokens(pipe, bf, Image.open(image_root / row["texture"]).convert("RGB"), text, width, height) for row in rows]
            reference = mapper(targets[name]).half()
            values = torch.cat([token_vector(pipe.tcpm_lite(torch.cat([base, p[None]], 1), text)) for base, p in zip(bases[name], reference)])
            reference_geometry[name] = summary(values.cpu(), rows)
        previous_a = json.loads((root / "e19/a/report.json").read_text())
        if abs(reference_geometry["primary"]["margin"] - previous_a["geometry"]["final_42"]["margin"]) > .001:
            raise ValueError("frozen conditioning differs from E19-A; check preprocessing/checkpoint")
        for arm, seeds in reports.items():
            for seed, report in seeds.items():
                for name, (rows, _) in datasets.items():
                    pattern = mapper(saved_predictions[arm, int(seed), name].to(device)).half()
                    final = torch.cat([token_vector(pipe.tcpm_lite(torch.cat([base, p[None]], 1), text)) for base, p in zip(bases[name], pattern)])
                    geometry = summary(final.cpu(), rows)
                    report[name]["final_geometry"] = geometry
                    m = report[name]["aggregate"]
                    report[name]["pass"] = bool(m["spectrum_cosine"] >= .9 and m["frequency_exact_accuracy"] >= .9 and
                                                m["frequency_mae"] <= .5 and report[name]["predicted_axis_frequency"]["frequency_exact_accuracy"] >= .9 and
                                                geometry["minimum_group_margin"] > .01 and
                                                geometry["margin"] >= .8 * old["geometry"][seed]["final"]["margin"])
    frozen = {k: digest(v) == before[k] for k, v in modules.items()}
    assert all(frozen.values())
    passed = {arm: all(r[name]["pass"] for r in seeds.values() for name in datasets) for arm, seeds in reports.items()}
    selected = next((arm for arm in ("cnn_frequency", "fft_bias") if passed[arm]), None)
    result = {"protocol": protocol, "arms": reports, "freeze_audit": frozen, "passed": passed,
              "reference_geometry": reference_geometry,
              "selected_for_c": selected, "complete": True,
              "next": "controlled E19-C attribute composition" if selected else "stop before C; frequency or direction gate failed"}
    write_json(out / "report.json", result)
    print("[e19.1-final]", json.dumps({"passed": passed, "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
