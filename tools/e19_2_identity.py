"""E19.2-A：只监督 pattern identity；颜色严格平衡，尺度/相位留出。"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from models.explicit_pattern import ExplicitPatternTokens
from models.frequency_pattern import FrequencyBiasedEncoder
from tools.e14_controlled_patterns import pattern_mask
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES
from tools.e18_joint_alignment import supervised_contrastive, token_vector
from tools.e19_c_composition import fit_probe
from tools.e19_frequency_audit import evaluate, load_images


PATTERNS = ("stripe", "plaid", "dots", "repeated_print")


def motif_mask(size, frequency, angle, phase):
    """重复 V 形印花原型，不把它等同于所有真实 repeated print。"""
    y, x = np.mgrid[:size, :size].astype(float)
    radians = np.deg2rad(angle)
    u = ((x * np.cos(radians) + y * np.sin(radians)) * frequency / size + phase + .5) % 1 - .5
    v = ((-x * np.sin(radians) + y * np.cos(radians)) * frequency / size + phase + .5) % 1 - .5
    score = np.abs(v - (.30 - 1.2 * np.abs(u))) + 4 * np.maximum(np.abs(u) - .30, 0) + 4 * np.maximum(np.abs(v) - .35, 0)
    order = np.argsort(score.ravel(), kind="stable")
    result = np.zeros(size * size, dtype=bool)
    result[order[:size * size // 4]] = True
    return result.reshape(size, size)


def make_data(output):
    output.mkdir(parents=True, exist_ok=True)
    settings = {"train": ((3, 5, 8, 12), (0., .23, .47), (0, 30, 90)),
                "primary": ((4, 10), (.11, .36), (0, 45, 90)),
                "extra": ((6, 14), (.07, .19, .41), (15, 60, 105))}
    data = {}
    for split, (frequencies, phases, angles) in settings.items():
        rows = []
        for frequency in frequencies:
            for phase in phases:
                for angle in angles:
                    for kind in PATTERNS:
                        mask = (motif_mask(256, frequency, angle, phase) if kind == "repeated_print"
                                else pattern_mask(kind, 256, frequency, angle, phase, fraction=.25))
                        assert int(mask.sum()) == 16384
                        for color, palette in enumerate(PALETTES):
                            image = Image.fromarray(np.where(mask[..., None], palette[0], palette[1]).astype(np.uint8))
                            name = "%s_%s_f%d_p%.2f_a%d_c%d.png" % (split, kind, frequency, phase, angle, color)
                            image.save(output / name)
                            rows.append({"texture": name, "frequency": frequency, "phase": phase, "angle": angle,
                                         "palette": color, "pattern": kind,
                                         "template": "%s_f%d_p%.2f_a%d" % (kind, frequency, phase, angle),
                                         "sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        data[split] = rows
    assert not ({r["template"] for r in data["train"]} & {r["template"] for r in data["primary"] + data["extra"]})
    write_json(output / "manifest.json", {"splits": data, "settings": settings,
                                          "histogram": "each palette has exactly 16384 foreground + 49152 background pixels for every class",
                                          "taxonomy_limit": "repeated_print = repeated chevron motif prototype, not all real prints"})
    sheet = Image.new("RGB", (4 * 160, 4 * 185), "white")
    draw = ImageDraw.Draw(sheet)
    for k, kind in enumerate(PATTERNS):
        for c in range(4):
            row = next(r for r in data["primary"] if r["pattern"] == kind and r["palette"] == c and r["angle"] == 0)
            sheet.paste(Image.open(output / row["texture"]).resize((160, 160)), (k * 160, c * 185))
            draw.text((k * 160 + 2, c * 185 + 160), "%s c%d" % (kind, c), fill="black")
    sheet.save(output / "preview.png")
    return data


def encode(model, images):
    return torch.cat([model(batch) for batch in images.split(32)])


def identity_geometry(vectors, rows):
    x = torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy()
    similarity = x @ x.T
    labels = np.array([r["pattern"] for r in rows])
    colors = np.array([r["palette"] for r in rows])
    templates = np.array([r["template"] for r in rows])
    same_pattern = labels[:, None] == labels[None, :]
    same_color = colors[:, None] == colors[None, :]
    same_template = templates[:, None] == templates[None, :]
    positive = same_pattern & ~same_color & ~same_template
    negative = ~same_pattern & same_color
    pos, neg = float(similarity[positive].mean()), float(similarity[negative].mean())
    return {"same_pattern_different_color_cross_template": pos,
            "same_pattern_recolor_only": float(similarity[same_template & ~same_color].mean()),
            "same_color_different_pattern_cosine": neg,
            "same_color_different_pattern_distance": 1 - neg, "margin": pos - neg,
            "positive_pairs": int(positive.sum()), "negative_pairs": int(negative.sum())}


def evaluate_identity(train_vectors, vectors, data):
    probe = fit_probe(train_vectors.numpy(), data["train"], "pattern")
    results = {}
    for split in ("primary", "extra"):
        labels = [r["pattern"] for r in data[split]]
        predictions = probe.predict(vectors[split].numpy())
        results[split] = {"category_accuracy": float(balanced_accuracy_score(labels, predictions)),
                          "class_order": PATTERNS,
                          "per_class_recall": recall_score(labels, predictions, labels=list(PATTERNS), average=None).tolist(),
                          "confusion": confusion_matrix(labels, predictions, labels=list(PATTERNS)).tolist(),
                          "geometry": identity_geometry(vectors[split], data[split])}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2"
    out.mkdir(parents=True, exist_ok=True)
    data = make_data(out / "data")
    torch.set_num_threads(4)
    protocol = {"classes": PATTERNS, "counts": {k: len(v) for k, v in data.items()},
                "start": "E19.1 fft_bias, corresponding seed 42/43/44",
                "trainable": "pattern encoder CNN + shared spectral filter only",
                "loss": "supervised contrastive on actual pattern token mean/std vectors, temperature .12; no auxiliary projection/classifier or other losses",
                "batch": "16 = 4 classes x 4 palettes; each class uses independently sampled scales/phases/angles across palettes",
                "steps": 500, "lr": .0002, "weight_decay": .0001, "seeds": [42, 43, 44],
                "gate": "all seeds on primary AND extra: category BA>=.85, each class recall>=.75, cross-template same-pattern/different-color cosine>=.80, identity margin>=.30",
                "probe": "train-only StandardScaler/PCA32/logistic C1, unchanged from E19-C",
                "frozen": "E19-A token map; BF/CLIP/TCPM/U-Net not loaded during identity training",
                "limits": "controlled A only; no claim about real fabrics; repeated_print is a chevron prototype; orientation/frequency not optimized"}
    write_json(out / "a_protocol.json", protocol)
    images = {split: load_images(out / "data", rows).to(device) for split, rows in data.items()}
    mapper = ExplicitPatternTokens().to(device)
    mapper.load_state_dict(torch.load(root / "e19/a/pattern_branch.pt", map_location=device, weights_only=False)["mapper"])
    mapper.requires_grad_(False)
    original_map = {k: v.clone() for k, v in mapper.state_dict().items()}
    clean = root / "e18_1/clean"
    old_rows = json.loads((clean / "validation.json").read_text())
    old_images = load_images(clean, old_rows).to(device)
    from models.explicit_pattern import pattern_features
    with torch.no_grad():
        old_teacher = pattern_features(old_images)
    buckets = {(kind, c): [i for i, r in enumerate(data["train"]) if r["pattern"] == kind and r["palette"] == c]
               for kind in PATTERNS for c in range(4)}
    labels = torch.tensor([i for i in range(4) for _ in range(4)], device=device)
    reports = {}
    for seed in protocol["seeds"]:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = FrequencyBiasedEncoder().to(device)
        model.load_state_dict(torch.load(root / ("e19_1/fft_bias/encoder_%d.pt" % seed), map_location=device, weights_only=False)["encoder"])
        with torch.inference_mode():
            start_vectors = {k: token_vector(mapper(encode(model, v))).cpu() for k, v in images.items()}
            initial = evaluate_identity(start_vectors["train"], start_vectors, data)
            initial_frequency = evaluate(encode(model, old_images), old_teacher, old_rows)["aggregate"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
        losses = []
        for step in range(1, 501):
            selected = [rng.choice(buckets[kind, c]) for kind in PATTERNS for c in range(4)]
            vectors = token_vector(mapper(model(images["train"][selected])))
            loss = supervised_contrastive(vectors, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                losses.append({"step": step, "identity_loss": float(loss.detach())})
                print("[e19.2-a]", seed, losses[-1], flush=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            vectors = {k: token_vector(mapper(encode(model, v))).cpu() for k, v in images.items()}
            final = evaluate_identity(vectors["train"], vectors, data)
            regression = evaluate(encode(model, old_images), old_teacher, old_rows)["aggregate"]
        passed = all(r["category_accuracy"] >= .85 and min(r["per_class_recall"]) >= .75 and
                     r["geometry"]["same_pattern_different_color_cross_template"] >= .8 and
                     r["geometry"]["margin"] >= .3 for r in final.values())
        reports[str(seed)] = {"initial": initial, "final": final, "losses": losses, "pass": passed,
                              "frequency_monitor_before": initial_frequency, "frequency_monitor_after": regression}
        torch.save({"encoder": model.state_dict(), "protocol": protocol, "seed": seed}, out / ("identity_%d.pt" % seed))
        write_json(out / "a_partial.json", reports)
        print("[e19.2-a-seed]", seed, passed, {k: (v["category_accuracy"], v["geometry"]["margin"]) for k, v in final.items()}, flush=True)
        del optimizer, model
    frozen = all(torch.equal(original_map[k], v) for k, v in mapper.state_dict().items())
    assert frozen
    result = {"protocol": protocol, "seeds": reports, "mapper_frozen": frozen,
              "a_controlled_pass": all(v["pass"] for v in reports.values()), "complete": True,
              "real_fabric_validated": False}
    write_json(out / "a_report.json", result)
    print("[e19.2-a-final]", result["a_controlled_pass"], flush=True)


if __name__ == "__main__":
    main()
