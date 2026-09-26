"""E19-C 受控属性组合：冻结模型，检测条件表示的属性可读性与干预特异性。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.explicit_pattern import pattern_features
from models.frequency_pattern import FrequencyBiasedEncoder
from models.learned_pattern import LearnedPatternEncoder
from tools.e14_controlled_patterns import pattern_mask
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES
from tools.e18_gold_probe import conditioner
from tools.e18_joint_alignment import token_vector
from tools.e19_1_frequency import conditioning
from tools.e19_handcrafted import bf_tokens, digest, rgb


def make_data(output):
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": ((3, 5, 8, 12), (0., .23, .47)),
              "primary": ((4, 10), (.11, .36)), "extra": ((6, 14), (.07, .19, .41))}
    data, histograms = {}, {}
    for split, (frequencies, phases) in splits.items():
        rows = []
        for frequency in frequencies:
            for phase in phases:
                for color, palette in enumerate(PALETTES):
                    for kind in ("stripe", "plaid", "dots"):
                        mask = pattern_mask(kind, 256, frequency, 0, phase, fraction=.25)
                        image = Image.fromarray(np.where(mask[..., None], palette[0], palette[1]).astype(np.uint8))
                        for axis in (("vertical", "horizontal") if kind == "stripe" else ("none",)):
                            value = image.transpose(Image.Transpose.ROTATE_90) if axis == "horizontal" else image
                            name = "%s_%s_f%d_p%.2f_c%d_%s.png" % (split, kind, frequency, phase, color, axis)
                            value.save(output / name)
                            counts = np.unique(np.array(value).reshape(-1, 3), axis=0, return_counts=True)
                            signature = (counts[0].tolist(), counts[1].tolist())
                            if color in histograms:
                                assert signature == histograms[color]
                            histograms[color] = signature
                            rows.append({"texture": name, "frequency": frequency, "phase": phase,
                                         "palette": color, "pattern": kind, "axis": axis,
                                         "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
        data[split] = rows
    write_json(output / "manifest.json", {"splits": data, "exact_palette_histograms": histograms,
                                          "source": "procedural labels, not human-confirmed real fabric"})
    # 每行同配色的横/竖条纹、格纹、波点；行间换配色，便于复核受控输入。
    example = data["primary"][:16]
    sheet = Image.new("RGB", (4 * 128, 4 * 150), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(example):
        sheet.paste(Image.open(output / row["texture"]).resize((128, 128)), ((i % 4) * 128, (i // 4) * 150))
        draw.text(((i % 4) * 128, (i // 4) * 150 + 130), "%s c%d" % (row["pattern"], row["palette"]), fill="black")
    sheet.save(output / "preview.png")
    return data


def fit_probe(train, rows, target):
    indices = [i for i, row in enumerate(rows) if target != "axis" or row["pattern"] == "stripe"]
    model = make_pipeline(StandardScaler(), PCA(n_components=min(32, len(indices) - 1), random_state=42),
                          LogisticRegression(C=1., max_iter=2000, random_state=42))
    model.fit(train[indices], [rows[i][target] for i in indices])
    return model


def interventions(rows, predictions):
    lookup = {(r["frequency"], r["phase"], r["palette"], r["pattern"], r["axis"]): i for i, r in enumerate(rows)}
    records = []
    for i, row in enumerate(rows):
        if row["pattern"] != "stripe":
            continue
        f, p, c, axis = row["frequency"], row["phase"], row["palette"], row["axis"]
        wrong_kind = "plaid" if c % 2 == 0 else "dots"
        targets = {"rot90": lookup[f, p, c, "stripe", "horizontal" if axis == "vertical" else "vertical"],
                   "same_color_different_pattern": lookup[f, p, c, wrong_kind, "none"],
                   "different_color_different_pattern": lookup[f, p, (c + 1) % 4, wrong_kind, "none"],
                   "color_only": lookup[f, p, (c + 1) % 4, "stripe", axis]}
        for intervention, j in targets.items():
            donor = rows[j]
            color_ok = predictions["palette"][i] == c and predictions["palette"][j] == donor["palette"]
            pattern_ok = predictions["pattern"][i] == "stripe" and predictions["pattern"][j] == donor["pattern"]
            direction_ok = predictions["axis"][i] == axis and predictions["axis"][j] == donor["axis"] if donor["pattern"] == "stripe" else True
            records.append({"source": i, "donor": j, "intervention": intervention,
                            "color_correct": bool(color_ok), "pattern_correct": bool(pattern_ok),
                            "direction_correct": bool(direction_ok),
                            "joint_correct": bool(color_ok and pattern_ok and direction_ok)})
    aggregate = {name: float(np.mean([r["joint_correct"] for r in records if r["intervention"] == name]))
                 for name in targets}
    return {"joint_success": aggregate, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    gate = json.loads((root / "e19_1/report.json").read_text())
    selected = gate["selected_for_c"]
    if not gate["complete"] or not selected or not gate["passed"][selected]:
        raise ValueError("frequency generalization gate has not passed")
    torch.set_num_threads(4)
    out = root / "e19/c"
    out.mkdir(parents=True, exist_ok=True)
    data = make_data(out / "data")
    protocol = {"scope": "controlled conditioning representation; no U-Net forward, no generated images",
                "selection": selected, "seed": 42, "data_counts": {k: len(v) for k, v in data.items()},
                "same_color": "exact RGB histogram equality by construction, 25% foreground across every pattern",
                "probes": "separate train-only StandardScaler/PCA32/logistic C1 for pattern, palette and stripe direction",
                "gate": "both held-out sets: final pattern BA>=.80, palette BA>=.95, stripe direction BA>=.95; each intervention joint success>=.90; pattern-branch pattern BA>=.80; color-only relative pattern change<.01 and <10% of same-color pattern change",
                "limits": "synthetic stripes/plaid/dots; probe prediction changes are not generated garment changes; real same-color pairs not validated here"}
    write_json(out / "protocol.json", protocol)
    pipe, bf, mapper, text, width, height = conditioning(root, device)
    state = torch.load(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt", map_location="cpu", weights_only=False)
    gam = conditioner(state["bf_texture_conditioner"]).to(device, torch.float16).eval().requires_grad_(False)
    del state
    learned = (FrequencyBiasedEncoder() if selected == "fft_bias" else LearnedPatternEncoder()).to(device).eval()
    learned.load_state_dict(torch.load(root / ("e19_1/%s/encoder_42.pt" % selected), map_location=device, weights_only=False)["encoder"])
    learned.requires_grad_(False)
    modules = {"bf": bf, "gam_bf": gam, "tcpm": pipe.tcpm_lite, "mapper": mapper, "encoder": learned, "clip": pipe.image_encoder}
    before = {k: digest(v) for k, v in modules.items()}
    features, raw = {}, {}
    with torch.inference_mode():
        for split, rows in data.items():
            values = {arm: [] for arm in ("gam", "bf_only", "handcrafted", selected, "pattern_only")}
            raw[split] = []
            for i, row in enumerate(rows):
                image = Image.open(out / "data" / row["texture"]).convert("RGB")
                tensor = rgb(image, device)
                appearance = bf_tokens(pipe, bf, image, text, width, height)
                original = bf_tokens(pipe, gam, image, text, width, height)
                manual = mapper(pattern_features(tensor)).half()
                pattern = mapper(learned(tensor)).half()
                tokens = {"gam": original, "bf_only": appearance,
                          "handcrafted": torch.cat([appearance, manual], 1), selected: torch.cat([appearance, pattern], 1)}
                for arm, value in tokens.items():
                    values[arm].append(token_vector(pipe.tcpm_lite(value, text)).cpu().numpy()[0])
                values["pattern_only"].append(token_vector(pattern).cpu().numpy()[0])
                raw[split].append(pattern.float().cpu().numpy()[0])
                if (i + 1) % 32 == 0:
                    print("[e19-c-extract]", split, i + 1, flush=True)
            features[split] = {k: np.stack(v) for k, v in values.items()}
            raw[split] = np.stack(raw[split])
    reports = {}
    for arm in features["train"]:
        probes = {target: fit_probe(features["train"][arm], data["train"], target) for target in ("pattern", "palette", "axis")}
        reports[arm] = {}
        for split in ("primary", "extra"):
            rows = data[split]
            predictions = {target: probe.predict(features[split][arm]) for target, probe in probes.items()}
            scores = {}
            for target in probes:
                indices = [i for i, row in enumerate(rows) if target != "axis" or row["pattern"] == "stripe"]
                scores[target] = float(balanced_accuracy_score([rows[i][target] for i in indices], predictions[target][indices]))
            reports[arm][split] = {"balanced_accuracy": scores, "interventions": interventions(rows, predictions)}
    passed = {}
    for split in ("primary", "extra"):
        result = reports[selected][split]
        changes = {name: [] for name in ("color_only", "same_color_different_pattern")}
        for record in result["interventions"]["records"]:
            name = record["intervention"]
            if name in changes:
                left, right = raw[split][record["source"]], raw[split][record["donor"]]
                changes[name].append(float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-8)))
        changes = {k: float(np.mean(v)) for k, v in changes.items()}
        result["pattern_branch_relative_change"] = changes
        ba = result["balanced_accuracy"]
        passed[split] = bool(ba["pattern"] >= .8 and ba["palette"] >= .95 and ba["axis"] >= .95 and
                             min(result["interventions"]["joint_success"].values()) >= .9 and
                             reports["pattern_only"][split]["balanced_accuracy"]["pattern"] >= .8 and
                             changes["color_only"] < .01 and changes["color_only"] < .1 * changes["same_color_different_pattern"])
    freeze = {k: before[k] == digest(v) for k, v in modules.items()}
    assert all(freeze.values())
    result = {"protocol": protocol, "arms": reports, "freeze_audit": freeze,
              "controlled_pass": all(passed.values()), "heldout_pass": passed, "complete": True,
              "real_fabric_validated": False, "generation_validated": False,
              "next": "real-fabric and generation verification remain" if all(passed.values()) else "stop; controlled attribute composition not established"}
    write_json(out / "report.json", result)
    print("[e19-c-final]", json.dumps({"controlled_pass": result["controlled_pass"], "heldout_pass": passed,
                                       "selected": {k: v["balanced_accuracy"] for k, v in reports[selected].items()}}), flush=True)


if __name__ == "__main__":
    main()
