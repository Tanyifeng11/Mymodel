"""核查 B 的频谱失真是否对应实际周期错误；不重训、不改预定门槛。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.explicit_pattern import pattern_features
from models.learned_pattern import LearnedPatternEncoder
from tools.e15_common import write_json


def load_images(root, rows):
    values = [torch.from_numpy(np.array(Image.open(root / row["texture"]).convert("RGB").resize((128, 128)),
                                       dtype=np.float32) / 255).permute(2, 0, 1) for row in rows]
    return torch.stack(values)


def evaluate(pred, teacher, rows):
    # 一个 token 的窗口宽 64，相对 128 输入覆盖一半：全图频率 = 2 * 峰值 bin。
    predicted, expected = [], []
    for p, t, row in zip(pred, teacher, rows):
        section = slice(4, 20) if row["axis"] == "vertical" else slice(20, 36)
        predicted.append(2 * (int(p[:, section].mean(0).argmax()) + 1))
        expected.append(2 * (int(t[:, section].mean(0).argmax()) + 1))
    actual = np.array([row["frequency"] for row in rows])
    predicted, expected = np.array(predicted), np.array(expected)
    metrics = {"frequency_mae": float(np.abs(predicted - actual).mean()),
               "frequency_exact_accuracy": float((predicted == actual).mean()),
               "teacher_peak_accuracy": float((expected == actual).mean()),
               "teacher_peak_agreement": float((predicted == expected).mean()),
               "spectrum_cosine": float(F.cosine_similarity(pred[..., 4:], teacher[..., 4:], dim=-1).mean())}
    groups = []
    for frequency, phase in sorted({(row["frequency"], row["phase"]) for row in rows}):
        indices = [i for i, row in enumerate(rows) if (row["frequency"], row["phase"]) == (frequency, phase)]
        groups.append({"frequency": frequency, "phase": phase,
                       "predicted_frequencies": predicted[indices].tolist(),
                       "teacher_frequencies": expected[indices].tolist(),
                       "frequency_mae": float(np.abs(predicted[indices] - actual[indices]).mean())})
    return {"aggregate": metrics, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    torch.set_num_threads(4)
    clean = root / "e18_1/clean"
    rows = json.loads((clean / "validation.json").read_text())
    train_rows = json.loads((clean / "train.json").read_text())
    images, train = load_images(clean, rows), load_images(clean, train_rows)
    with torch.inference_mode():
        target = pattern_features(images)
        train_mean = pattern_features(train).mean((0, 1))[None, None].expand_as(target)
        audit = {"teacher": evaluate(target, target, rows),
                 "constant_train_mean": evaluate(train_mean, target, rows), "learned": {}}
        for seed in (42, 43, 44):
            model = LearnedPatternEncoder().eval()
            state = torch.load(root / ("e19/b/encoder_%d.pt" % seed), map_location="cpu", weights_only=False)
            model.load_state_dict(state["encoder"], strict=True)
            audit["learned"][str(seed)] = evaluate(model(images), target, rows)
    write_json(root / "e19/b/frequency_audit.json", audit)
    a = json.loads((root / "e19/a/report.json").read_text())
    b = json.loads((root / "e19/b/report.json").read_text())
    direction_pass = all(g["final"]["minimum_group_margin"] > .01 and
                         g["final"]["margin"] >= .5 * a["geometry"]["final_42"]["margin"] for g in b["geometry"].values())
    result = {"a_pass": a["a_pass"], "b_pass": b["b_pass"], "b_direction_geometry_pass": direction_pass,
              "b_frequency_generalization": audit,
              "stopped_before": ["C", "D"] if not b["b_pass"] else [],
              "conclusion": "explicit pattern branch preserves direction geometry and drives frozen U-Net; learned CNN preserves direction but frequency generalization is insufficient",
              "limits": ["R_rot measures response, not correct generation or matched denoising advantage",
                         "learned candidate trained on clean stripes; no claim of real-fabric or plaid/dots generalization",
                         "failed frequency audit does not mean direction learning failed",
                         "BF-only here is E18.2 checkpoint; R_rot uses full GAM and only active texture layers, not historical flat-adapter metrics"],
              "jobs": {"a": 114105, "b": 114106}}
    write_json(root / "e19/comparison.json", result)
    print(json.dumps({"a_pass": a["a_pass"], "b_pass": b["b_pass"],
                      "direction_pass": direction_pass,
                      "frequency": {k: v["aggregate"] for k, v in audit["learned"].items()},
                      "constant": audit["constant_train_mean"]["aggregate"],
                      "teacher": audit["teacher"]["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
