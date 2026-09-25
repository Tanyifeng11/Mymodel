"""Build a fixed, visually reviewed E18 diagnostic set from the E14 candidates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


# E14 real-orientation references were inspected individually, including their
# orientation. The remaining IDs were inspected for broad pattern class/color.
ORIENTATION = {2: "vertical", 3: "vertical", 13: "horizontal", 17: "vertical",
               19: "horizontal", 23: "vertical", 25: "horizontal", 30: "vertical",
               36: "vertical", 85: "horizontal"}
FREQUENCY = {13: "fine", 17: "fine", 23: "fine", 3: "fine",
             19: "coarse", 25: "coarse", 36: "coarse", 85: "coarse"}
PATTERN_IDS = {2, 6, 23, 27, 31, 49, 54, 67, 87, 92, 106, 113, 114}
# Each pair was checked visually for a similar grayscale palette and clear
# pattern-category difference. Grayscale histogram overlap is reported, not
# used as a claim that the colors are identical.
COLOR_PATTERN_PAIRS = [(2, 92), (31, 67), (49, 114), (23, 106)]


def pixel_hash(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.candidates), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = {int(row["review_index"]): row for row in csv.DictReader(handle)}
    rows, images = [], {}
    for index in sorted(set(ORIENTATION) | PATTERN_IDS):
        label = labels[index]
        if label["confirmed"] != "1" or label["pattern_visible"] != "1":
            raise ValueError("unconfirmed reference: %d" % index)
        image = Image.open(root / "thumbnails" / ("%04d.png" % index)).convert("RGB")
        if pixel_hash(image) != label["pixel_sha256"]:
            raise ValueError("pixel hash changed: %d" % index)
        images[index] = image
        variants = ("original", "rot90") if index in ORIENTATION else ("original",)
        for variant in variants:
            oriented = image.transpose(Image.Transpose.ROTATE_90) if variant == "rot90" else image
            name = "%04d_%s.png" % (index, variant)
            oriented.save(output / "images" / name)
            axis = ORIENTATION.get(index)
            if variant == "rot90":
                axis = "horizontal" if axis == "vertical" else "vertical"
            rows.append({"review_index": index, "image": "images/" + name,
                         "variant": variant, "orientation": axis,
                         "frequency": FREQUENCY.get(index), "pattern": label["pattern"],
                         "color_group": label["color_group"],
                         "source_group": label["source_group"],
                         "original_texture": label["texture"],
                         "sha256": pixel_hash(oriented)})
    pairs = []
    for left, right in COLOR_PATTERN_PAIRS:
        a, b = labels[left], labels[right]
        if a["pattern"] == b["pattern"] or a["color_group"] != b["color_group"]:
            raise ValueError("invalid color/pattern pair: %d %d" % (left, right))
        if a["source_group"] == b["source_group"]:
            raise ValueError("pair shares provisional source: %d %d" % (left, right))
        hist = []
        for index in (left, right):
            gray = np.asarray(images[index].convert("L"))
            counts = np.histogram(gray, bins=16, range=(0, 256))[0].astype(float)
            hist.append(counts / counts.sum())
        pairs.append({"left": left, "right": right, "color_group": a["color_group"],
                      "patterns": [a["pattern"], b["pattern"]],
                      "gray_hist_overlap": float(np.minimum(*hist).sum()),
                      "verification": "visual pattern/category and broad grayscale palette"})
    report = {"rows": rows, "same_color_different_pattern_pairs": pairs,
              "orientation_sources": len(ORIENTATION),
              "train_split": "held-out validation references; train only on BF training split",
              "source_identity": "provisional sample groups; fabric identity not verified",
              "review": "E14 assistant visual review, rechecked E18 thumbnails",
              "frequency": "manual coarse/fine labels for eight clear stripe references"}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print("[e18-gold] %d images, %d oriented sources, %d pattern pairs" %
          (len(rows), len(ORIENTATION), len(pairs)), flush=True)


if __name__ == "__main__":
    main()
