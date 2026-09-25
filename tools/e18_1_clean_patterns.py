"""Build paired clean stripes with frequency/phase-disjoint validation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.e14_controlled_patterns import pattern_mask
from tools.e15_common import write_json


TRAIN_FREQUENCIES = (3, 5, 8, 12)
VALIDATION_FREQUENCIES = (4, 10)
TRAIN_PHASES = (0.0, 0.23, 0.47)
VALIDATION_PHASES = (0.11, 0.36)
PALETTES = (
    ((32, 32, 32), (224, 224, 224)),
    ((80, 40, 32), (232, 210, 194)),
    ((26, 72, 92), (198, 224, 232)),
    ((42, 78, 45), (206, 229, 206)),
)


def make_image(frequency, phase, palette):
    mask = pattern_mask("stripe", 256, frequency, 0, phase, fraction=0.25)
    rgb = np.where(mask[..., None], palette[0], palette[1]).astype(np.uint8)
    return Image.fromarray(rgb)


def make_dataset(output):
    output = Path(output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "validation": []}
    preview = Image.new("RGB", (8 * 128, 2 * 152), "white")
    draw = ImageDraw.Draw(preview)
    for split, frequencies, phases in (("train", TRAIN_FREQUENCIES, TRAIN_PHASES),
                                        ("validation", VALIDATION_FREQUENCIES, VALIDATION_PHASES)):
        for frequency in frequencies:
            for phase_index, phase in enumerate(phases):
                for palette_index, palette in enumerate(PALETTES):
                    base = make_image(frequency, phase, palette)
                    # Both directions have exactly the same pixels and palette.
                    for variant, image in (("original", base),
                                           ("rot90", base.transpose(Image.Transpose.ROTATE_90))):
                        axis = "vertical" if variant == "original" else "horizontal"
                        name = "%s_f%d_p%d_c%d_%s.png" % (
                            split, frequency, phase_index, palette_index, variant)
                        image.save(images / name)
                        rows[split].append({"texture": "images/" + name,
                                            "source_group": "%s_f%d_p%d_c%d" % (
                                                split, frequency, phase_index, palette_index),
                                            "axis": axis, "frequency": frequency,
                                            "phase": phase, "palette": palette_index,
                                            "variant": variant,
                                            "sha256": hashlib.sha256(image.tobytes()).hexdigest()})
                    if split == "validation":
                        index = (frequencies.index(frequency) * len(phases) + phase_index) * len(PALETTES) + palette_index
                        preview.paste(base.resize((128, 128)), ((index % 8) * 128, (index // 8) * 152))
                        draw.text(((index % 8) * 128 + 2, (index // 8) * 152 + 130),
                                  "f%d p%d c%d" % (frequency, phase_index, palette_index), fill="black")
    preview.save(output / "validation_preview.png")
    # E18-B training_batch expects base images with both input axes and rotates each again.
    base_rows = [row for row in rows["train"]]
    train_path = output / "train.json"
    write_json(train_path, base_rows)
    write_json(output / "validation.json", rows["validation"])
    examples = [[index, 0 if row["axis"] == "vertical" else 1, 1.0]
                for index, row in enumerate(base_rows)]
    write_json(output / "train_selection.json", {
        "manifest": str(train_path), "seed": 42, "scan_count": len(base_rows),
        "per_axis": len(base_rows) // 2, "examples": examples,
        "axis_label_source": "deterministic stripe construction and exact 90-degree rotation"})
    write_json(output / "protocol.json", {
        "train_frequencies": TRAIN_FREQUENCIES,
        "validation_frequencies": VALIDATION_FREQUENCIES,
        "train_phases": TRAIN_PHASES, "validation_phases": VALIDATION_PHASES,
        "palettes": PALETTES, "train_images": len(base_rows),
        "validation_images": len(rows["validation"]),
        "color_balance": "each source has original and rot90 with identical pixels and palette",
        "split": "frequency and phase disjoint; paired variants stay in the same split",
        "limits": "synthetic regular stripes; success does not establish real-fabric transfer"})
    print("[e18.1-clean] train=%d validation=%d" %
          (len(base_rows), len(rows["validation"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    make_dataset(parser.parse_args().output)
