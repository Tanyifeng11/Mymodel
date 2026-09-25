"""Measure whether rot90 changes generated interior texture in the reference direction."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.e14_generation_response import spectrum, tiles_inside
from tools.e15_common import summary_stats, write_json


def direction(path, tiles=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if tiles is None:
            image = image.resize((128, 128), Image.BICUBIC)
            tiles = [(0, 0, 128, 128)]
        return spectrum(image, tiles)


def audit(report_path, mask_root):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    records = {(int(row["index"]), row["config"]): row for row in report["per_sample"]}
    pairs = []
    for index in report["protocol"]["indices"]:
        index = int(index)
        matched, rotated = records[index, "matched"], records[index, "rot90"]
        with Image.open(matched["gen"]) as generated_image:
            size = generated_image.size
        with Image.open(mask_root / ("%05d_mask.png" % index)) as image:
            mask = np.asarray(image.convert("L").resize(size, Image.NEAREST)) > 127
        tiles = tiles_inside(mask) or tiles_inside(mask, 64)
        if not tiles:
            continue
        reference = [direction(row["texture_used"])["vertical_score"]
                     for row in (matched, rotated)]
        generated = [direction(row["gen"], tiles)["vertical_score"]
                     for row in (matched, rotated)]
        if any(value is None for value in reference + generated):
            continue
        expected = reference[0] - reference[1]
        observed = generated[0] - generated[1]
        pairs.append({"index": index, "tiles": len(tiles), "reference_delta": expected,
                      "generated_delta": observed,
                      "signed_direction_following": float(np.sign(expected) * observed)})
    confident = [row for row in pairs if abs(row["reference_delta"]) >= 0.2]
    signed = [row["signed_direction_following"] for row in confident]
    pixel_change = {name: report["aggregate"][name]["pixel_diff_vs_ref"]["mean"]
                    for name in ("rot90", "color_near", "wrong_ref")}
    return {"generation_report": str(report_path), "evaluated": len(pairs),
            "direction_confident": len(confident), "reference_threshold": 0.2,
            "signed_direction_following": summary_stats(signed),
            "positive_fraction": float(np.mean(np.asarray(signed) > 0)) if signed else None,
            "pixel_change_vs_matched": pixel_change, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {Path(path).parent.name: audit(path, Path(args.mask_root)) for path in args.reports}
    write_json(args.output, result)
    for name, row in result.items():
        print(name, row["direction_confident"], row["signed_direction_following"],
              row["pixel_change_vs_matched"], flush=True)


if __name__ == "__main__":
    main()
