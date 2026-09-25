"""Build an E15-compatible flat probe checkpoint using the BF weights in a GAM checkpoint."""

import argparse
from pathlib import Path

import torch

from tools.e15_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat", required=True)
    parser.add_argument("--gam", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    flat = torch.load(args.flat, map_location="cpu", weights_only=False)
    gam = torch.load(args.gam, map_location="cpu", weights_only=False)
    source = gam["bf_texture_conditioner"]
    differences = {}
    replaced = 0
    for key, value in source.items():
        name = "bf_texture_conditioner." + key
        if name not in flat:
            continue
        if flat[name].shape != value.shape:
            raise ValueError("BF shape mismatch: %s" % name)
        old = flat[name].float()
        new = value.float()
        differences[key] = float((old - new).norm() / new.norm().clamp_min(1e-8))
        flat[name] = value
        replaced += 1
    if replaced < 30:
        raise ValueError("too few BF weights replaced: %d" % replaced)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, output)
    write_json(output.with_suffix(".json"), {
        "flat_source": args.flat, "gam_source": args.gam, "replaced": replaced,
        "relative_weight_change": differences})
    print("[e17] replaced %d BF tensors; max relative change %.4f" %
          (replaced, max(differences.values())), flush=True)


if __name__ == "__main__":
    main()
