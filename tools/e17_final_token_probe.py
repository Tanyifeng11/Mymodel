"""Probe E17 direct texture tokens using the independent E17-A validation split."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from models.bf_texture_module import FusedDirectReadout
from tools.e17_probe import evaluate


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source = Path(args.fused_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / "samples.json").read_text(encoding="utf-8"))
    with np.load(source / "features.npz") as features:
        arrays = {key: features[key] for key in features.files}
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bf_state = state["bf_texture_conditioner"]
    direct_state = {key[len("direct_readout."):]: value
                    for key, value in bf_state.items() if key.startswith("direct_readout.")}
    layout = "select" if "selection_marker" in direct_state else "mean"
    readout = FusedDirectReadout(arrays["fused"].shape[-1],
                                 bf_state["resampler_queries"].shape[1], layout)
    readout.load_state_dict(direct_state, strict=True)
    readout.to(args.device).eval().requires_grad_(False)
    tokens = []
    with torch.inference_mode():
        for batch in np.array_split(arrays["fused"], max(1, (len(rows) + 7) // 8)):
            tensor = torch.from_numpy(batch.astype(np.float32)).to(args.device)
            tokens.append(readout(tensor, (8, 8)).cpu().numpy().astype(np.float16))
    arrays["fused"] = np.concatenate(tokens)
    np.savez_compressed(output / "features.npz", **arrays)
    (output / "samples.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    settings = SimpleNamespace(checkpoint=args.checkpoint, manifest=str(source / "samples.json"),
                               seed=42, orientation_count=120, epochs=100, device=args.device)
    report = evaluate(settings, rows, arrays, output)
    report["protocol"]["readout_layout"] = layout
    report["protocol"]["input_fused_dir"] = str(source)
    from tools.e15_common import write_json
    write_json(output / "report.json", report)
    print("[e17-final]", layout, {task: {name: round(score["mean"], 3)
                                        for name, score in result["scores"].items()
                                        if name in ("linear", "mlp", "attention")}
                                   for task, result in report["tasks"].items()}, flush=True)


if __name__ == "__main__":
    main()
