"""E17-B: BF token D_rot and frozen U-Net texture-residual R_rot on E15 pairs."""

import argparse
from pathlib import Path

import numpy as np

from tools.e15_common import normalized_response, summary_stats, write_json
from tools.e15_diagnosis import build_inputs, build_models
from tools.e15_stages import build_token_bank, d2_residual_trace, reference_batch


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "base-checkpoint", "direct-checkpoint",
                 "base-model", "clip-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ctx = {"args": {"data_root": args.data_root}, "output": output}
    dataset, indices, hashes, pairs, colors = build_inputs(args)
    ctx.update(dataset=dataset, indices=indices, hashes=hashes, pairs=pairs, colors=colors)
    build_models(argparse.Namespace(checkpoint=args.base_checkpoint, base_model=args.base_model,
                                    clip_model=args.clip_model, device=args.device), ctx)
    state = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    bf_state = state["bf_texture_conditioner"]
    ctx["bf"].configure_direct_readout()
    missing, unexpected = ctx["bf"].load_state_dict(bf_state, strict=True)
    if missing or unexpected:
        raise RuntimeError("E17 BF state mismatch: %s %s" % (missing, unexpected))
    ctx["bf"].to(device=ctx["device"], dtype=ctx["dtype"]).eval().requires_grad_(False)
    build_token_bank(ctx)
    d_rot = []
    with torch.inference_mode():
        for index in indices:
            clip_input, cnn_input = reference_batch("zero_image", index, ctx)
            visual = ctx["vision"](clip_input.to(ctx["device"], ctx["dtype"]),
                                   output_hidden_states=True)
            zero = ctx["model"].get_texture_condition_tokens(
                visual, cnn_input.to(ctx["device"], ctx["dtype"])).float().flatten().cpu().numpy()
            matched = ctx["token_bank"]["matched"][index].float().flatten().cpu().numpy()
            rotated = ctx["token_bank"]["rot90"][index].float().flatten().cpu().numpy()
            d_rot.append(normalized_response(matched, rotated, zero))
    d2 = d2_residual_trace(ctx)
    report = {"protocol": {"base_checkpoint": args.base_checkpoint,
                           "direct_checkpoint": args.direct_checkpoint,
                           "samples": len(indices), "indices": indices,
                           "note": "R_rot uses the E15 frozen base U-Net and fixed latent/timesteps"},
              "final_token_d_rot": summary_stats(d_rot),
              "unet_r_rot": {"mean": float(np.nanmean(d2["r_rot90"])),
                             "by_layer": np.nanmean(d2["r_rot90"], axis=0).tolist()},
              "d2_report": str(output / "d2_residual_trace" / "report.json")}
    write_json(output / "report.json", report)
    print(report["final_token_d_rot"], report["unet_r_rot"], flush=True)


if __name__ == "__main__":
    main()
