import argparse
from pathlib import Path

import torch


ADAPTER_KEYS = (
    "to_k_ip",
    "to_v_ip",
    "to_k_palette",
    "to_v_palette",
    "palette_branch_scale",
    "texture_gate_delta",
    "balanced_gate",
)


def parse_block_ids(value):
    block_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not block_ids:
        raise ValueError("--unet_late_blocks 不能为空")
    return block_ids


def is_interpolation_target(name, block_ids, include_output_layer):
    block_prefixes = tuple(f"up_blocks.{block_id}." for block_id in block_ids)
    in_late_block = name.startswith(block_prefixes)
    in_output_layer = include_output_layer and (
        name.startswith("conv_norm_out.") or name.startswith("conv_out.")
    )
    is_adapter_param = any(key in name for key in ADAPTER_KEYS)
    return (in_late_block or in_output_layer) and not is_adapter_param


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 不是字典: {path}")
    if "unet" not in checkpoint or not isinstance(checkpoint["unet"], dict):
        raise KeyError(f"checkpoint 缺少 unet state_dict: {path}")
    return checkpoint


def interpolate_checkpoints(
    e5_path,
    e6_path,
    output_path,
    alpha,
    block_ids,
    include_output_layer,
    overwrite=False,
):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须位于 [0, 1]，当前为 {alpha}")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，请传入 --overwrite 覆盖: {output_path}")

    e5_checkpoint = load_checkpoint(e5_path)
    e6_checkpoint = load_checkpoint(e6_path)
    e5_format = e5_checkpoint.get("checkpoint_format")
    e6_format = e6_checkpoint.get("checkpoint_format")
    if e5_format != e6_format:
        raise ValueError(
            f"checkpoint 格式不一致: E5={e5_format}, E6={e6_format}"
        )

    e5_unet = e5_checkpoint["unet"]
    e6_unet = e6_checkpoint["unet"]

    selected_names = {
        name
        for name in e5_unet
        if is_interpolation_target(name, block_ids, include_output_layer)
    }
    if not selected_names:
        raise RuntimeError("没有找到需要插值的 UNet 参数，请检查 block 配置")
    e6_selected_names = {
        name
        for name in e6_unet
        if is_interpolation_target(name, block_ids, include_output_layer)
    }
    if selected_names != e6_selected_names:
        missing_in_e6 = sorted(selected_names - e6_selected_names)
        missing_in_e5 = sorted(e6_selected_names - selected_names)
        raise KeyError(
            "E5/E6 待插值参数集合不一致: "
            f"E6缺少={missing_in_e6[:5]}, E5缺少={missing_in_e5[:5]}"
        )

    selected_numel = 0
    interpolated_count = 0
    skipped_non_float = []
    with torch.no_grad():
        for name in sorted(selected_names):
            e5_tensor = e5_unet[name]
            e6_tensor = e6_unet[name]
            if not isinstance(e5_tensor, torch.Tensor) or not isinstance(
                e6_tensor, torch.Tensor
            ):
                raise TypeError(f"参数不是 Tensor: {name}")
            if e5_tensor.shape != e6_tensor.shape:
                raise ValueError(
                    f"参数 shape 不一致: {name}, E5={tuple(e5_tensor.shape)}, "
                    f"E6={tuple(e6_tensor.shape)}"
                )
            if e5_tensor.dtype != e6_tensor.dtype:
                raise ValueError(
                    f"参数 dtype 不一致: {name}, E5={e5_tensor.dtype}, "
                    f"E6={e6_tensor.dtype}"
                )
            if not (torch.is_floating_point(e5_tensor) or torch.is_complex(e5_tensor)):
                skipped_non_float.append(name)
                continue

            # 直接在 E5 payload 上插值，避免为完整 checkpoint 再复制一份内存。
            e5_tensor.mul_(1.0 - alpha).add_(e6_tensor, alpha=alpha)
            selected_numel += e5_tensor.numel()
            interpolated_count += 1

    meta = dict(e5_checkpoint.get("meta") or {})
    meta["checkpoint_interpolation"] = {
        "method": "linear_e5_e6_unet_late",
        "alpha": alpha,
        "e5_checkpoint": str(Path(e5_path).resolve()),
        "e6_checkpoint": str(Path(e6_path).resolve()),
        "unet_late_blocks": block_ids,
        "include_output_layer": bool(include_output_layer),
        "interpolated_tensor_count": interpolated_count,
        "interpolated_parameter_count": selected_numel,
        "skipped_non_float_tensors": skipped_non_float,
    }
    e5_checkpoint["meta"] = meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(e5_checkpoint, output_path)
    print(
        f"[完成] alpha={alpha:.4f}, tensors={interpolated_count}, "
        f"parameters={selected_numel}, output={output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="仅对 E5/E6 的 UNet 后段权重做线性插值"
    )
    parser.add_argument("--e5_ckpt", required=True)
    parser.add_argument("--e6_ckpt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--unet_late_blocks", default="2,3")
    parser.add_argument(
        "--include_output_layer",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    interpolate_checkpoints(
        e5_path=args.e5_ckpt,
        e6_path=args.e6_ckpt,
        output_path=args.output_path,
        alpha=args.alpha,
        block_ids=parse_block_ids(args.unet_late_blocks),
        include_output_layer=bool(args.include_output_layer),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
