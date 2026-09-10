"""在固定样本和固定 seed 下，对 E9-B 局部旁路做推理期因果诊断。"""

import argparse
import os
import subprocess
import sys


def build_argparser():
    parser = argparse.ArgumentParser(description="运行 E9-B 局部旁路诊断矩阵")
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--gam-ckpt", required=True)
    parser.add_argument("--texture-ckpt", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-fid", type=int, choices=[0, 1], default=0)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--variants",
        default="all",
        help="逗号分隔的诊断条件名称；默认 all。",
    )
    return parser


def variants(num_steps):
    last = num_steps - 1
    third = num_steps // 3
    return {
        "alpha_000": {"scale": 0.0},
        "alpha_025": {"scale": 0.25},
        "alpha_050": {"scale": 0.50},
        "alpha_075": {"scale": 0.75},
        "alpha_100": {"scale": 1.0},
        "lowpass_appearance": {"scale": 1.0, "input_transform": "lowpass"},
        "highpass_pattern": {"scale": 1.0, "input_transform": "highpass_gray"},
        "spatial_shuffle": {"scale": 1.0, "permutation": "shuffle"},
        "donor_shift": {"scale": 1.0, "donor_shift": 1},
        "donor_pattern_color_matched": {
            "scale": 1.0,
            "donor_shift": 1,
            "input_transform": "donor_color_matched",
        },
        "window_early": {"scale": 1.0, "start": 0, "end": third - 1},
        "window_middle": {"scale": 1.0, "start": third, "end": 2 * third - 1},
        "window_late": {"scale": 1.0, "start": 2 * third, "end": last},
    }


def run_variant(args, name, config):
    command = [
        sys.executable,
        "tools/run_fixed_benchmark.py",
        "--dataset_json", args.dataset_json,
        "--data_root", args.data_root,
        "--split_path", args.split_path,
        "--num_samples", str(args.num_samples),
        "--sample_id_start", "0",
        "--sample_id_end", str(args.num_samples),
        "--seed", str(args.seed),
        "--generation_seed", str(args.generation_seed),
        "--gam_ckpt", args.gam_ckpt,
        "--texture_ckpt", args.texture_ckpt,
        "--clip_model_path", args.clip_model_path,
        "--device", args.device,
        "--modes", "token",
        "--texture_preprocess_mode", "plain_resize",
        "--use_tcpm_lite", "1",
        "--use_texture_gate", "1",
        "--layer_group_enabled", "1",
        "--use_aa_tcr_fuse", "0",
        "--use_text_guided_resampler", "0",
        "--use_local_detail_adapter", "-1",
        "--mask_policy", "sketch_only",
        "--evaluation_protocol", "original_image_size",
        "--compute_fid", str(args.compute_fid),
        "--compute_kid", "0",
        "--resume_generation", "1",
        "--skip_existing", "1",
        "--overwrite", str(args.overwrite),
        "--save_local_detail_trace", "1",
        "--local_detail_scale", str(config.get("scale", 1.0)),
        "--local_detail_step_start", str(config.get("start", 0)),
        "--local_detail_step_end", str(config.get("end", 49)),
        "--local_detail_token_permutation", config.get("permutation", "none"),
        "--local_detail_permutation_seed", str(args.seed),
        "--local_detail_donor_shift", str(config.get("donor_shift", 0)),
        "--local_detail_input_transform", config.get("input_transform", "none"),
        "--output_dir", os.path.join(args.output_dir, name),
        "--run_name", "e9_b_diagnosis",
    ]
    print("[E9-B 诊断]", name, flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = build_argparser().parse_args()
    if args.num_samples < 2:
        raise ValueError("num-samples 至少为 2，donor_shift 需要不同的参考图")
    os.makedirs(args.output_dir, exist_ok=True)
    # 固定评测器与既有 E9 评测一样，使用 inference_IMAGGarment-1.py 的 50 步默认值。
    available_variants = variants(50)
    if args.variants == "all":
        selected_variants = list(available_variants)
    else:
        selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
        unknown = sorted(set(selected_variants) - set(available_variants))
        if unknown:
            raise ValueError(f"未知诊断条件：{', '.join(unknown)}")
    if "alpha_100" not in selected_variants:
        raise ValueError("诊断汇总需要包含 alpha_100 作为配对基线")
    for name in selected_variants:
        config = available_variants[name]
        run_variant(args, name, config)
    subprocess.run(
        [
            sys.executable,
            "tools/analyze_e9_b_diagnostics.py",
            "--diagnostic-root", args.output_dir,
            "--baseline", "alpha_100",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
