"""E5/E8a/E8b 第一轮只读诊断，输出机器可读结果，不写说明文档。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.e8_checkpoint_audit import (
    compare_checkpoints, load_checkpoint, metadata_comparison, summarize_differences,
)
from tools.e8_token_probe import run_token_probe, summarize_tokens


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
                    encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("e5", "e8a", "e8b"):
        parser.add_argument("--" + name + "-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights-only", action="store_true", help="只检查权重，跳过条件编码")
    parser.add_argument("--texture-ckpt", help="正式评测用的纹理适配器，用于解析 CLIP 图像编码器路径")
    parser.add_argument("--samples-json", help="本轮 E5 的 benchmark_samples.json")
    parser.add_argument("--data-root", default="/share/home/u2515283058/datasets/BF/validation")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--base-model-path", default="auto")
    parser.add_argument("--image-encoder-path", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.weights_only and (not args.samples_json or not args.texture_ckpt):
        raise ValueError("条件探针需要 --samples-json 和 --texture-ckpt；或使用 --weights-only")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)  # 每次诊断独立保存，避免旧结果混入。
    torch.set_num_threads(args.num_threads)
    report = {
        "status": "running", "torch_version": str(torch.__version__), "errors": [],
        "scope": "权重差异及 BF/TCPM 条件分支；不训练、不修改权重、不生成图片。",
        "unverified": ["完整 UNet/reference UNet 加载和零训练步图像等价性",
                       "早期 checkpoint 的生成质量", "文本开关的最终图像质量"],
        "checkpoints": {}, "metadata_differences": {},
    }
    try:
        checkpoints = {}
        for name in ("e5", "e8a", "e8b"):
            path = Path(getattr(args, name + "_ckpt"))
            print("[load] %s: %s" % (name, path), flush=True)
            checkpoints[name] = load_checkpoint(path)
            report["checkpoints"][name] = {
                "path": str(path.resolve()), "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "format": checkpoints[name].get("checkpoint_format"),
                "recorded_text_guidance_stats": checkpoints[name].get("text_guidance_last_stats", {}),
            }
            for component in ("unet", "ref_unet", "texture_adapter", "bf_texture_conditioner", "tcpm_lite"):
                if not isinstance(checkpoints[name].get(component), dict) or not checkpoints[name][component]:
                    report["errors"].append("%s 缺少必需的非空组件 %s" % (name, component))
        weight_rows = []
        for candidate, reference in (("e8a", "e5"), ("e8b", "e5"), ("e8b", "e8a")):
            comparison = candidate + "_vs_" + reference
            print("[weight] " + comparison, flush=True)
            weight_rows.extend(compare_checkpoints(checkpoints[reference], checkpoints[candidate],
                                                   comparison, candidate_has_text=candidate == "e8b"))
            report["metadata_differences"][comparison] = metadata_comparison(
                checkpoints[reference], checkpoints[candidate])
        summary = summarize_differences(weight_rows)
        write_csv(output / "weight_differences.csv", weight_rows)
        write_csv(output / "weight_summary.csv", summary)
        report["weights"] = {
            "unexpected_entries": sum(row["unexpected"] for row in weight_rows),
            "precision_cast_only_entries": sum(row["precision_cast_only"] for row in weight_rows),
            "note": "冻结部分的精度转换也标为需复核；并不直接证明发生了训练更新。",
            "summary": summary,
        }
        print("[weight] 需复核项: %d" % report["weights"]["unexpected_entries"], flush=True)
        write_json(output / "diagnosis.json", report)  # 即使后续模型依赖失败，也保留权重结果。
        if args.weights_only:
            report["token_probe"] = {"status": "skipped"}
        else:
            print("[token] 严格加载 BF/TCPM，开始相同输入的条件分支对照", flush=True)
            rows, stats, details = run_token_probe(
                checkpoints, load_checkpoint(args.texture_ckpt), args)
            token_summary = summarize_tokens(rows)
            write_csv(output / "token_per_sample.csv", rows)
            write_csv(output / "token_summary.csv", token_summary)
            write_csv(output / "text_guidance_stats.csv", stats)
            zero_rows = [row for row in rows if row["comparison"] == "e5_zero_init_vs_e5"]
            zero_ok = bool(zero_rows) and all(row["exact_equal"] for row in zero_rows)
            report["token_probe"] = {
                "status": "completed", "zero_gate_bf_tcpm_equal": zero_ok,
                "note": "token 变化幅度仅供归因，不能当作生成质量指标。",
                **details, "summary": token_summary,
            }
            if not zero_ok:
                report["errors"].append("E5 加零门控模块后 BF/TCPM 输出未保持逐值相等")
    except Exception as error:
        report["errors"].append("%s: %s" % (type(error).__name__, error))
        print("[ERROR] " + report["errors"][-1], file=sys.stderr, flush=True)
    finally:
        attention = bool(report["errors"]) or bool(report.get("weights", {}).get("unexpected_entries", 0))
        report["status"] = "attention_required" if attention else "completed"
        write_json(output / "diagnosis.json", report)
        print("[结果] %s，目录: %s" % (report["status"], output), flush=True)
    return 2 if attention else 0


if __name__ == "__main__":
    sys.exit(main())
