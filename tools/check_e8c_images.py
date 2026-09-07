#!/usr/bin/env python3
"""在原图尺度检查 E8c 开关对照；像素差异仅表示变化幅度，不是质量指标。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


EXPERIMENTS = ("e5", "text_only_off", "text_only_on")
COMPARISONS = (("e5", "text_only_off"), ("e5", "text_only_on"),
               ("text_only_off", "text_only_on"))
INPUT_FIELDS = ("generation_seed", "prompt", "texture_path", "sketch_path")
ROW_FIELDS = ("comparison", "sample_id", "reference_path", "candidate_path", "width", "height",
              "exact_equal", "pixel_mae_255", "max_abs_delta")
SUMMARY_FIELDS = ("comparison", "sample_count", "exact_equal_count", "pixel_mae_255", "max_abs_delta")


def load_samples(run_dir, expected_count):
    with (run_dir / "metrics_per_sample.json").open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if len(rows) != expected_count:
        raise ValueError("%s: 样本数 %s，预期 %s" % (run_dir.name, len(rows), expected_count))
    samples = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")) if row.get("sample_id") is not None else ""
        if not sample_id or sample_id in samples:
            raise ValueError("%s: 样本 ID 缺失或重复：%s" % (run_dir.name, sample_id))
        for field in INPUT_FIELDS + ("gen_path", "target_path"):
            if field not in row or row[field] is None:
                raise ValueError("%s/%s: 缺少 %s" % (run_dir.name, sample_id, field))
        samples[sample_id] = row
    return samples


def target_input(row):
    # target_path 通常是每组自己的 real/ 副本，原始官方 gt 路径才是共同输入。
    return row.get("source_target_path") or row["target_path"]


def check_alignment(samples):
    reference = samples["e5"]
    for experiment in EXPERIMENTS[1:]:
        candidate = samples[experiment]
        if set(reference) != set(candidate):
            raise ValueError("%s 与 E5 的样本 ID 集合不一致" % experiment)
        for sample_id, first in reference.items():
            second = candidate[sample_id]
            for field in INPUT_FIELDS:
                if first[field] != second[field]:
                    raise ValueError("%s/%s: 输入 %s 与 E5 不一致" % (experiment, sample_id, field))
            if target_input(first) != target_input(second):
                raise ValueError("%s/%s: 原始 target_path 与 E5 不一致" % (experiment, sample_id))


def generated_path(run_dir, row):
    path = Path(row["gen_path"])
    if path.is_file():
        return path
    # 服务器结果下载到本地后，保留的服务器绝对路径可能不可用。
    filename = str(row["gen_path"]).replace("\\", "/").rsplit("/", 1)[-1]
    local_path = run_dir / "generated" / filename
    if local_path.is_file():
        return local_path
    raise FileNotFoundError("生成图不存在：%s；本地候选：%s" % (path, local_path))


def compare_images(experiments_dir, samples):
    rows = []
    for sample_id in sorted(samples["e5"]):
        images, paths = {}, {}
        for experiment in EXPERIMENTS:
            paths[experiment] = generated_path(experiments_dir / experiment, samples[experiment][sample_id])
            with Image.open(paths[experiment]) as image:
                images[experiment] = np.asarray(image.convert("RGB"), dtype=np.int16)
        for reference, candidate in COMPARISONS:
            first, second = images[reference], images[candidate]
            if first.shape != second.shape:
                raise ValueError("%s: %s 与 %s 尺寸不一致，不能进行原尺度比较" %
                                 (sample_id, reference, candidate))
            delta = np.abs(first - second)
            maximum = int(delta.max())
            rows.append(dict(comparison=candidate + "_vs_" + reference, sample_id=sample_id,
                             reference_path=str(paths[reference]), candidate_path=str(paths[candidate]),
                             width=first.shape[1], height=first.shape[0], exact_equal=maximum == 0,
                             pixel_mae_255=float(delta.mean()), max_abs_delta=maximum))
    return rows


def summarize(rows):
    summaries = []
    for reference, candidate in COMPARISONS:
        comparison = candidate + "_vs_" + reference
        group = [row for row in rows if row["comparison"] == comparison]
        summaries.append(dict(comparison=comparison, sample_count=len(group),
                              exact_equal_count=sum(row["exact_equal"] for row in group),
                              pixel_mae_255=float(np.mean([row["pixel_mae_255"] for row in group])),
                              max_abs_delta=max(row["max_abs_delta"] for row in group)))
    return summaries


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    report = dict(status="completed", expected_count=args.expected_count,
                  experiments_dir=str(args.experiments_dir), comparisons=[], errors=[],
                  scope="RGB 0–255 原尺度像素对照；汇总 MAE 为逐图 MAE 的均值。变化幅度不是质量指标。")
    try:
        if args.expected_count <= 0:
            raise ValueError("expected-count 必须大于 0")
        samples = {name: load_samples(args.experiments_dir / name, args.expected_count)
                   for name in EXPERIMENTS}
        check_alignment(samples)
        rows = compare_images(args.experiments_dir, samples)
        report["comparisons"] = summarize(rows)
        for summary in report["comparisons"]:
            print("[image] %(comparison)s: 完全相同 %(exact_equal_count)s/%(sample_count)s，"
                  "MAE(0–255)=%(pixel_mae_255).6f，最大像素差=%(max_abs_delta)s" % summary, flush=True)
        if report["comparisons"][0]["exact_equal_count"] != args.expected_count:
            report["status"] = "attention_required"
            print("[image] 关闭文本模块后与 E5 并非逐像素一致，需要结合 MAE 复核；"
                  "不能据此判定质量退化。", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append("%s: %s" % (type(error).__name__, error))
        print("[image] ERROR " + report["errors"][-1], file=sys.stderr, flush=True)
    write_csv(args.output_dir / "image_differences.csv", ROW_FIELDS, rows)
    write_csv(args.output_dir / "image_summary.csv", SUMMARY_FIELDS, report["comparisons"])
    with (args.output_dir / "image_check.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("[image] %s，结果目录：%s" % (report["status"], args.output_dir), flush=True)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
