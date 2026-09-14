#!/usr/bin/env python3
"""从 BF 当前文本生成 FiLM 训练清单，仅检查配套路径和文本是否可用。

路径相对 BF 根目录（包含 training、validation、test），不是其 training 子目录。
不会修改源清单或 BF 文件，也不代表完成语义筛选、近重复或评测泄漏检查。
"""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import random


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/processed/bf_full_audit_v1/training_clean.json"
)
SPLITS = {"training", "validation", "test"}


def rooted_path(root, value):
    """仅接受 BF 根目录内的相对路径，兼容清单中的两种目录分隔符。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("缺少路径")
    relative = PurePosixPath(value.replace("\\", "/"))
    if (not relative.parts or relative.is_absolute() or PureWindowsPath(value).drive
            or ".." in relative.parts or relative.parts[0] not in SPLITS):
        raise ValueError("路径必须从 training、validation 或 test 开始且不能越界")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("路径越出 BF 根目录")
    return resolved


def prepare_manifest(bf_root, source_manifest=DEFAULT_SOURCE, max_samples=None, seed=42):
    root = Path(bf_root).resolve()
    if not root.is_dir() or not any((root / split).is_dir() for split in SPLITS):
        raise ValueError("--bf-root 应指向包含 training、validation 或 test 的 BF 根目录")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples 必须大于 0")
    with Path(source_manifest).open(encoding="utf-8-sig") as handle:
        source = json.load(handle)
    if not isinstance(source, list):
        raise ValueError("源清单必须是 JSON 数组")

    rows = []
    excluded = Counter()
    captions_changed = 0
    for original in source:
        if not isinstance(original, dict):
            excluded["invalid_row"] += 1
            continue
        # 原来没有 mask 的 test 清单可继续使用；已有 mask 则必须存在。
        fields = ["cloth", "sketch", "texture"]
        if original.get("mask") is not None:
            fields.append("mask")
        try:
            paths = {key: rooted_path(root, original.get(key)) for key in fields}
        except ValueError:
            excluded["invalid_path"] += 1
            continue
        missing = next((key for key in fields if not paths[key].is_file()), None)
        if missing:
            excluded["missing_" + missing] += 1
            continue

        # training/cloth/id.jpg 和 test/dress/gt/id.jpg 都从目标图的父级推导。
        relative_cloth = PurePosixPath(original["cloth"].replace("\\", "/"))
        expected_id = relative_cloth.parent.parent / relative_cloth.stem
        if original.get("sample_id") is not None:
            sample_id = str(original["sample_id"]).replace("\\", "/")
            if sample_id != str(expected_id):
                excluded["sample_id_mismatch"] += 1
                continue
        try:
            text_path = rooted_path(root, str(expected_id.parent / "text" / (relative_cloth.stem + ".txt")))
        except ValueError:
            excluded["invalid_path"] += 1
            continue
        try:
            caption = text_path.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            excluded["missing_text"] += 1
            continue
        except (OSError, UnicodeError):
            excluded["unreadable_text"] += 1
            continue
        if not caption:
            excluded["empty_text"] += 1
            continue
        row = dict(original)
        row["caption"] = caption
        captions_changed += caption != original.get("caption")
        rows.append(row)

    eligible_count = len(rows)
    if max_samples is not None and len(rows) > max_samples:
        # 对相同源清单和 seed 复现选样，并保持入选条目的原有顺序。
        indices = sorted(random.Random(seed).sample(range(len(rows)), max_samples))
        rows = [rows[index] for index in indices]
    summary = {
        "source_count": len(source),
        "eligible_count": eligible_count,
        "selected_count": len(rows),
        "caption_changed_eligible_count": captions_changed,
        "excluded_count": sum(excluded.values()),
        "excluded_reasons": dict(sorted(excluded.items())),
        "seed": seed,
    }
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf-root", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="显式允许覆盖已有输出，但禁止覆盖源清单")
    args = parser.parse_args(argv)
    source_path = args.source_manifest.resolve()
    output = args.output.resolve()
    if output == source_path:
        parser.error("输出不能覆盖源清单")
    if output.suffix.lower() != ".json":
        parser.error("--output 必须是新的 .json 清单")
    if output.exists() and not args.overwrite:
        parser.error("输出已存在；请更换路径或显式使用 --overwrite")
    try:
        rows, summary = prepare_manifest(args.bf_root, source_path, args.max_samples, args.seed)
        if not rows:
            parser.error("没有可用样本，未写入输出：" + json.dumps(summary, ensure_ascii=False))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False))
    print("已写入：" + str(output))
    print("仅同步当前文本并核对路径；未进行图像质量、语义或重复样本审计。")
    return summary


if __name__ == "__main__":
    main()
