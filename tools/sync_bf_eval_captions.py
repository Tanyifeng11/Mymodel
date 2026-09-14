#!/usr/bin/env python3
"""将 BF 当前验证/测试 txt 同步到现有清单；默认预览，--apply 备份后写入。

保留样本、顺序和路径。同步 caption/prompt，以及归一化清单的文本冲突标记。
历史审计、已生成结果、反事实干预文本和训练清单不在默认范围内。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.normalize_attributes import detect_conflicts, summarize


DEFAULT_MANIFESTS = [
    f"data/processed/{version}/{split}_clean.{suffix}"
    for version in ("bf_full_audit_v1", "bf_annotation_patch_v1")
    for split in ("validation", "test")
    for suffix in ("json", "jsonl")
] + [
    "data/bf_validation_e9.json", "data/bf_validation_e9_64.json",
    "data/raw/bf_fashion_validation_garment_only.jsonl",
    "data/processed/val_normalized.json", "data/processed/val_full.json",
    "data/processed/dev100.json", "data/processed/dev500.json",
    "data/processed/e10_supervision_v1/validation.jsonl",
]


def read_records(path):
    content = path.read_bytes()
    text = content.decode("utf-8-sig")
    records = ([json.loads(line) for line in text.splitlines() if line.strip()]
               if path.suffix == ".jsonl" else json.loads(text))
    if not isinstance(records, list):
        raise ValueError(f"清单必须是记录数组，不能直接覆盖反事实等嵌套产物：{path}")
    return content, records


def text_path_for(record, bf_root):
    value = record.get("cloth") or record.get("target") or record.get("target_image")
    if not value:
        raise ValueError("清单记录缺少目标图像路径")
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError(f"目标图像必须使用 BF 相对路径：{value}")
    # E9 的固定清单相对 validation 根目录，其余清单相对 BF 根目录。
    if relative.parts[0] not in ("validation", "test"):
        if relative.parts[0] == "gt" and record.get("category") == "validation":
            relative = PurePosixPath("validation") / relative
        else:
            raise ValueError(f"无法确认属于 BF 验证/测试集：{value}")
    path = (bf_root / relative.parent.parent / "text" / (relative.stem + ".txt")).resolve()
    try:
        path.relative_to(bf_root)
    except ValueError:
        raise ValueError(f"文本路径越出 BF 根目录：{value}")
    return path


def updated_record(record, caption):
    updated = dict(record)
    fields = [key for key in ("caption", "prompt") if key in record]
    if not fields or any(not isinstance(record[key], str) for key in fields):
        raise ValueError("同步清单必须具有字符串 caption 或 prompt")
    changed_text = any(record[key] != caption for key in fields)
    for key in fields:
        updated[key] = caption
    norm = record.get("attributes_normalized")
    if norm is not None and "attribute_conflicts" in record:
        conflicts = detect_conflicts(caption, norm)
        updated["attribute_conflicts"] = conflicts
        if "aux_supervision" in record:
            updated["aux_supervision"] = {
                **record["aux_supervision"],
                "color": "color" not in conflicts and norm["color"] != ["unknown"],
                "pattern": "pattern" not in conflicts and norm["pattern"] != ["unknown"],
            }
    derived_changed = any(updated.get(k) != record.get(k)
                          for k in ("attribute_conflicts", "aux_supervision"))
    return updated, changed_text, derived_changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf-root", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", action="append", help="可重复传入项目内清单路径；省略时同步既有 BF 清单")
    parser.add_argument("--apply", action="store_true", help="备份旧清单后实际更新；默认只报告变化")
    args = parser.parse_args(argv)
    project, bf_root = args.project_root.resolve(), args.bf_root.resolve()
    files = [project / name for name in (args.manifest or DEFAULT_MANIFESTS)]
    if not args.manifest:
        files = [path for path in files if path.is_file()]
    if not files:
        parser.error("未找到需要同步的清单")
    sources, needed = [], set()
    for path in files:
        path = path.resolve()
        try:
            path.relative_to(project)
        except ValueError:
            parser.error("清单必须位于项目目录内")
        original, records = read_records(path)
        text_paths = [text_path_for(row, bf_root) for row in records]
        sources.append((path, original, records, text_paths))
        needed.update(text_paths)

    def read_caption(path):
        caption = path.read_text(encoding="utf-8-sig").strip()
        if not caption:
            raise ValueError(f"当前 txt 为空，停止同步：{path}")
        return path, caption

    # 相同样本出现在多个清单中时只读取一次，全部读完才允许写入。
    print(f"核对 {len(files)} 份清单、{len(needed)} 份唯一 txt……", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        captions = dict(executor.map(read_caption, sorted(needed)))
    plans, reports = [], []
    normalized_validation = None
    for path, original, records, text_paths in sources:
        updates = [updated_record(row, captions[txt]) for row, txt in zip(records, text_paths)]
        updated = [item[0] for item in updates]
        report = {"file": str(path.relative_to(project)).replace("\\", "/"),
                  "rows": len(records), "text_changed": sum(item[1] for item in updates),
                  "derived_changed": sum(item[2] for item in updates)}
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if path == project / "data/processed/val_normalized.json":
            normalized_validation = updated
        if updated != records:
            content = (("\n".join(json.dumps(row, ensure_ascii=False) for row in updated) + "\n")
                       if path.suffix == ".jsonl" else json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
            plans.append((path, original, content.encode("utf-8"), updated))

    # 只刷新派生标记对应的验证集统计，不重新抽样或改变属性标签。
    statistics_path = project / "data/processed/attribute_statistics.json"
    if normalized_validation is not None and statistics_path.is_file():
        original = statistics_path.read_bytes()
        statistics = json.loads(original.decode("utf-8-sig"))
        refreshed = summarize(normalized_validation, "val")
        changed = any(statistics["val"].get(k) != refreshed[k] for k in ("conflicts", "aux_usable"))
        if changed:
            for key in ("conflicts", "aux_usable"):
                statistics["val"][key] = refreshed[key]
            plans.append((statistics_path, original,
                          (json.dumps(statistics, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), statistics))

    if args.apply and plans:
        backup = project / "tmp/bf_eval_caption_backups" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        )
        # 所有原文件先备份，再逐文件替换，方便恢复旧评测口径。
        for path, original, _, _ in plans:
            if path.read_bytes() != original:
                raise RuntimeError(f"预检后清单已被修改，停止：{path}")
            saved = backup / path.relative_to(project)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(original)
        for path, _, content, expected in plans:
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
            if path.suffix == ".jsonl":
                actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            else:
                actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError(f"写入校验失败：{path}")
        print(f"已更新 {len(plans)} 个文件；原文件备份：{backup}", flush=True)
    else:
        print(f"{'预览' if not args.apply else '检查'}完成：{len(plans)} 个文件需要更新。", flush=True)
    return reports


if __name__ == "__main__":
    main()
