"""生成 CTD 纹样候选四联图，并导出逐条视觉审核结果；不修改源数据。"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from garment_mask_utils import build_sketch_garment_mask, estimate_cloth_foreground_mask, mask_backend_info

PATTERNS = {"striped", "plaid", "polka dot"}
REASONS = {
    "a": "参考可辨认重复纹样，与目标主要区域一致；内部 mask 视觉可用",
    "b": "参考纹样过于模糊或近纯色，不能可靠监督纹样",
    "c": "参考只含局部线条、色块或图案边缘，重复结构证据不足",
    "m": "拼色、拼接或局部纹样，整件服装内部不适合统一纹样监督",
    "l": "不属于本轮清晰重复图案目标：疑似罗纹、褶皱、毛皮或标签不符；不代表原 CTD 标签必然错误",
    "u": "参考清晰度、尺度或配对一致性存疑，暂不纳入严格子集",
    "k": "纹样候选可用，但当前自动 mask 不可靠，需修正后重审",
}


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "masks").mkdir()
    (args.output / "sheets").mkdir()
    rows = []
    for split, filename in [("train", "bf_fashion_training_full.jsonl"), ("validation", "bf_fashion_validation_garment_only.jsonl")]:
        source = args.project / "data/raw" / filename
        for record in read_jsonl(source):
            if not PATTERNS.intersection(record.get("attributes", {}).get("pattern", [])):
                continue
            row = dict(record)
            row.update(review_index=len(rows), split=split, source_manifest=str(source), source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            rows.append(row)
    hashes = {}
    sheet = None
    for i, row in enumerate(rows):
        gt = Image.open(args.root / row["target_image"]).convert("RGB")
        ref = Image.open(args.root / row["texture_candidates"][0]).convert("RGB")
        sketch = Image.open(args.root / row["sketch"]).convert("RGB")
        row["target_pixel_sha256"] = hashlib.sha256(str(gt.size).encode() + gt.tobytes()).hexdigest()
        hashes.setdefault(row["target_pixel_sha256"], []).append(i)
        mask, info = build_sketch_garment_mask(sketch, *gt.size)
        if info["mask_low_confidence"]:
            mask, info = estimate_cloth_foreground_mask(gt, *gt.size)
        inner = mask.filter(ImageFilter.MinFilter(9))
        mask_path = args.output / "masks" / (row["id"] + ".png")
        mask.save(mask_path)
        inner.save(mask_path.with_name(row["id"] + "_inner.png"))
        row.update(original_mask=row.get("mask"), original_mask_exists=bool(row.get("mask") and (args.root / row["mask"]).is_file()), review_mask=str(mask_path.resolve()), review_inner_mask=str(mask_path.with_name(row["id"] + "_inner.png").resolve()), mask_diagnostics=info, image_sizes={"target": gt.size, "texture": ref.size, "sketch": sketch.size})
        overlay = np.array(gt).copy()
        valid = np.asarray(inner) > 127
        overlay[valid] = (overlay[valid] * .65 + np.array([0, 200, 50]) * .35).astype(np.uint8)
        overlay = Image.fromarray(overlay)
        # 每页 24 条，每条 GT/reference/sketch/绿色内部 mask；数字对应 review_index。
        if i % 24 == 0:
            sheet = Image.new("RGB", (1536, 1344), "white")
        x, y = (i % 24 % 3) * 512, (i % 24 // 3) * 168
        draw = ImageDraw.Draw(sheet)
        title = f'{i:03d} {row["split"]} {row["id"].split("_")[-1]} ' + ",".join(row["attributes"]["pattern"])
        draw.text((x + 2, y + 2), title, fill="black")
        for j, im in enumerate([gt, ref, sketch, overlay]):
            im = im.copy()
            im.thumbnail((128, 128))
            sheet.paste(im, (x + j * 128, y + 22))
        draw.text((x + 2, y + 151), "GT / REF / SKETCH / INNER    mask=" + str(round(info["mask_confidence"], 2)), fill="black")
        if i % 24 == 23 or i == len(rows) - 1:
            sheet.save(args.output / "sheets" / f"page_{i // 24:02d}.png")
            print(f"sheet {i // 24:02d}: through {i}", flush=True)
    write_jsonl(args.output / "candidates.jsonl", rows)
    (args.output / "prepare_summary.json").write_text(json.dumps({"counts": dict(Counter(r["split"] for r in rows)), "exact_pixel_duplicate_groups": [v for v in hashes.values() if len(v) > 1], "mask_backend": mask_backend_info(), "scope": "仅三类已标注候选；像素哈希只检查本批 GT 完全重复，不等于全 BF 近重复审查"}, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize(args):
    rows = read_jsonl(args.output / "candidates.jsonl")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    assert set(decisions) == {str(i) for i in range(len(rows))}, "每条候选必须有视觉审核记录"
    pixel_groups = {}
    for row in rows:
        pixel_groups.setdefault(row["target_pixel_sha256"], []).append(row)
    for i, row in enumerate(rows):
        code = decisions[str(i)]
        reason = code[0]
        assert reason in REASONS
        row["review"] = {"decision": "accept" if reason == "a" else ("uncertain" if reason in "uck" else "reject"), "reason_code": reason, "reason": REASONS[reason], "sketch_pattern": {"0": "none", "1": "partial", "2": "strong"}.get(code[1:2], "not_certified"), "reviewer": "Codex视觉复核，非独立双人标注", "mask_visual_ok": True if reason == "a" else None, "pattern_type_verified": row["attributes"]["pattern"] if reason == "a" else None}
        row["mask"] = row["review_mask"]
        row["review"]["mask_status"] = "自动生成的粗前景 mask，经视觉检查；不是人工像素级标注"
        row["review"]["pattern_type_verified"] = sorted(PATTERNS.intersection(row["attributes"]["pattern"])) if reason == "a" else None
        row["review"]["exact_duplicate_ids"] = [r["id"] for r in pixel_groups[row["target_pixel_sha256"]] if r["id"] != row["id"]]
        row["review"]["cross_split_exact_duplicate"] = len({r["split"] for r in pixel_groups[row["target_pixel_sha256"]]}) > 1
        if reason == "a":
            assert not row["review"]["cross_split_exact_duplicate"], "跨划分重复必须先排除"
    write_jsonl(args.output / "reviewed.jsonl", rows)
    for split in ["train", "validation"]:
        write_jsonl(args.output / f"{split}_accepted.jsonl", [r for r in rows if r["split"] == split and r["review"]["decision"] == "accept"])
    write_jsonl(args.output / "needs_review.jsonl", [r for r in rows if r["review"]["decision"] == "uncertain"])
    summary = {split: {"decisions": dict(Counter(r["review"]["decision"] for r in rows if r["split"] == split)), "reasons": dict(Counter(r["review"]["reason_code"] for r in rows if r["split"] == split)), "accepted_patterns": dict(Counter(p for r in rows if r["split"] == split and r["review"]["decision"] == "accept" for p in r["review"]["pattern_type_verified"]))} for split in ["train", "validation"]}
    summary["reason_legend"] = REASONS
    summary["limitations"] = ["488条均已查看四联图；初步通过29条另以原始分辨率复核，4条降级为待确认", "通过仅表示适合本轮严格小样本实验，不是人工双人金标准", "mask为自动粗分割，未做像素级修订", "跨划分近重复只做缩略图最近邻初筛及前16对视觉核对，不能保证全量无近重复", "原始划分保持；验证通过样本过少，不能据此评估泛化或否定方向"]
    (args.output / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def recheck(args):
    """通过候选以原始 256 像素显示，复核参考细节及 mask。"""
    rows = read_jsonl(args.output / "candidates.jsonl")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    selected = [r for r in rows if decisions[str(r["review_index"])].startswith("a")]
    for page in range((len(selected) + 5) // 6):
        sheet = Image.new("RGB", (2048, 864), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(selected[page * 6:page * 6 + 6]):
            x, y = j % 2 * 1024, j // 2 * 288
            draw.text((x + 2, y + 2), f'{row["review_index"]} {row["id"]}', fill="black")
            gt, ref, sk = [Image.open(args.root / p).convert("RGB") for p in [row["target_image"], row["texture_candidates"][0], row["sketch"]]]
            overlay = np.asarray(gt).copy()
            valid = np.asarray(Image.open(row["review_inner_mask"])) > 127
            overlay[valid] = (overlay[valid] * .65 + np.array([0, 200, 50]) * .35).astype(np.uint8)
            for k, im in enumerate([gt, ref, sk, Image.fromarray(overlay)]):
                im.thumbnail((256, 256))
                sheet.paste(im, (x + k * 256, y + 25))
        sheet.save(args.output / "sheets" / f"accepted_detail_{page:02d}.png")


def duplicates(args):
    """列出跨划分最相似的 GT 供视觉核对，距离不是自动去重结论。"""
    rows = read_jsonl(args.output / "candidates.jsonl")
    images = [Image.open(args.root / r["target_image"]).convert("RGB") for r in rows]
    thumbs = np.stack([np.asarray(im.resize((32, 32)), dtype=np.float32) for im in images])
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    pairs = []
    for j, r in enumerate(rows):
        if r["split"] != "validation":
            continue
        scores = np.abs(thumbs[train] - thumbs[j]).mean(axis=(1, 2, 3))
        k = int(np.argmin(scores))
        pairs.append({"train_index": train[k], "validation_index": j, "thumbnail_rgb_mae": float(scores[k])})
    pairs.sort(key=lambda x: x["thumbnail_rgb_mae"])
    (args.output / "cross_split_similarity_candidates.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (1024, 1120), "white")
    draw = ImageDraw.Draw(sheet)
    for pos, p in enumerate(pairs[:16]):
        x, y = pos % 4 * 256, pos // 4 * 280
        draw.text((x, y), f'{p["train_index"]} / {p["validation_index"]} MAE={p["thumbnail_rgb_mae"]:.2f}', fill="black")
        for col, idx in enumerate([p["train_index"], p["validation_index"]]):
            im = images[idx].copy()
            im.thumbnail((128, 240))
            sheet.paste(im, (x + col * 128, y + 30))
    sheet.save(args.output / "sheets/cross_split_similar.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "finalize", "recheck", "duplicates"])
    parser.add_argument("--root", type=Path, default=Path("F:/fuxian/dataset/datasets/BF"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/ctd_pattern_review_20260912"))
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "finalize": finalize, "recheck": recheck, "duplicates": duplicates}[args.action](args)
