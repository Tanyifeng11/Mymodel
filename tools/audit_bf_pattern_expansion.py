"""完整 BF 条纹/格纹抽审：只检查参考信息，不用 mask 质量淘汰样本。"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def save_lines(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def fingerprint(im):
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sheets").mkdir()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    previous = read_lines(args.previous)
    excluded = {Path(r["target_image"]).stem for r in previous}
    seen_hash = {r["target_pixel_sha256"] for r in previous}
    plaid = re.compile(r"plaid|tartan|checkered|checked|gingham", re.I)
    stripe = re.compile(r"strip", re.I)
    pools = {"stripe": [], "plaid": []}
    counts = Counter()
    for r in data:
        caption = str(r["caption"])
        group = "plaid" if plaid.search(caption) else ("stripe" if stripe.search(caption) else None)
        if not group:
            continue
        counts[group + "_before_exclusion"] += 1
        if Path(r["cloth"]).stem not in excluded:
            pools[group].append(r)
    rng = random.Random(42)
    rows = []
    for group in ["stripe", "plaid"]:
        pool = sorted(pools[group], key=lambda r: r["cloth"])
        counts[group + "_after_id_exclusion"] = len(pool)
        rng.shuffle(pool)
        selected = 0
        for r in pool:
            gt = Image.open(args.root / r["cloth"]).convert("RGB")
            key = fingerprint(gt)
            if key in seen_hash:
                counts[group + "_sampled_pixel_duplicates_skipped"] += 1
                continue
            seen_hash.add(key)
            rows.append(dict(r, review_index=len(rows), candidate_group=group, split="training", target_pixel_sha256=key, mask_review="未评估；不参与本轮信息质量分类"))
            selected += 1
            if selected == 100:
                break
        assert selected == 100
    save_lines(args.output / "sampled_200.jsonl", rows)
    info = {"seed": 42, "manifest": str(args.manifest.resolve()), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "previous": str(args.previous.resolve()), "previous_sha256": hashlib.sha256(args.previous.read_bytes()).hexdigest(), "counts": dict(counts), "sampling_rule": "caption关键词候选；同时命中时分到plaid；各组排序后随机打乱；排除上一轮ID和已知像素重复；仅训练集", "limits": "只排查被抽中图片与上轮/本轮的完全像素重复，未排查整个BF近重复"}
    (args.output / "sampling_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    for page in range((len(rows) + 23) // 24):
        sheet = Image.new("RGB", (1536, 1792), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(rows[page * 24:page * 24 + 24]):
            x, y = j % 3 * 512, j // 3 * 224
            draw.text((x + 2, y + 2), f'{row["review_index"]:03d} {row["candidate_group"]} {Path(row["cloth"]).stem}', fill="black")
            for key, dx, width in [("cloth", 0, 192), ("texture", 192, 192), ("sketch", 384, 128)]:
                im = Image.open(args.root / row[key]).convert("RGB")
                im.thumbnail((width, 192))
                sheet.paste(im, (x + dx, y + 24))
        sheet.save(args.output / "sheets" / f"page_{page:02d}.png")
    print(json.dumps(info, ensure_ascii=True, indent=2))


LEGEND = {
    "a": ["original_reference_usable", "原参考可辨认重复纹样，并与GT可见纹样对应；仅信息层面通过"],
    "b": ["recrop_promising", "原参考局部过小、模糊或混入非纹样成分，但GT存在可辨认纹样区域，值得试裁；尚未证明重裁成功"],
    "c": ["gt_detail_insufficient", "GT中的目标纹样也过细、过淡或模糊，难以确认可裁出有效参考"],
    "d": ["outside_pattern_scope", "关键词命中但主要是罗纹、褶皱、局部装饰或非本轮目标图案，不能视为原图质量差"],
    "u": ["uncertain", "纹样类别、参考对应或GT可重裁性不确定，需进一步核对"],
}


def finalize(args):
    rows = read_lines(args.output / "sampled_200.jsonl")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    assert set(decisions) == {str(i) for i in range(200)}
    for row in rows:
        code = decisions[str(row["review_index"])]
        status, reason = LEGEND[code[0]]
        row["review"] = {"decision": status, "reason": reason, "sketch_pattern": {"0": "none", "1": "partial", "2": "strong"}[code[1]], "coverage": {"w": "主要区域一致", "m": "局部/拼接/多种图案", "u": "未确认"}[code[2]], "reviewer": "Codex视觉抽审；非独立双人标注", "training_ready": False}
    save_lines(args.output / "reviewed_200.jsonl", rows)
    for status, _ in LEGEND.values():
        save_lines(args.output / (status + ".jsonl"), [r for r in rows if r["review"]["decision"] == status])
    summary = {group: dict(Counter(r["review"]["decision"] for r in rows if r["candidate_group"] == group)) for group in ["stripe", "plaid"]}
    summary["total"] = dict(Counter(r["review"]["decision"] for r in rows))
    summary["legend"] = LEGEND
    summary["limits"] = ["重裁有希望是视觉判断，没有实际重裁或训练验证", "未审核mask，原参考信息通过不等于整件服装统一监督可用", "关键词分层抽样结果不能外推为全部BF合格率", "与上一轮严格通过数标准不同，不应直接比较通过率"]
    (args.output / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def detail(args):
    rows = read_lines(args.output / "sampled_200.jsonl")
    indices = [int(i) for i in args.indices.split(",")]
    for page in range((len(indices) + 9) // 10):
        sheet = Image.new("RGB", (1280, 1400), "white")
        draw = ImageDraw.Draw(sheet)
        for j, idx in enumerate(indices[page * 10:page * 10 + 10]):
            r = rows[idx]
            x, y = j % 2 * 640, j // 2 * 280
            draw.text((x, y), str(idx) + " " + Path(r["cloth"]).stem, fill="black")
            for key, dx, width in [("cloth", 0, 256), ("texture", 256, 256), ("sketch", 512, 128)]:
                im = Image.open(args.root / r[key]).convert("RGB")
                im.thumbnail((width, 256))
                sheet.paste(im, (x + dx, y + 24))
        sheet.save(args.output / "sheets" / f"detail_{page:02d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "finalize", "detail"])
    parser.add_argument("--root", type=Path, default=Path("F:/fuxian/dataset/datasets/BF/training"))
    parser.add_argument("--manifest", type=Path, default=Path("data/train_bf_texture.json"))
    parser.add_argument("--previous", type=Path, default=Path("eval_outputs/ctd_pattern_review_20260912/reviewed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/bf_pattern_expansion_200_seed42"))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--indices")
    args = parser.parse_args()
    {"prepare": prepare, "finalize": finalize, "detail": detail}[args.action](args)
