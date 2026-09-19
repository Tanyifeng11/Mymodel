"""E14：先人工标注参考图，再审计冻结 E5 表示；不加载 U-Net、不训练生成模型。

python -m tools.e14_pattern_probe prepare --help
python -m tools.e14_pattern_probe extract --help
python -m tools.e14_pattern_probe evaluate --help
"""
import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PATTERNS = {
    "stripe": r"\bstrip(?:e|es|ed)\b",
    "plaid": r"\b(?:plaid|checkered|checked|gingham|tartan)\b",
    "dots": r"\b(?:polka|dotted|dots)\b",
    "floral": r"\b(?:floral|flowers?)\b",
    "solid": r"\b(?:solid|plain)\b",
}
FIELDS = ["sample_id", "texture", "caption", "candidate_labels", "pattern", "color_group",
          "source_group", "confirmed", "pattern_visible", "notes", "pixel_sha256"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(args):
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    rng = np.random.default_rng(args.seed)
    candidates = defaultdict(list)
    for row in rows:
        labels = [name for name, expr in PATTERNS.items() if re.search(expr, row.get("caption", ""), re.I)]
        for label in labels:
            candidates[label].append({**row, "candidate_labels": ";".join(labels)})
    selected = {}
    for label in PATTERNS:
        pool = candidates[label]
        for i in rng.permutation(len(pool))[:args.per_class]:
            selected[pool[i]["sample_id"]] = pool[i]
    # 原文件不覆盖，避免重跑清空人工标签。
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "thumbnails").mkdir()
    records, cards = [], []
    for i, row in enumerate(selected.values()):
        path = Path(args.data_root) / row["texture"]
        with Image.open(path) as im:
            image = im.convert("RGB")
        record = {key: "" for key in FIELDS}
        record.update({key: row.get(key, "") for key in ("sample_id", "texture", "caption", "candidate_labels")})
        record["pixel_sha256"] = pixel_hash(image)
        records.append(record)
        image.thumbnail((256, 256))
        image.save(out / "thumbnails" / f"{i:04d}.png")
        cards.append(f'<figure><img src="thumbnails/{i:04d}.png"><figcaption>'
                     f'{i:04d} {html.escape(record["sample_id"])}<br>'
                     f'候选：{html.escape(record["candidate_labels"])}<br>'
                     f'{html.escape(record["caption"])}</figcaption></figure>')
    with (out / "labels.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (out / "review.html").write_text(
        '<meta charset="utf-8"><style>main{display:flex;flex-wrap:wrap}figure{width:270px;margin:12px}'
        'img{width:256px;height:256px;object-fit:contain}figcaption{overflow-wrap:anywhere}</style>'
        '<h1>E14 参考图人工核对</h1><p>编辑 labels.csv：pattern 填 stripe/plaid/dots/floral/solid；'
        'color_group 填人工配平的调色板组；source_group 填独立来源组；'
        'confirmed 和 pattern_visible 均填 1 才入选（素色指确认没有图案）。'
        '同图裁剪、同款及改色图归入同一来源组。不确定的留空。候选词不是标签。</p><main>'
        + "\n".join(cards) + "</main>", encoding="utf-8")
    write_json(out / "candidate_summary.json", {
        "seed": args.seed, "manifest": str(Path(args.manifest).resolve()),
        "keyword_counts": {k: len(v) for k, v in candidates.items()}, "selected": len(records),
        "scope": "caption 仅筛候选；印花首轮限定 floral；标签需要人工确认。",
    })
    print(f"候选 {len(records)} 张；请核对 {out / 'review.html'} 并编辑 labels.csv", flush=True)


def read_labels(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["confirmed"].strip() == "1"
                and r["pattern_visible"].strip() == "1"]
    if not rows:
        raise ValueError("没有已人工确认的可用参考图；请先标注 labels.csv")
    ids, hashes = set(), {}
    for r in rows:
        for key in ("pattern", "color_group", "source_group"):
            r[key] = r[key].strip()
            if not r[key]:
                raise ValueError(f'{r["sample_id"]} 缺少 {key}')
        if r["pattern"] not in PATTERNS:
            raise ValueError(f'未知图案类别：{r["pattern"]}')
        if r["sample_id"] in ids:
            raise ValueError("sample_id 重复")
        ids.add(r["sample_id"])
        h = r.get("pixel_sha256")
        if h and h in hashes and hashes[h] != r["source_group"]:
            raise ValueError("完全相同参考图被标成不同 source_group")
        if h:
            hashes[h] = r["source_group"]
    return rows


def coverage(rows):
    groups = defaultdict(set)
    for r in rows:
        groups[(r["pattern"], r["color_group"])].add(r["source_group"])
    return [{"pattern": k[0], "color_group": k[1], "independent_sources": len(v)}
            for k, v in sorted(groups.items())]


def capture(bf, tcpm, inputs, neutral, caption):
    """只挂 hook；不重新实现或修改模型前向，CPU FP32 保存后再构造读出。"""
    import torch
    values, handles = {}, []

    def save(name, value):
        values[name] = value.detach().float().cpu()

    def hook(name):
        return lambda m, a, y: save(name, y)

    for i in range(1, 5):
        handles.append(getattr(bf, f"stage{i}").register_forward_hook(hook(f"cnn{i}_native")))
    for i, module in enumerate(bf.token_source_proj):
        name = "clip" if i == 0 else f"cnn{i}"
        handles.append(module.register_forward_pre_hook(lambda m, a, n=name: save(n + "_pool", a[0])))
        handles.append(module.register_forward_hook(hook(name + "_projected")))
    handles.append(bf.resampler.register_forward_pre_hook(lambda m, a: save("fused", a[1])))
    handles.append(bf.resampler.register_forward_hook(lambda m, a, y: save("resampler_raw", y[0])))
    handles.append(bf.token_norm.register_forward_pre_hook(lambda m, a: save("tokens_pre_ln", a[0])))
    try:
        with torch.inference_mode():
            tokens, _ = bf(**inputs)
            save("tokens_post_ln", tokens)
            save("tcpm_neutral", tcpm(tokens, neutral))
            save("tcpm_caption", tcpm(tokens, caption))
            save("cnn_projected_combined", torch.cat([values[f"cnn{i}_projected"] for i in range(1, 5)], dim=1))
    finally:
        for h in handles:
            h.remove()
    return values


def readouts(values, seed=42):
    import torch
    import torch.nn.functional as F
    result = {}
    shapes = {}
    for name, value in values.items():
        shapes[name] = list(value.shape)
        if value.ndim == 4:
            tokens = value.flatten(2).transpose(1, 2)[0]
            grid = F.adaptive_avg_pool2d(value, (16, 16))
            result[name + "__grid16"] = grid.flatten().numpy()
            # stage1/2 native flatten 过大；保留原生空间统计和采样，不宣称保存了完整特征。
            if name in ("cnn3_native", "cnn4_native"):
                result[name + "__flatten"] = value.flatten().numpy()
        else:
            tokens = value[0]
            result[name + "__flatten"] = tokens.flatten().numpy()
        result[name + "__meanstd"] = torch.cat((tokens.mean(0), tokens.std(0, unbiased=False))).numpy()
        # 每层同为16个token，以免 max matching 因候选数多而虚高；三个种子做敏感性检查。
        if len(tokens) < 16:
            raise ValueError("set matching 至少需要16个 token")
        for offset in range(3):
            idx = np.random.default_rng(seed + offset).choice(len(tokens), 16, replace=False)
            result[name + f"__set16_s{offset}"] = tokens[idx].numpy()
    return result, shapes


def image_baselines(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    hist = np.histogramdd(rgb.reshape(-1, 3), bins=8, range=((0, 1),) * 3)[0].ravel()
    result = {"image__color_hist": (hist / hist.sum()).astype(np.float32)}
    # 去直流分量的灰度频谱，在两个分辨率保留方向/频率信息；不是纹样身份指标。
    for size in (64, 128):
        gray = np.asarray(image.convert("L").resize((size, size)), dtype=np.float32) / 255
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
        result[f"image__gray_fft{size}"] = np.log1p(spectrum).astype(np.float32).ravel()
    return result


def extract(args):
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
    from checkpoint_utils import load_checkpoint_file
    from tools.e8_token_probe import build_conditioner

    rows = read_labels(args.labels)
    eligible = retrieval(np.zeros((len(rows), len(rows))), rows)
    if not eligible:
        raise ValueError("标注集没有同色、跨来源的正负样本组合；请先补齐覆盖")
    print(f"已确认 {len(rows)} 张；可比较查询 {len(eligible)} 张", flush=True)
    checkpoint = load_checkpoint_file(args.checkpoint)
    meta = checkpoint.get("meta", {})
    if meta.get("texture_mode", "patch_resampled") != "patch_resampled":
        raise ValueError("E14 要求 E5 patch_resampled checkpoint")
    state = checkpoint["bf_texture_conditioner"]
    if any(k.startswith(("film.", "nexus.", "text_guidance.")) for k in state):
        raise ValueError("请使用未叠加后续模块的 E5 checkpoint")
    if meta.get("use_aa_tcr_fuse", 0):
        raise ValueError("本探针不适用于 AA-TCR checkpoint")
    if state["resampler_queries"].shape[1] != 16:
        raise ValueError("本次 E14 预期 E5 的16个 texture tokens")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if args.device == "cpu" and args.dtype == "fp16":
        raise ValueError("CPU 检查请显式使用 --dtype fp32；正式实验使用 GPU FP16")
    bf, tcpm = build_conditioner(checkpoint, args.device, dtype)
    del checkpoint
    base = args.base_model or meta.get("pretrained_model_name_or_path")
    if not base:
        raise ValueError("请指定 --base-model")
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", local_files_only=True)
    text_encoder.to(device=args.device, dtype=dtype).eval().requires_grad_(False)
    try:
        vision = CLIPVisionModelWithProjection.from_pretrained(args.clip_model, local_files_only=True)
    except OSError:
        vision = CLIPVisionModelWithProjection.from_pretrained(
            args.clip_model, subfolder="models/image_encoder", local_files_only=True)
    vision.to(device=args.device, dtype=dtype).eval().requires_grad_(False)
    processor = CLIPImageProcessor()
    image_processor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "features").mkdir()

    def encode(prompt):
        ids = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").to(args.device)
        mask = ids.attention_mask if getattr(text_encoder.config, "use_attention_mask", False) else None
        return text_encoder(ids.input_ids, attention_mask=mask)[0]

    actual_hashes = {}
    with torch.inference_mode():
        neutral = encode(args.neutral_prompt)
        for i, row in enumerate(rows):
            with Image.open(Path(args.data_root) / row["texture"]) as im:
                image = im.convert("RGB")
            digest = pixel_hash(image)
            if row.get("pixel_sha256") and digest != row["pixel_sha256"]:
                raise ValueError("参考图已改变，请重新人工核对：" + row["sample_id"])
            if digest in actual_hashes and actual_hashes[digest] != row["source_group"]:
                raise ValueError("相同参考图不能分属不同来源组")
            actual_hashes[digest] = row["source_group"]
            row["pixel_sha256"] = digest
            clip_pixels = processor(images=[image], return_tensors="pt").pixel_values.to(args.device, dtype)
            clip = vision(clip_pixels,
                          output_hidden_states=True)
            texture = image_processor.preprocess([image], height=args.height, width=args.width).to(args.device, dtype)
            inputs = dict(clip_image_embeds=clip.image_embeds, clip_vision_tokens=clip.hidden_states[-1][:, 1:, :],
                          texture_images=texture * 2 - 1, text_embeds=neutral)
            values = capture(bf, tcpm, inputs, neutral, encode(row["caption"]))
            features, shapes = readouts(values, args.seed)
            features.update(image_baselines(image))
            # 记录真实预处理后的颜色基线，检测缩放/CLIP裁剪重新引入的颜色差异。
            clip_rgb = clip_pixels[0].float().cpu().numpy().transpose(1, 2, 0)
            clip_rgb = clip_rgb * np.asarray(processor.image_std) + np.asarray(processor.image_mean)
            cnn_rgb = texture[0].float().cpu().numpy().transpose(1, 2, 0)
            for name, pixels in (("clip_input", clip_rgb), ("cnn_input", cnn_rgb)):
                hist = np.histogramdd(np.clip(pixels, 0, 1).reshape(-1, 3), bins=8,
                                      range=((0, 1),) * 3)[0].ravel()
                features[name + "__color_hist"] = (hist / hist.sum()).astype(np.float32)
            if not all(np.isfinite(v).all() for v in features.values()):
                raise ValueError("特征含 NaN/Inf：" + row["sample_id"])
            row["feature_file"] = f"features/{i:04d}.npz"
            np.savez_compressed(out / row["feature_file"], **features)
            print(f"[E14] {i + 1}/{len(rows)} {row['sample_id']}", flush=True)
    write_json(out / "index.json", {"rows": rows, "coverage": coverage(rows), "shapes": shapes,
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "resolved_base_model": str(base), "bf_training": bf.training,
        "scope": "冻结 E5 BF/TCPM；无 U-Net；neutral 为受控文本，caption 为部署文本。",
        "complete": True})


def similarity(features):
    x = np.asarray(features, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    if x.ndim == 2:
        return x @ x.T
    if x.ndim != 3:
        raise ValueError("未知读出形状")
    result = np.zeros((len(x), len(x)), dtype=np.float32)
    for i in range(len(x)):
        for j in range(i, len(x)):
            pair = x[i] @ x[j].T
            result[i, j] = result[j, i] = (pair.max(0).mean() + pair.max(1).mean()) / 2
    return result


def retrieval(scores, rows):
    details = []
    for i, anchor in enumerate(rows):
        pool = [j for j, r in enumerate(rows) if r["color_group"] == anchor["color_group"]
                and r["source_group"] != anchor["source_group"]]
        pos = [j for j in pool if rows[j]["pattern"] == anchor["pattern"]]
        neg = [j for j in pool if rows[j]["pattern"] != anchor["pattern"]]
        if not pos or not neg:
            continue
        # 相同来源的多个 crop 先平均相似度，避免增加该来源在检索中的票数。
        units = defaultdict(list)
        for j in pool:
            units[(rows[j]["source_group"], rows[j]["pattern"])].append(j)
        candidates = [(key, float(scores[i, js].mean())) for key, js in units.items()]
        positive = np.array([s for (g, p), s in candidates if p == anchor["pattern"]])
        negative = np.array([s for (g, p), s in candidates if p != anchor["pattern"]])
        delta = positive[:, None] - negative[None, :]
        # 平分最高相似度并列，避免全常数特征按清单顺序取得虚假高分。
        top = max(s for key, s in candidates)
        tied = [p for (g, p), s in candidates if abs(s - top) < 1e-7]
        order = sorted(candidates, key=lambda v: -v[1])
        k = min(3, len(order))
        boundary = order[k - 1][1]
        higher = [(key, s) for key, s in order if s > boundary + 1e-7]
        boundary_ties = [(key, s) for key, s in order if abs(s - boundary) <= 1e-7]
        needed = k - len(higher)
        failures = sum(p != anchor["pattern"] for (g, p), s in boundary_ties)
        r3 = 1.0 if any(p == anchor["pattern"] for (g, p), s in higher) else (
            1 - (math.comb(failures, needed) / math.comb(len(boundary_ties), needed)
                 if failures >= needed else 0))
        details.append({"sample_id": anchor["sample_id"], "source_group": anchor["source_group"],
            "pattern": anchor["pattern"], "color_group": anchor["color_group"],
            "r1": sum(p == anchor["pattern"] for p in tied) / len(tied),
            "r1_chance": len(positive) / len(candidates),
            "r3": r3,
            "r3_chance": 1 - (math.comb(len(negative), k) / math.comb(len(candidates), k)
                              if len(negative) >= k else 0),
            "triplet": float(((delta > 1e-7) + 0.5 * (np.abs(delta) <= 1e-7)).mean()),
            "margin": float(delta.mean()), "positive_sources": len(positive), "negative_sources": len(negative)})
    return details


def aggregate(details):
    # 先按来源平均，再宏平均，避免重复裁剪扩大样本权重。
    grouped = defaultdict(list)
    for row in details:
        grouped[row["source_group"]].append(row)
    metrics = ("r1", "r1_chance", "r3", "r3_chance", "triplet", "margin")
    return {"eligible_queries": len(details), "eligible_sources": len(grouped),
            **{m: float(np.mean([np.mean([r[m] for r in rs]) for rs in grouped.values()]))
               if grouped else None for m in metrics}}


def linear_probe(x, rows, seed):
    # 可选辅助读出：所有预处理仅 fit 在训练折；同一 seed/标签保证跨层共用划分。
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.array([r["pattern"] for r in rows])
    groups = np.array([r["source_group"] for r in rows])
    if min(len(set(groups[y == c])) for c in set(y)) < 3:
        return {"status": "skipped: 每类至少需要3个独立来源"}
    outer = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    scores, splits = [], []
    for train, test in outer.split(x, y, groups):
        if set(y[train]) != set(y) or set(y[test]) != set(y):
            return {"status": "skipped: 分组后某折缺失类别"}
        inner = list(StratifiedGroupKFold(2, shuffle=True, random_state=seed).split(x[train], y[train], groups[train]))
        if any(set(y[train][a]) != set(y) for a, b in inner):
            return {"status": "skipped: 内层训练折缺失类别"}
        dim = min(32, x.shape[1], min(len(a) - 1 for a, b in inner))
        if dim < 1:
            return {"status": "skipped: 独立训练数据不足"}
        model = Pipeline([("scale", StandardScaler()), ("pca", PCA(dim, random_state=seed)),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        search = GridSearchCV(model, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner, scoring="balanced_accuracy", error_score="raise")
        search.fit(x[train], y[train])
        scores.append(float(balanced_accuracy_score(y[test], search.predict(x[test]))))
        splits.append({"train": train.tolist(), "test": test.tolist(), "pca_dim": dim,
                       "C": search.best_params_["clf__C"]})
    return {"status": "ok", "balanced_accuracy": float(np.mean(scores)), "fold_scores": scores,
            "splits": splits, "scope": "辅助粗类别读出；不控制颜色，不能单独证明纹样信息保留"}


def paired_changes(details, seed):
    pairs = [(f"cnn{i}_native", f"cnn{i}_pool") for i in range(1, 5)]
    pairs += [("fused", "resampler_raw"), ("resampler_raw", "tokens_pre_ln"),
              ("tokens_pre_ln", "tokens_post_ln"), ("tokens_post_ln", "tcpm_neutral"),
              ("tokens_post_ln", "tcpm_caption")]
    result = []
    for before, after in pairs:
        for readout in ("meanstd", "flatten", "set16_s0", "set16_s1", "set16_s2"):
            a, b = before + "__" + readout, after + "__" + readout
            if a not in details or b not in details:
                continue
            reference = {r["sample_id"]: r for r in details[a]}
            for metric in ("r1", "triplet"):
                groups = defaultdict(list)
                for row in details[b]:
                    if row["sample_id"] in reference:
                        groups[row["source_group"]].append(row[metric] - reference[row["sample_id"]][metric])
                delta = np.array([np.mean(v) for v in groups.values()])
                if not len(delta):
                    continue
                rng = np.random.default_rng(seed)
                ci = None
                if len(delta) >= 3:
                    draws = rng.choice(delta, size=(1000, len(delta)), replace=True).mean(axis=1)
                    ci = np.quantile(draws, [0.025, 0.975]).tolist()
                result.append({"before": a, "after": b, "metric": metric,
                               "after_minus_before": float(delta.mean()), "source_count": len(delta),
                               "query_source_bootstrap_ci95": ci})
    return result


def evaluate(args):
    root = Path(args.features)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows = index["rows"]
    with np.load(root / rows[0]["feature_file"]) as f:
        keys = sorted(f.files)
    summaries, details, probes = {}, {}, {}
    for key in keys:
        values = []
        for row in rows:
            with np.load(root / row["feature_file"]) as f:
                values.append(f[key])
        x = np.stack(values)
        details[key] = retrieval(similarity(x), rows)
        summaries[key] = aggregate(details[key])
        if args.linear and (key.endswith("__meanstd") or key.startswith("image__")):
            probes[key] = linear_probe(x, rows, args.seed)
        print(key, summaries[key], flush=True)
    write_json(root / "report.json", {"coverage": coverage(rows), "retrieval": summaries,
        "per_query": details, "linear": probes, "paired_changes": paired_changes(details, args.seed),
        "interpretation": "仅粗图案可读性；不同层的 margin 不直接比较。set16 无位置信息；"
                          "grid16 为受控降采样读出，native 为原生分辨率。零有效查询表示数据不支持比较。"
                          "CI 仅重采样查询来源、固定检索库，不能视为总体泛化置信区间。"
                          "比较跨层 R1/triplet 须配对查看同一来源；不以固定降幅判定信息永久丢失。"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="筛选候选并生成参考图预览、待人工标注 CSV")
    p.add_argument("--manifest", default="data/processed/bf_full_audit_v1/validation_clean.json")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=prepare)
    p = sub.add_parser("extract", help="仅提取已人工确认样本的冻结 E5 特征")
    p.add_argument("--labels", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model")
    p.add_argument("--clip-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--neutral-prompt", default="a garment")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=extract)
    p = sub.add_parser("evaluate", help="同色跨来源检索；可选嵌套分组线性探针")
    p.add_argument("--features", required=True)
    p.add_argument("--linear", action="store_true", help="需要 scikit-learn；只训练辅助分类器")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
