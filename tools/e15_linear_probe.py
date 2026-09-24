"""E15 线性探针：冻结权重、不训练，检验 16 个 texture token 还能解码哪些参考图信息。

与 E15-D1 共用同一份 BF 预训练权重与同一预处理：D1 回答“条件改变时表示变化多大”，
本工具回答“表示里还剩多少可线性解码的参考信息”，两者共同量化 fused -> 16 token 的压缩。

python -m tools.e15_linear_probe --manifest ... --data-root ... --checkpoint ... \
    --base-model ... --clip-model ... --labels ... --output ...
"""

import argparse
import csv
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.e15_common import (LayerCapture, compact_table, flatten_representation,
                              gram_statistics, write_json)

# 只探测压缩前后的关键层；cnn1-4 的响应强度已由 E15-D1 覆盖。
PROBE_LAYERS = ["clip_patch", "fused", "resampler_out", "mlp_out", "pre_ln", "final"]
# cnn1-4 单样本约 5.7MB，只做内存探针、不落盘，用于定位方向信息在哪一级消失。
EXTRA_LAYERS = ["cnn1", "cnn2", "cnn3", "cnn4"]
PROBED_LAYERS = PROBE_LAYERS + EXTRA_LAYERS
TASKS = ("color", "orientation", "pattern")
IMAGE_FEATURES = ["image__color_hist", "image__gray_fft128"]
COLOR_BASELINE = "lab_mean__3d"
AXES = (("vertical", 0.0), ("horizontal", 90.0))
ORIENTATION_TOLERANCE = 25.0


def build_dataset(args):
    from transformers import CLIPTokenizer

    from train_texture_adapter import MyDataset

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer",
                                              local_files_only=True)
    return MyDataset(args.manifest, tokenizer, height=512, width=384,
                     image_root_path=args.data_root, texture_preprocess_mode="plain_resize",
                     t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)


def read_labelled(path):
    """返回 {参考图相对路径: 人工确认的纹样类别}。"""
    labelled = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("confirmed") == "1" and row.get("pattern"):
                labelled[row["texture"]] = row["pattern"]
    if not labelled:
        raise ValueError("标注文件没有已确认的纹样类别：%s" % path)
    return labelled


def select_indices(dataset, count, seed, labelled):
    """固定随机抽样与全部已标注样本的并集，保证两个任务都有足够样本。"""
    if not 8 <= count <= len(dataset):
        raise ValueError("样本数不合法：%d" % count)
    picked = set(random.Random(seed).sample(range(len(dataset)), count))
    location = {row.get("texture", row.get("color")): index
                for index, row in enumerate(dataset.data)}
    tagged = []
    for texture in sorted(labelled):
        if texture not in location:
            raise ValueError("标注样本不在清单内：%s" % texture)
        tagged.append(location[texture])
    return sorted(picked | set(tagged)), tagged


def orientation_angle(gray, band=(0.06, 0.45), smooth=3):
    """灰度图高频能量方向谱；返回 (主方向角, 各向异性)。

    角度定义在频率平面：能量落在 x 频率轴(0°)表示图案沿 x 变化，即竖条纹；
    落在 y 频率轴(90°)表示横条纹。各向异性是方向谱的 (max-min)/mean。
    """
    height, width = gray.shape
    spectrum = np.abs(np.fft.fft2(gray - gray.mean()))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    keep = (radius >= band[0]) & (radius <= band[1])
    angle = np.degrees(np.arctan2(fy, fx)) % 180.0
    index = np.clip(np.digitize(angle[keep], np.arange(1.0, 181.0)), 0, 179)
    energy = np.zeros(180)
    np.add.at(energy, index, spectrum[keep])
    kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    padded = np.concatenate([energy[-smooth:], energy, energy[:smooth]])
    energy = np.convolve(padded, kernel, "same")[smooth:-smooth]
    return int(np.argmax(energy)), float((energy.max() - energy.min()) / (energy.mean() + 1e-12))


def axis_label(angle, tolerance=ORIENTATION_TOLERANCE):
    """只保留接近水平/垂直轴的方向；斜纹与无方向样本返回 None。"""
    for name, target in AXES:
        delta = abs(angle - target)
        if min(delta, 180.0 - delta) <= tolerance:
            return name
    return None


def image_labels(tensor):
    """tensor 为 CNN 分支输入 [3,H,W]（归一化到 [-1,1]），标签全部由该视图计算。"""
    from PIL import Image
    from skimage.color import rgb2lab

    from tools.e14_pattern_probe import image_baselines

    rgb = ((tensor.float() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float64)
    angle, anisotropy = orientation_angle(rgb.mean(axis=2))
    baselines = image_baselines(Image.fromarray((rgb * 255.0).round().astype(np.uint8)))
    return {
        "lab_mean": rgb2lab(rgb).reshape(-1, 3).mean(0).tolist(),
        "angle_deg": float(angle),
        "anisotropy": float(anisotropy),
        "axis": axis_label(angle) or "",
        "image__color_hist": baselines["image__color_hist"].astype(np.float16),
        "image__gray_fft128": baselines["image__gray_fft128"].astype(np.float16),
    }


def extract(args, ctx, dataset, indices, output):
    """逐样本捕获冻结表示的指定层；同时算像素域基线与标签。"""
    import torch

    capture = LayerCapture(ctx["bf"])
    features_dir = output / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels, model_features, image_features = {}, {}, {}
    with torch.inference_mode():
        for index in indices:
            batch = dataset[index]
            texture = batch["texture_image"]
            capture.reset()
            visual = ctx["vision"](batch["clip_texture_image"].to(ctx["device"], ctx["dtype"]),
                                   output_hidden_states=True)
            ctx["model"].get_texture_condition_tokens(
                visual, texture[None].to(ctx["device"], ctx["dtype"]))
            captured = dict(capture.current)
            captured["clip_patch"] = visual.hidden_states[-1][:, 1:, :]
            arrays = {name: flatten_representation(captured[name]).astype(np.float16)
                      for name in PROBED_LAYERS}
            np.savez(features_dir / ("%05d.npz" % index),
                     **{name: arrays[name] for name in PROBE_LAYERS})
            model_features[index] = arrays
            row = dataset.data[index]
            entries = image_labels(texture)
            image_features[index] = {name: entries[name] for name in IMAGE_FEATURES}
            labels[index] = {name: value for name, value in entries.items()
                             if name not in IMAGE_FEATURES}
            labels[index]["texture"] = row.get("texture", row.get("color"))
            print("probe sample %d done" % index, flush=True)
    capture.remove()
    return labels, model_features, image_features


def gram_blocks(train, test):
    """用训练折均值方差标准化后的线性核矩阵；测试折只做同一变换。"""
    train = np.asarray(train, dtype=np.float32)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    centred = (train - mean) / scale
    k_train = centred @ centred.T
    if test is None:
        return k_train, None
    other = (np.asarray(test, dtype=np.float32) - mean) / scale
    return k_train, other @ centred.T


def spectral_basis(k_train):
    """训练折核矩阵的特征分解，供所有 alpha/C 复用。"""
    weights, vectors = np.linalg.eigh(k_train)
    weights, vectors = weights[::-1], vectors[:, ::-1]
    keep = weights > max(float(weights[0]), 1e-12) * 1e-10
    return np.clip(weights[keep], 0.0, None), vectors[:, keep]


def ridge_predict(weights, vectors, k_test, target, alpha):
    """核形式岭回归；标准化后特征均值为零，故截距取训练标签均值。"""
    centred = k_test @ ((vectors * (1.0 / (weights + alpha))) @ (vectors.T @ target))
    return centred + target.mean(axis=0)


def kernel_pca(weights, vectors, k_test, components):
    """核空间 PCA 投影，用于线性分类探针；等价于在训练折上做 PCA。"""
    width = int(min(components, weights.size))
    singular = np.sqrt(weights[:width])
    vectors = vectors[:, :width]
    return vectors * singular, (k_test @ vectors) / singular


def prepare_folds(matrix, target, folds, seed, stratified):
    """缓存每一折的核矩阵与谱分解；标签置换复用同一批折。"""
    from sklearn.model_selection import KFold, StratifiedKFold

    splitter = (StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                if stratified else KFold(n_splits=folds, shuffle=True, random_state=seed))
    prepared = []
    for train, test in splitter.split(matrix, target if stratified else None):
        k_train, k_test = gram_blocks(matrix[train], matrix[test])
        weights, vectors = spectral_basis(k_train)
        prepared.append({"train": train, "test": test, "k_train": k_train, "k_test": k_test,
                         "weights": weights, "vectors": vectors})
    return prepared


def regression_scores(prepared, target, alphas, seed, inner=3):
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    prediction = np.zeros_like(target)
    chosen = []
    for fold in prepared:
        k_train, train = fold["k_train"], fold["train"]
        inner_splits = list(KFold(n_splits=inner, shuffle=True, random_state=seed + 1).split(train))
        best, best_score = float(alphas[0]), -np.inf
        for alpha in alphas:
            scores = []
            for inner_train, inner_test in inner_splits:
                sub_weights, sub_vectors = spectral_basis(k_train[np.ix_(inner_train, inner_train)])
                guess = ridge_predict(sub_weights, sub_vectors,
                                      k_train[np.ix_(inner_test, inner_train)],
                                      target[train][inner_train], alpha)
                scores.append(r2_score(target[train][inner_test], guess))
            if float(np.mean(scores)) > best_score:
                best, best_score = float(alpha), float(np.mean(scores))
        chosen.append(best)
        prediction[fold["test"]] = ridge_predict(fold["weights"], fold["vectors"], fold["k_test"],
                                                 target[train], best)
    return {"r2": float(r2_score(target, prediction, multioutput="uniform_average")),
            "per_target_r2": [float(r2_score(target[:, channel], prediction[:, channel]))
                              for channel in range(target.shape[1])],
            "alpha": chosen, "samples": int(len(target))}


def classification_scores(prepared, target, cs, seed, components, inner=3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    prediction = np.empty(len(target), dtype=object)
    chosen = []
    for fold in prepared:
        train, test = fold["train"], fold["test"]
        train_projection, test_projection = kernel_pca(fold["weights"], fold["vectors"],
                                                       fold["k_test"], components)
        inner_splits = list(StratifiedKFold(n_splits=inner, shuffle=True,
                                            random_state=seed + 1).split(train_projection,
                                                                         target[train]))
        best, best_score = float(cs[0]), -np.inf
        for value in cs:
            model = LogisticRegression(C=float(value), max_iter=5000)
            score = cross_val_score(model, train_projection, target[train], cv=inner_splits,
                                    scoring="balanced_accuracy").mean()
            if float(score) > best_score:
                best, best_score = float(value), float(score)
        chosen.append(best)
        model = LogisticRegression(C=best, max_iter=5000).fit(train_projection, target[train])
        prediction[test] = model.predict(test_projection)
    predicted = prediction.astype(target.dtype)
    names, counts = np.unique(target, return_counts=True)
    return {"balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
            "accuracy": float((predicted == target).mean()),
            "classes": {str(name): int(count) for name, count in zip(names, counts)},
            "c": chosen, "samples": int(len(target))}


def permutation_control(prepared, target, cs, seed, components, repeats):
    """标签置换对照：折结构不变，只打乱标签。"""
    values = []
    for offset in range(repeats):
        shuffled = np.random.default_rng(seed + offset).permutation(target)
        values.append(classification_scores(prepared, shuffled, cs, seed, components)
                      ["balanced_accuracy"])
    return {"mean": float(np.mean(values)), "values": values}


def describe(matrix):
    stats = gram_statistics(np.asarray(matrix, dtype=np.float64))
    return {"dimension": int(matrix.shape[1]), "effective_rank": float(stats["effective_rank"]),
            "pc1_share": float(stats["pc_variance"][0]) if stats["pc_variance"] else None}


def classification_task(entry, features, positions, target, folds, args, cs, description):
    entry.update({"metric": "balanced_accuracy", "target": description, "samples": int(len(target)),
                  "folds": folds,
                  "classes": {str(name): int(count) for name, count in
                              zip(*np.unique(target, return_counts=True))}, "results": {}})
    for name, matrix in features.items():
        subset = matrix[positions]
        prepared = prepare_folds(subset, target, folds, args.seed, stratified=True)
        result = classification_scores(prepared, target, cs, args.seed, args.components)
        result["permutation"] = permutation_control(prepared, target, cs, args.seed,
                                                    args.components, args.permutations)
        entry["results"][name] = result
    return entry


def analyse(args, indices, labels, features, output, tasks=TASKS):
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    cs = [0.03, 0.1, 0.3, 1.0, 3.0]
    location = {index: position for position, index in enumerate(indices)}
    report = {"protocol": {"manifest": args.manifest, "checkpoint": args.checkpoint,
                           "samples": len(indices), "seed": args.seed,
                           "random_count": args.count, "folds": args.folds,
                           "permutations": args.permutations,
                           "components": args.components,
                           "orientation_tolerance_deg": ORIENTATION_TOLERANCE,
                           "note": "全部为线性探针，不训练生成模型；标签置换为对照。"},
              "features": {name: describe(matrix) for name, matrix in features.items()},
              "tasks": {}}

    if "color" not in tasks:
        return report
    colour_target = np.array([labels[key]["lab_mean"] for key in indices], dtype=np.float64)
    print("task color_lab_mean", flush=True)
    entry = {"metric": "r2", "target": "参考图 Lab 均值", "samples": len(indices), "results": {}}
    for name, matrix in features.items():
        if name == COLOR_BASELINE:
            continue
        prepared = prepare_folds(matrix, colour_target, args.folds, args.seed, stratified=False)
        entry["results"][name] = regression_scores(prepared, colour_target, alphas, args.seed)
    report["tasks"]["color_lab_mean"] = entry

    if "orientation" not in tasks:
        return report
    ordered = sorted(indices, key=lambda key: -labels[key]["anisotropy"])
    orientation = [key for key in ordered if labels[key]["axis"]][:args.orientation_count]
    if len(orientation) < 40:
        raise ValueError("方向子集样本不足：%d" % len(orientation))
    print("task orientation_axis (%d samples)" % len(orientation), flush=True)
    report["tasks"]["orientation_axis"] = classification_task(
        {}, features, [location[key] for key in orientation],
        np.array([labels[key]["axis"] for key in orientation]), args.folds, args, cs,
        "竖条纹 vs 横条纹（各向异性 top %d）" % len(orientation))

    if "pattern" not in tasks:
        return report
    tagged = [key for key in indices if labels[key].get("pattern")]
    report["protocol"]["labelled_samples"] = len(tagged)
    positions = [location[key] for key in tagged]
    patterns = np.array([labels[key]["pattern"] for key in tagged])
    folds = int(min(args.folds, int(np.bincount(np.unique(patterns, return_inverse=True)[1]).min())))
    print("task pattern_class (%d samples, %d folds)" % (len(patterns), folds), flush=True)
    report["tasks"]["pattern_class"] = classification_task(
        {}, features, positions, patterns, folds, args, cs, "五类纹样（人工确认标注）")
    binary = np.where(patterns == "solid", "solid", "other")
    print("task pattern_solid_binary (%d samples)" % len(binary), flush=True)
    report["tasks"]["pattern_solid_binary"] = classification_task(
        {}, features, positions, binary, args.folds, args, cs, "solid vs 非 solid 纹样")
    return report


def summarise(report, output):
    lines = ["# E15 线性探针（冻结权重、不训练）", "",
             "- 样本 %d（随机 %d + 已确认纹样 %d），折数 %d，置换 %d 次。"
             % (report["protocol"]["samples"], report["protocol"]["random_count"],
                report["protocol"].get("labelled_samples", 0), report["protocol"]["folds"],
                report["protocol"]["permutations"]),
             "- 特征全部来自冻结的 BF 纹理条件分支；分类探针先做核空间 PCA（<= %d 维）。"
             % report["protocol"]["components"], ""]
    if "color_lab_mean" in report["tasks"]:
        rows = [[name, report["features"][name]["dimension"],
                 report["features"][name]["effective_rank"], item["r2"]]
                for name, item in report["tasks"]["color_lab_mean"]["results"].items()]
        lines += ["## 颜色：Lab 均值回归（R^2）", "", "```",
                  compact_table(["feature", "dim", "erank", "R2"], rows), "```", ""]
    for task in ("orientation_axis", "pattern_class", "pattern_solid_binary"):
        if task not in report["tasks"]:
            continue
        entry = report["tasks"][task]
        rows = [[name, report["features"][name]["dimension"],
                 report["features"][name]["effective_rank"], item["balanced_accuracy"],
                 item["permutation"]["mean"]] for name, item in entry["results"].items()]
        lines += ["## %s：%s（balanced accuracy，n=%d，%d 折）"
                  % (task, entry["target"], entry["samples"], entry["folds"]), "",
                  "```", compact_table(["feature", "dim", "erank", "bal_acc", "perm"], rows),
                  "```", ""]
    text = "\n".join(lines) + "\n"
    (Path(output) / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def run(args):
    args.tasks = tuple(name for name in args.tasks.split(",") if name)
    unknown = set(args.tasks) - set(TASKS)
    if unknown or not args.tasks:
        raise ValueError("不支持的探针任务：%s" % sorted(unknown))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    labelled = read_labelled(args.labels)
    indices, tagged = select_indices(dataset, args.count, args.seed, labelled)
    print("samples %d (tagged %d)" % (len(indices), len(tagged)), flush=True)
    from tools.e15_diagnosis import build_models

    ctx = {"args": vars(args), "output": output}
    build_models(SimpleNamespace(checkpoint=args.checkpoint, base_model=args.base_model,
                                 clip_model=args.clip_model, device=args.device), ctx)
    labels, model_features, image_features = extract(args, ctx, dataset, indices, output)
    del ctx["model"], ctx["unet"], ctx["vae"], ctx["text"], ctx["vision"]
    import torch

    torch.cuda.empty_cache()
    features = {name: np.stack([model_features[index][name] for index in indices])
                for name in PROBED_LAYERS}
    for name in IMAGE_FEATURES:
        features[name] = np.stack([image_features[index][name] for index in indices])
    features[COLOR_BASELINE] = np.array([labels[index]["lab_mean"] for index in indices],
                                        dtype=np.float16)
    for index in indices:
        labels[index]["pattern"] = labelled.get(labels[index]["texture"], "")
    del model_features, image_features
    report = analyse(args, indices, labels, features, output, args.tasks)
    write_json(output / "report.json", report)
    write_json(output / "labels.json", {str(index): labels[index] for index in indices})
    summarise(report, output)
    print("E15 linear probe finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "data-root", "checkpoint", "base-model", "clip-model", "labels",
                 "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=256, help="随机验证样本数")
    parser.add_argument("--orientation-count", type=int, default=120, help="方向任务样本数上限")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--components", type=int, default=64, help="分类探针保留的核主成分")
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default=",".join(TASKS), help="逗号分隔：color/orientation/pattern")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
