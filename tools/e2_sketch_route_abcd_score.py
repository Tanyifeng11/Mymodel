"""服装草图消融的配对频谱与结构读出。运行 python -m tools.e2_sketch_route_abcd_score --help。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from eval.eval_utils import estimate_foreground_mask
from garment_mask_utils import mask_backend_info


ARMS = ("A", "B", "C", "D")
DIRECTIONS = ("horizontal", "vertical")
METRICS = ("both_directions_correct", "direction_switch_response", "orientation_accuracy",
           "readout_coverage", "target_band_fraction", "protected_line_recall",
           "contour_f1", "silhouette_iou")


def morphology(mask, size, grow=False):
    operation = ImageFilter.MaxFilter if grow else ImageFilter.MinFilter
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(operation(size))) > 127


def tiles_inside(mask, size):
    # 与 E14 相同，避开边界，按半块步长在固定网格中选择完整的衣内块。
    interior = morphology(mask, 9)
    height, width = mask.shape
    return [(x, y, size, size)
            for y in range(0, height - size + 1, size // 2)
            for x in range(0, width - size + 1, size // 2)
            if interior[y:y+size, x:x+size].all()]


def stripe_response(image, tiles, period, direction):
    # 用原生像素的衣内方块。横纹的能量在纵向频率轴；三个颜色通道分别去均值。
    if not tiles:
        return dict(readable=False, correct=False, vertical_score=None, vertical_band_contrast=None, target_band_fraction=None,
                    rms=None, tile_count=0)
    size = tiles[0][2]
    rgb = np.asarray(image, dtype=np.float64) / 255
    window = np.outer(np.hanning(size), np.hanning(size))
    power = np.zeros((size, size))
    variance = []
    for x, y, w, h in tiles:
        patch = rgb[y:y+h, x:x+w]
        patch = patch - patch.mean(axis=(0, 1))
        variance.append(np.mean(patch**2))
        power += (abs(np.fft.fft2(patch * window[:, :, None], axes=(0, 1)))**2).sum(axis=2)
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    radius = np.hypot(fx, fy)
    band = (radius >= 2 / size) & (radius <= .5)
    near = band & (abs(radius - 1 / period) <= max(1 / size, .2 / period))
    slope = np.tan(np.deg2rad(22.5))
    vertical = power[near & (abs(fy) <= slope * abs(fx))].sum()
    horizontal = power[near & (abs(fx) <= slope * abs(fy))].sum()
    total = float(power[band].sum())
    signed = float((vertical - horizontal) / max(vertical + horizontal, 1e-20))
    fraction = float((vertical if direction == "vertical" else horizontal) / max(total, 1e-20))
    rms = float(np.sqrt(np.mean(variance)))
    readable = bool(rms >= .02 and max(vertical, horizontal) / max(total, 1e-20) >= .15
                    and abs(signed) >= .3)
    return dict(readable=readable, correct=bool(readable and (signed > 0) == (direction == "vertical")),
                vertical_score=signed, vertical_band_contrast=float((vertical-horizontal)/max(total, 1e-20)),
                target_band_fraction=fraction, rms=rms, tile_count=len(tiles))


def shape_response(image, mask, protected):
    predicted = estimate_foreground_mask(image, image.size)
    actual_edge = predicted & ~morphology(predicted, 3)
    expected_edge = mask & ~morphology(mask, 3)
    precision = (actual_edge & morphology(expected_edge, 5, True)).sum() / max(1, actual_edge.sum())
    recall = (expected_edge & morphology(actual_edge, 5, True)).sum() / max(1, expected_edge.sum())
    # 只统计预先保留的衣内线。纹样本身也可能产生边缘，因此这是结构保护的代理量。
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    gy, gx = np.gradient(gray)
    detected = morphology(np.hypot(gx, gy) >= .08, 5, True)
    return dict(protected_line_recall=float(detected[protected].mean()) if protected.any() else None,
                contour_f1=float(2 * precision * recall / max(precision + recall, 1e-12)),
                silhouette_iou=float((predicted & mask).sum() / max((predicted | mask).sum(), 1)))


def mean(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def measure_pair(horizontal, vertical):
    rows = (horizontal, vertical)
    # 以整个频段作分母，错周期时少量旁瓣不会造成接近满分的切换响应。
    hs, vs = horizontal["vertical_band_contrast"], vertical["vertical_band_contrast"]
    result = dict(both_directions_correct=all(r["correct"] for r in rows),
                  direction_switch_response=(vs - hs) / 2 if hs is not None and vs is not None else None,
                  orientation_accuracy=mean(r["correct"] for r in rows),
                  readout_coverage=mean(r["readable"] for r in rows))
    for key in ("target_band_fraction", "protected_line_recall", "contour_f1", "silhouette_iou"):
        result[key] = mean(r[key] for r in rows)
    return result


def summarize(pairs):
    # 一个身份先跨种子聚合。对比先在同一身份同一种子内计算，再汇总有效身份。
    contrasts = {"B-A": {"B": 1, "A": -1}, "C-A": {"C": 1, "A": -1},
                 "D-A": {"D": 1, "A": -1}, "D-C": {"D": 1, "C": -1}, "D-B": {"D": 1, "B": -1},
                 "D-C-B+A": {"D": 1, "C": -1, "B": -1, "A": 1}}
    by_label = {**{arm: {arm: 1} for arm in ARMS}, **contrasts}
    result = {}
    for label, coefficients in by_label.items():
        result[label] = {}
        for metric in METRICS:
            values = defaultdict(list)
            for record in pairs:
                arms = record["arms"]
                if all(arms[arm][metric] is not None for arm in coefficients):
                    value = sum(weight * arms[arm][metric] for arm, weight in coefficients.items())
                    values[record["case_id"]].append(value)
            result[label][metric] = dict(mean=mean(mean(seeds) for seeds in values.values()),
                                          cases=len(values))
    return result


def score(manifest):
    root = manifest.parent
    groups = defaultdict(dict)
    strata = {}
    references = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id, seed = str(row["case_id"]), int(row["seed"])
        arm, direction = row["arm"], row["texture_direction"]
        sample_type = row.get("sample_type", "patterned")
        if sample_type not in ("patterned", "clean_control") or strata.setdefault(case_id, sample_type) != sample_type:
            raise ValueError(f"{case_id}: sample_type 无效或不一致")
        signature = tuple(row.get(key) for key in ("mask", "sketch", "structure_mask", "period"))
        if references.setdefault(case_id, signature) != signature:
            raise ValueError(f"{case_id}: 所有臂与种子的评分 mask、结构参考及周期必须固定")
        cells = groups[(case_id, seed)]
        if arm not in ARMS or direction not in DIRECTIONS or (arm, direction) in cells:
            raise ValueError(f"{case_id}/{seed}: 消融臂或方向无效/重复 {arm}/{direction}")
        cells[arm, direction] = row
    if not groups:
        raise ValueError("评分清单为空")

    images, pairs = [], []
    for (case_id, seed), cells in sorted(groups.items()):
        if len(cells) != len(ARMS) * len(DIRECTIONS):
            raise ValueError(f"{case_id}/{seed}: 需要完整 A/B/C/D × 横竖八格")
        basis = cells["A", "horizontal"]
        with Image.open(root / basis["mask"]) as img:
            mask = np.asarray(img.convert("L")) > 127
        height, width = mask.shape
        annotated = bool(basis.get("structure_mask"))
        path = basis["structure_mask"] if annotated else basis["sketch"]
        with Image.open(root / path) as img:
            gray = np.asarray(img.convert("L"))
        if gray.shape != mask.shape or not mask.any():
            raise ValueError(f"{case_id}/{seed}: mask 为空或参考图尺寸不一致")
        protected = ((gray > 0) if annotated else (gray < 128)) & morphology(mask, 9)
        period = float(basis["period"])
        if not np.isfinite(period) or period < 3:
            raise ValueError(f"{case_id}/{seed}: period 须为原生像素单位且不少于 3")
        tiles = tiles_inside(mask, size=round(4 * period))
        identity = dict(case_id=case_id, seed=seed, sample_type=strata[case_id])
        pair = {**identity, "arms": {}}
        for arm in ARMS:
            measurements = []
            for direction in DIRECTIONS:
                path = cells[arm, direction]["image"]
                with Image.open(root / path) as img:
                    image = img.convert("RGB")
                if image.size != (width, height):
                    raise ValueError(f"{path}: 生成图与评分 mask 的原生尺寸不一致")
                item = stripe_response(image, tiles, period, direction)
                item.update(shape_response(image, mask, protected))
                measurements.append(item)
                images.append({**identity, "arm": arm, "texture_direction": direction, "image": path,
                               "structure_source": "structure_mask" if annotated else "sketch_proxy",
                               "protected_pixels": int(protected.sum()), **item})
            pair["arms"][arm] = measure_pair(*measurements)
        pairs.append(pair)
    return dict(mask_backend=mask_backend_info(), images=images, paired=pairs,
                thresholds=dict(min_rms=.02, min_dominant_band_fraction=.15, min_direction_margin=.3,
                                direction_cone_degrees=22.5, period_unit="image pixels"),
                strata={label: summarize([p for p in pairs if p["sample_type"] == label])
                        for label in sorted(set(strata.values()))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="每个身份和种子八行的 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="输出 JSON")
    args = parser.parse_args()
    result = score(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
