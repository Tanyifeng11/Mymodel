import argparse
import csv
import json
import math
import re
from pathlib import Path


TEXTURE_KEYS = (
    "g_texture",
    "texture_gate",
    "balanced_texture_gate",
    "last_balanced_texture_gate",
)
PALETTE_KEYS = (
    "g_palette",
    "palette_gate",
    "balanced_palette_gate",
    "last_balanced_palette_gate",
)
LAYER_KEYS = ("layer_group", "group", "layer")
TIMESTEP_KEYS = ("timestep", "balanced_gate_timestep", "t")


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _first_float(row, keys):
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _first_text(row, keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normal_layer_group(value):
    value = value.lower()
    if "semantic" in value:
        return "semantic"
    if "detail" in value:
        return "detail"
    if "middle" in value:
        return "middle"
    if value in {"all", "mid", "middle/all"}:
        return "all"
    return value or "unknown"


def _timestep_bin(timestep):
    if timestep is None:
        return "unknown"
    # Diffusion timesteps are usually high -> early denoising, low -> late denoising.
    if timestep >= 667:
        return "early"
    if timestep >= 334:
        return "middle"
    return "late"


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def _read_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_json(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("rows", "records", "gate_rows", "data"):
            rows = data.get(key)
            if isinstance(rows, list):
                yield from rows
                return
        yield data


def _read_log(path):
    pattern = re.compile(
        r"balanced_texture_gate=(?P<texture>[-+0-9.eE]+).*?"
        r"balanced_palette_gate=(?P<palette>[-+0-9.eE]+)"
    )
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                yield {
                    "g_texture": match.group("texture"),
                    "g_palette": match.group("palette"),
                    "layer_group": "unknown",
                    "timestep": "",
                }


def _iter_rows(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _read_csv(path)
    elif suffix == ".jsonl":
        yield from _read_jsonl(path)
    elif suffix == ".json":
        yield from _read_json(path)
    elif suffix in {".log", ".txt"}:
        yield from _read_log(path)


def _row_to_record(row, source):
    g_texture = _first_float(row, TEXTURE_KEYS)
    g_palette = _first_float(row, PALETTE_KEYS)
    if g_texture is None and g_palette is None:
        return None
    layer_group = _normal_layer_group(_first_text(row, LAYER_KEYS))
    timestep = _first_float(row, TIMESTEP_KEYS)
    return {
        "source": str(source),
        "sample_id": _first_text(row, ("sample_id", "sample", "id")),
        "layer_name": _first_text(row, ("layer_name", "name", "processor")),
        "layer_group": layer_group,
        "timestep": timestep,
        "timestep_bin": _timestep_bin(timestep),
        "g_texture": g_texture,
        "g_palette": g_palette,
    }


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def _ratio(values, threshold):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(v > threshold for v in values) / len(values)


def _summarize(records):
    g_texture = [r["g_texture"] for r in records]
    g_palette = [r["g_palette"] for r in records]
    texture_stats = _stats(g_texture)
    palette_stats = _stats(g_palette)
    return {
        "num_records": len(records),
        "g_texture_mean": texture_stats["mean"],
        "g_texture_std": texture_stats["std"],
        "g_texture_min": texture_stats["min"],
        "g_texture_max": texture_stats["max"],
        "g_palette_mean": palette_stats["mean"],
        "g_palette_std": palette_stats["std"],
        "g_palette_min": palette_stats["min"],
        "g_palette_max": palette_stats["max"],
        "g_texture_gt_1_05_ratio": _ratio(g_texture, 1.05),
        "g_texture_gt_1_10_ratio": _ratio(g_texture, 1.10),
        "g_texture_gt_1_15_ratio": _ratio(g_texture, 1.15),
        "g_palette_gt_1_10_ratio": _ratio(g_palette, 1.10),
        "g_palette_gt_1_20_ratio": _ratio(g_palette, 1.20),
    }


def _group(records, key):
    buckets = {}
    for record in records:
        buckets.setdefault(record.get(key) or "unknown", []).append(record)
    return {name: _summarize(rows) for name, rows in sorted(buckets.items())}


def _find_sources(root):
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower in {"gate_stats_e4a.csv", "gate_stats_e4a.json"}:
            continue
        if any(token in lower for token in ("gate", "trace", "debug", "attn")):
            if path.suffix.lower() in {".csv", ".json", ".jsonl", ".log", ".txt"}:
                candidates.append(path)
    return candidates


def _write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build(input_paths, search_root, out_dir):
    sources = [Path(p) for p in input_paths]
    if not sources and search_root:
        sources = _find_sources(Path(search_root))

    records = []
    scanned = []
    for source in sources:
        scanned.append(str(source))
        if not source.exists() or not source.is_file():
            continue
        for row in _iter_rows(source):
            record = _row_to_record(row, source)
            if record is not None:
                records.append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gate_stats_e4a.csv"
    json_path = out_dir / "gate_stats_e4a.json"

    if not records:
        result = {
            "status": "unavailable",
            "reason": (
                "eval_outputs 中没有包含 g_texture/g_palette 或 "
                "balanced_texture_gate/balanced_palette_gate 的原始 gate trace。"
                "生成图像、metrics_per_sample 和 summary metrics 不能反推出运行时 gate。"
            ),
            "scanned_sources": scanned,
            "required_input_columns": {
                "texture_gate": list(TEXTURE_KEYS),
                "palette_gate": list(PALETTE_KEYS),
                "optional_layer_group": list(LAYER_KEYS),
                "optional_timestep": list(TIMESTEP_KEYS),
            },
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(csv_path, [{"status": result["status"], "reason": result["reason"]}], ["status", "reason"])
        return result

    summary = {
        "status": "ok",
        "sources": sorted(set(r["source"] for r in records)),
        "overall": _summarize(records),
        "by_layer_group": _group(records, "layer_group"),
        "by_timestep_bin": _group(records, "timestep_bin"),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    rows.append({"section": "overall", "name": "all", **summary["overall"]})
    for section, values in (
        ("layer_group", summary["by_layer_group"]),
        ("timestep_bin", summary["by_timestep_bin"]),
    ):
        for name, stats in values.items():
            rows.append({"section": section, "name": name, **stats})
    fieldnames = [
        "section",
        "name",
        "num_records",
        "g_texture_mean",
        "g_texture_std",
        "g_texture_min",
        "g_texture_max",
        "g_palette_mean",
        "g_palette_std",
        "g_palette_min",
        "g_palette_max",
        "g_texture_gt_1_05_ratio",
        "g_texture_gt_1_10_ratio",
        "g_texture_gt_1_15_ratio",
        "g_palette_gt_1_10_ratio",
        "g_palette_gt_1_20_ratio",
    ]
    _write_csv(csv_path, rows, fieldnames)
    _write_csv(
        out_dir / "gate_stats_e4a_records.csv",
        records,
        ["source", "sample_id", "layer_name", "layer_group", "timestep", "timestep_bin", "g_texture", "g_palette"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build E4a balanced gate stats from runtime gate traces.")
    parser.add_argument("--input", nargs="*", default=[], help="Gate trace CSV/JSON/JSONL/log files.")
    parser.add_argument("--search_root", default="eval_outputs", help="Directory to auto-scan when --input is empty.")
    parser.add_argument("--out_dir", default="eval_outputs/e4a_diagnosis")
    args = parser.parse_args()
    result = build(args.input, args.search_root, Path(args.out_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
