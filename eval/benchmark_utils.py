import csv
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone

from eval.eval_utils import json_safe


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def git_commit_hash(default="unknown"):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return out or default
    except Exception:
        return default


def write_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows):
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def write_markdown_table(path: str, rows, title="Results"):
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        content = f"# {title}\n\n(no rows)\n"
    else:
        keys = sorted({k for r in rows for k in r.keys()})
        header = "| " + " | ".join(keys) + " |"
        sep = "| " + " | ".join(["---"] * len(keys)) + " |"
        lines = [f"# {title}", "", header, sep]
        for r in rows:
            safe_row = json_safe(r)
            lines.append(
                "| "
                + " | ".join(
                    "—" if safe_row.get(k) is None else str(safe_row.get(k, ""))
                    for k in keys
                )
                + " |"
            )
        content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _split_item(item, dataset_index, sample_position):
    return {
        "sample_id": f"{sample_position:06d}",
        "idx": dataset_index,
        "prompt": item["caption"] if isinstance(item["caption"], str) else item["caption"][0],
        "sketch": item["sketch"],
        "texture": item.get("texture", item.get("color", item["cloth"])),
        "target": item.get("cloth"),
        "mask": item.get("mask", None),
    }


def create_or_load_fixed_split(
    dataset_json_path: str,
    split_path: str,
    num_samples: int = 16,
    seed: int = 42,
    sample_id_start: int = 0,
    sample_id_end: int = None,
):
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    requested_end = num_samples if sample_id_end is None else sample_id_end
    if sample_id_start < 0 or requested_end <= sample_id_start:
        raise ValueError(
            f"invalid sample range: start={sample_id_start}, end={requested_end}"
        )
    if requested_end > len(data):
        raise ValueError(
            f"dataset only contains {len(data)} samples, but sample_id_end={requested_end}"
        )

    split = []
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

    # Keep every existing sample in its original position so previously generated
    # results retain the same uid and seed association.
    normalized = []
    seen_indices = set()
    for position, sample in enumerate(split):
        dataset_index = int(sample.get("idx", sample.get("dataset_index", -1)))
        if dataset_index < 0 or dataset_index >= len(data):
            raise ValueError(
                f"invalid dataset index in existing split at position {position}: "
                f"{dataset_index}"
            )
        if dataset_index in seen_indices:
            raise ValueError(
                f"duplicate dataset index in existing split: {dataset_index}"
            )
        seen_indices.add(dataset_index)
        normalized_sample = dict(sample)
        normalized_sample["sample_id"] = f"{position:06d}"
        normalized_sample["idx"] = dataset_index
        normalized.append(normalized_sample)
    split = normalized

    # Extend the old split using the same deterministic shuffled dataset order.
    # Existing entries are never reordered or replaced.
    rnd = random.Random(seed)
    idxs = list(range(len(data)))
    rnd.shuffle(idxs)
    for dataset_index in idxs:
        if len(split) >= requested_end:
            break
        if dataset_index in seen_indices:
            continue
        split.append(_split_item(data[dataset_index], dataset_index, len(split)))
        seen_indices.add(dataset_index)

    if len(split) < requested_end:
        raise RuntimeError(
            f"could only build {len(split)} fixed samples; requested {requested_end}"
        )

    write_json(split_path, split)
    return split[sample_id_start:requested_end]


def sample_uid(sample):
    key = f"{sample.get('idx','na')}::{sample.get('sketch','')}::{sample.get('texture','')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def write_manifest(path: str, payload: dict):
    payload = dict(payload)
    payload.setdefault("timestamp_utc", utc_timestamp())
    payload.setdefault("git_commit", git_commit_hash())
    write_json(path, payload)
