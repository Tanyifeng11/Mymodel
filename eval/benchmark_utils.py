import csv
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone


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
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
            writer.writerow(row)


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
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def create_or_load_fixed_split(dataset_json_path: str, split_path: str, num_samples: int = 16, seed: int = 42):
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rnd = random.Random(seed)
    idxs = list(range(len(data)))
    rnd.shuffle(idxs)
    chosen = idxs[: min(num_samples, len(idxs))]
    split = []
    for i in chosen:
        item = data[i]
        split.append(
            {
                "idx": i,
                "prompt": item["caption"] if isinstance(item["caption"], str) else item["caption"][0],
                "sketch": item["sketch"],
                "texture": item.get("texture", item.get("color", item["cloth"])),
                "target": item.get("cloth"),
                "mask": item.get("mask", None),
            }
        )
    write_json(split_path, split)
    return split


def sample_uid(sample):
    key = f"{sample.get('idx','na')}::{sample.get('sketch','')}::{sample.get('texture','')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def write_manifest(path: str, payload: dict):
    payload = dict(payload)
    payload.setdefault("timestamp_utc", utc_timestamp())
    payload.setdefault("git_commit", git_commit_hash())
    write_json(path, payload)
