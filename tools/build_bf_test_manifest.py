#!/usr/bin/env python3
import argparse
import json
import os
import random


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def files_by_stem(directory, extensions):
    files = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        stem, extension = os.path.splitext(name)
        if os.path.isfile(path) and extension.lower() in extensions:
            if stem in files:
                raise ValueError(f"duplicate stem in {directory}: {stem}")
            files[stem] = name
    return files


def relative_path(*parts):
    return "/".join(parts)


def load_training_target_stems(path):
    if not path:
        return set()
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    stems = set()
    for item in data:
        target = item.get("cloth") or item.get("target")
        if target:
            stems.add(os.path.splitext(os.path.basename(target))[0])
    return stems


def resolve_categories(data_root, layout, classes):
    flat_dirs_exist = all(
        os.path.isdir(os.path.join(data_root, name))
        for name in ("gt", "sketch", "texture", "text")
    )
    if layout == "flat" and not flat_dirs_exist:
        raise FileNotFoundError(f"flat BF layout is incomplete: {data_root}")
    if layout == "flat" or (layout == "auto" and flat_dirs_exist):
        return [("validation", data_root, "")]
    return [(category, os.path.join(data_root, category), category) for category in classes]


def main():
    parser = argparse.ArgumentParser(
        description="Build the text-conditioned BF independent test manifest."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument(
        "--layout",
        choices=["auto", "flat", "categorized"],
        default="auto",
        help="Auto-detect validation/{gt,sketch,texture,text} or use category subdirectories.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["top", "outwear", "pants", "dress"],
    )
    parser.add_argument(
        "--split_count",
        type=int,
        default=0,
        help="Number of deterministic shuffled samples in split_path; 0 keeps all samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_json",
        default=None,
        help="Optional training JSON. Validation target stems must not overlap it.",
    )
    args = parser.parse_args()

    dataset = []
    split = []
    counts = {}

    categories = resolve_categories(args.data_root, args.layout, args.classes)
    validation_target_stems = set()
    for category, category_root, relative_prefix in categories:
        required_dirs = {
            "gt": os.path.join(category_root, "gt"),
            "sketch": os.path.join(category_root, "sketch"),
            "texture": os.path.join(category_root, "texture"),
            "text": os.path.join(category_root, "text"),
        }
        for name, directory in required_dirs.items():
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"missing {category}/{name}: {directory}")

        gt = files_by_stem(required_dirs["gt"], IMAGE_EXTENSIONS)
        sketch = files_by_stem(required_dirs["sketch"], IMAGE_EXTENSIONS)
        texture = files_by_stem(required_dirs["texture"], IMAGE_EXTENSIONS)
        text = files_by_stem(required_dirs["text"], {".txt"})
        stem_sets = {
            "gt": set(gt),
            "sketch": set(sketch),
            "texture": set(texture),
            "text": set(text),
        }
        expected = stem_sets["gt"]
        mismatches = {
            name: len(expected.symmetric_difference(stems))
            for name, stems in stem_sets.items()
            if stems != expected
        }
        if mismatches:
            raise RuntimeError(
                f"unpaired files in {category}: counts="
                f"{ {name: len(stems) for name, stems in stem_sets.items()} }, "
                f"symmetric_differences={mismatches}"
            )

        for stem in sorted(expected):
            text_path = os.path.join(required_dirs["text"], text[stem])
            with open(text_path, "r", encoding="utf-8-sig") as handle:
                caption = " ".join(handle.read().strip().split())
            if not caption:
                raise ValueError(f"empty caption: {text_path}")

            dataset_index = len(dataset)
            item = {
                "caption": caption,
                "texture": relative_path(relative_prefix, "texture", texture[stem]).lstrip("/"),
                "cloth": relative_path(relative_prefix, "gt", gt[stem]).lstrip("/"),
                "sketch": relative_path(relative_prefix, "sketch", sketch[stem]).lstrip("/"),
                "category": category,
                "filename": gt[stem],
            }
            validation_target_stems.add(stem)
            dataset.append(item)
            split.append(
                {
                    "sample_id": f"{dataset_index:06d}",
                    "idx": dataset_index,
                    "prompt": caption,
                    "sketch": item["sketch"],
                    "texture": item["texture"],
                    "target": item["cloth"],
                    "mask": None,
                    "category": category,
                    "filename": gt[stem],
                }
            )
        counts[category] = len(expected)

    training_target_stems = load_training_target_stems(args.train_json)
    overlap = sorted(validation_target_stems & training_target_stems)
    if overlap:
        raise RuntimeError(
            "validation/training target stem overlap detected: "
            f"count={len(overlap)}, examples={overlap[:20]}"
        )

    if args.split_count < 0 or args.split_count > len(split):
        raise ValueError(
            f"split_count must be in [0, {len(split)}], got {args.split_count}"
        )
    if args.split_count:
        indices = list(range(len(split)))
        random.Random(args.seed).shuffle(indices)
        split = [split[index] for index in indices[: args.split_count]]
        for position, sample in enumerate(split):
            sample["sample_id"] = f"{position:06d}"

    for path, payload in ((args.dataset_json, dataset), (args.split_path, split)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "total": len(dataset),
                "split_count": len(split),
                "classes": counts,
                "layout": "flat" if categories[0][2] == "" else "categorized",
                "train_overlap_count": len(overlap),
                "excluded": ["bag"] if categories[0][2] else [],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
