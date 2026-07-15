#!/usr/bin/env python3
import argparse
import json
import os


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


def main():
    parser = argparse.ArgumentParser(
        description="Build the text-conditioned BF independent test manifest."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["top", "outwear", "pants", "dress"],
    )
    args = parser.parse_args()

    dataset = []
    split = []
    counts = {}

    for category in args.classes:
        category_root = os.path.join(args.data_root, category)
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
                "texture": relative_path(category, "texture", texture[stem]),
                "cloth": relative_path(category, "gt", gt[stem]),
                "sketch": relative_path(category, "sketch", sketch[stem]),
                "category": category,
                "filename": gt[stem],
            }
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

    for path, payload in ((args.dataset_json, dataset), (args.split_path, split)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {"total": len(dataset), "classes": counts, "excluded": ["bag"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
