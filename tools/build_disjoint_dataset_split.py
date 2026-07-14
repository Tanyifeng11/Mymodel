#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import write_json


def benchmark_item(item, dataset_index, position):
    caption = item["caption"] if isinstance(item["caption"], str) else item["caption"][0]
    return {
        "sample_id": f"{position:06d}",
        "idx": dataset_index,
        "prompt": caption,
        "sketch": item["sketch"],
        "texture": item.get("texture", item.get("color", item["cloth"])),
        "target": item.get("cloth"),
        "mask": item.get("mask"),
    }


def normalized_path(value):
    return os.path.normcase(os.path.normpath(value or ""))


def parse_args():
    parser = argparse.ArgumentParser(description="Build future train/validation files with zero overlap")
    parser.add_argument("--master_json", required=True)
    parser.add_argument("--output_train_json", required=True)
    parser.add_argument("--output_val_json", required=True)
    parser.add_argument("--output_split_json", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--val_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--existing_benchmark_split", default=None)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = [args.output_train_json, args.output_val_json, args.output_split_json, args.report_json]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist; pass --overwrite 1: {existing}")
    with open(args.master_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if args.val_count <= 0 or args.val_count >= len(data):
        raise ValueError(f"val_count must be in [1, {len(data) - 1}]")

    indices = list(range(len(data)))
    random.Random(args.seed).shuffle(indices)
    val_indices = indices[: args.val_count]
    train_indices = indices[args.val_count :]
    train_data = [data[index] for index in train_indices]
    val_data = [data[index] for index in val_indices]
    benchmark_split = [benchmark_item(data[index], index, position) for position, index in enumerate(val_indices)]

    write_json(args.output_train_json, train_data)
    write_json(args.output_val_json, val_data)
    write_json(args.output_split_json, benchmark_split)

    train_targets = {normalized_path(data[index].get("cloth")) for index in train_indices}
    val_targets = {normalized_path(data[index].get("cloth")) for index in val_indices}
    current_overlap = []
    if args.existing_benchmark_split and os.path.isfile(args.existing_benchmark_split):
        with open(args.existing_benchmark_split, "r", encoding="utf-8") as handle:
            current = json.load(handle)
        for sample in current:
            target = normalized_path(sample.get("target"))
            index = sample.get("idx", sample.get("dataset_index"))
            if target in train_targets or index in train_indices:
                current_overlap.append({
                    "sample_id": sample.get("sample_id"),
                    "idx": index,
                    "target": sample.get("target"),
                })

    report = {
        "master_json": args.master_json,
        "seed": args.seed,
        "num_total": len(data),
        "num_train": len(train_data),
        "num_validation": len(val_data),
        "train_validation_target_overlap": len(train_targets & val_targets),
        "existing_benchmark_train_overlap_count": len(current_overlap),
        "existing_benchmark_train_overlap": current_overlap,
        "retroactively_makes_existing_models_unseen": False,
        "valid_for_models_trained_from_output_train_json": True,
    }
    write_json(args.report_json, report)
    print(report)


if __name__ == "__main__":
    main()
