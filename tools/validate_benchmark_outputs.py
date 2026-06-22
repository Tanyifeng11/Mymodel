#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter

from PIL import Image


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_experiment(exp_dir, expected_count, required_size=None):
    errors = []
    diagnostics_path = os.path.join(exp_dir, "diagnostics.json")
    rows_path = os.path.join(exp_dir, "metrics_per_sample.json")
    if not os.path.isfile(diagnostics_path):
        return [f"missing diagnostics: {diagnostics_path}"]
    if not os.path.isfile(rows_path):
        return [f"missing per-sample metrics: {rows_path}"]

    diagnostics = load_json(diagnostics_path)
    rows = load_json(rows_path)
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    duplicates = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if not sample_id or count > 1
    )

    if len(rows) != expected_count:
        errors.append(f"row count={len(rows)}, expected={expected_count}")
    if diagnostics.get("final_generated_count") != expected_count:
        errors.append(
            "final_generated_count="
            f"{diagnostics.get('final_generated_count')}, "
            f"expected={expected_count}"
        )
    if diagnostics.get("missing_sample_ids"):
        errors.append(
            f"missing sample ids={diagnostics['missing_sample_ids']}"
        )
    if diagnostics.get("duplicate_sample_ids"):
        errors.append(
            f"diagnostic duplicate ids={diagnostics['duplicate_sample_ids']}"
        )
    if duplicates:
        errors.append(f"duplicate row sample ids={duplicates}")

    for key in ("gen_path", "target_path", "texture_path", "sketch_path"):
        missing = [
            row.get("sample_id")
            for row in rows
            if not row.get(key) or not os.path.isfile(row[key])
        ]
        if missing:
            errors.append(f"{key} missing for sample ids={missing[:20]}")

    if required_size:
        expected_size = (required_size, required_size)
        invalid = []
        for row in rows:
            for key in ("gen_path", "target_path"):
                path = row.get(key)
                if not path or not os.path.isfile(path):
                    continue
                with Image.open(path) as image:
                    if image.size != expected_size or image.mode != "RGB":
                        invalid.append(
                            f"{row.get('sample_id')}:{key}:"
                            f"{image.mode}:{image.size}"
                        )
        if invalid:
            errors.append(f"invalid resized images={invalid[:20]}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", required=True)
    parser.add_argument("--experiment_names", required=True)
    parser.add_argument("--expected_count", type=int, default=500)
    parser.add_argument("--required_size", type=int, default=0)
    args = parser.parse_args()

    failed = False
    for name in [item.strip() for item in args.experiment_names.split(",")]:
        exp_dir = os.path.join(args.experiments_dir, name)
        errors = validate_experiment(
            exp_dir,
            expected_count=args.expected_count,
            required_size=args.required_size or None,
        )
        if errors:
            failed = True
            print(f"[validate] ERROR {name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(
                f"[validate] OK {name}: "
                f"{args.expected_count} unique samples"
            )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
